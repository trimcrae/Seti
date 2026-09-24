"""Rebuild ``candidates.json`` and ``summary.json`` from EVERY per-star record.

MEASURED, 2026-09-23.  Five ``vetstar`` runs (35802589042, 35802594315,
35802596474, 35802601869, 35802603859) vetted five different stars in
parallel.  Each checked out the same ``candidates.json``, demoted its own
star in it, and committed it; ``scripts/commit_results.sh`` is last-writer-wins
by design, so the file that landed carried ONE of the five demotions and
``vetstar.json`` carried one of the five reports.  Three demotions were lost
and ``summary.json`` read ``n_interest: 4`` for stars the vet had already
explained -- and ``VETSTAR_DEMOTED_2; VETSTAR_DEMOTED_1`` in one verdict,
because the invocation's own count was appended beside the count derived from
the records.

The remedy is the rule ``rebuild_summary`` already follows for the counts,
extended to the demotions themselves: **state, not increments.**  Every
downstream verdict lives in its own file --

* ``stars_vetted.csv``      -- the assess stage's tier for every star (the
  baseline; it already carries the assess-time vetoes, including the
  aperture-scale contamination cone);
* ``redetect.json``         -- the light-curve verdict per shortlisted star;
* ``vetstar_<mission>_<id>.json`` -- one file per single-star vet, so two
  runs on two stars never write the same path;

and :func:`reconcile_all` recomputes each shortlisted star's tier from those
files every time.  Running it twice, or on a checkout that has since gained a
third star's vet, gives the same answer as running it once on everything.
Demotion only ever removes a claim: nothing here can promote a star above the
tier assess gave it.
"""

from __future__ import annotations

import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

#: per-star vet files; the legacy single ``vetstar.json`` is read only for a
#: star that has no per-star file of its own
VETSTAR_GLOB = "vetstar_*.json"
LEGACY_VETSTAR = "vetstar.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def vetstar_filename(star_key: str, *, suffix: str = ".json", prefix: str = "vetstar_") -> str:
    """``kepler:5879574`` -> ``vetstar_kepler_5879574.json``.

    Anything but letters, digits, ``.`` and ``-`` becomes ``_`` so the name is
    a safe path on every filesystem and in a git pathspec.
    """
    safe = re.sub(r"[^A-Za-z0-9.\-]+", "_", str(star_key)).strip("_")
    return f"{prefix}{safe}{suffix}"


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def load_vetstar_reports(out: Path) -> dict[str, dict]:
    """``{star_key: report}`` from every per-star vet file (plus the legacy
    file for a star with no per-star one).  A file that does not parse is
    skipped, never guessed at."""
    out = Path(out)
    reps: dict[str, dict] = {}
    for fp in sorted(glob.glob(str(out / VETSTAR_GLOB))):
        name = Path(fp).name
        if name.startswith("vetstar_fold"):
            continue
        try:
            d = json.loads(Path(fp).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("star_key"):
            d.setdefault("_file", name)
            prev = reps.get(str(d["star_key"]))
            # two files for one star: the later generated_utc is the later word
            if prev is None or str(d.get("generated_utc") or "") >= str(
                    prev.get("generated_utc") or ""):
                reps[str(d["star_key"])] = d
    lp = out / LEGACY_VETSTAR
    if lp.exists():
        try:
            d = json.loads(lp.read_text())
        except (OSError, ValueError):
            d = None
        if isinstance(d, dict) and d.get("star_key") and str(d["star_key"]) not in reps:
            d.setdefault("_file", LEGACY_VETSTAR)
            reps[str(d["star_key"])] = d
    return reps


def _assess_rows(out: Path) -> dict[str, dict]:
    vp = Path(out) / "stars_vetted.csv"
    if not vp.exists():
        return {}
    try:
        df = pd.read_csv(vp, dtype={"star_key": str, "star_id": str},
                         usecols=lambda c: c in ("star_key", "tier", "first_veto", "flags"))
    except (OSError, ValueError):
        return {}
    rows = {}
    for rec in df.to_dict(orient="records"):
        k = str(rec.get("star_key"))
        if not k or k == "nan":
            continue
        rows[k] = {kk: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                   for kk, v in rec.items()}
    return rows


def reconcile_all(out: Path, *, stage: str = "reconcile") -> dict:
    """Recompute every shortlisted star's tier from all per-star records and
    rebuild ``summary.json`` from the result.  Returns a small report."""
    from .redetect import RECONCILE_KEYS, _lightcurve_veto, rebuild_summary
    from .vetstar import vetstar_veto

    out = Path(out)
    res: dict = {"status": "NO_SUMMARY", "stage": stage}
    sp, cp = out / "summary.json", out / "candidates.json"
    if not sp.exists():
        return res
    try:
        summary = json.loads(sp.read_text())
    except (OSError, ValueError) as exc:
        res["status"] = f"SUMMARY_UNREADABLE:{exc!r}"[:200]
        return res

    assess = _assess_rows(out)

    red: dict = {}
    rd_by_key: dict[str, dict] = {}
    rp = out / "redetect.json"
    if rp.exists():
        try:
            red = json.loads(rp.read_text())
        except (OSError, ValueError):
            red = {}
    for t in (red.get("targets") or []):
        k = str(t.get("star_key"))
        if k:
            rd_by_key[k] = {kk: t.get(kk) for kk in RECONCILE_KEYS}
    vets = load_vetstar_reports(out)
    from .aperture import aperture_veto

    ap: dict = {}
    app = out / "aperture.json"
    if app.exists():
        try:
            ap = json.loads(app.read_text())
        except (OSError, ValueError):
            ap = {}
    ap_per = ap.get("per_star") or {}

    demoted_ap: list[str] = []
    demoted_lc: list[str] = []
    demoted_vet: list[str] = []
    n_rows = 0
    cj = None
    if cp.exists():
        try:
            cj = json.loads(cp.read_text())
        except (OSError, ValueError):
            cj = None
    if isinstance(cj, dict):
        for bucket in ("candidates", "watch"):
            for row in cj.get(bucket) or []:
                n_rows += 1
                key = str(row.get("star_key"))
                base = assess.get(key)
                # restore the assess verdict, then lay every downstream
                # verdict over it afresh -- never trust a prior invocation's
                # in-place edit
                if base is not None:
                    for fld in ("tier", "first_veto", "flags"):
                        if fld in base:
                            row[fld] = base.get(fld)
                    if "first_veto" not in base and str(row.get("first_veto") or "") \
                            .startswith(("vet_", "catalogue_epochs_absent",
                                         "photometric_oscillation")):
                        row["first_veto"] = None
                    if "flags" not in base:
                        row["flags"] = ";".join(
                            f for f in str(row.get("flags") or "").split(";")
                            if f and not f.startswith("vet_")
                            and f not in ("catalogue_epochs_absent",
                                          "photometric_oscillation"))
                # with no baseline row the star is left as it stands
                # the sky around the star first: it is the assess-time question
                arec = ap_per.get(key)
                if arec is not None:
                    row["aperture"] = {k: arec.get(k) for k in (
                        "identity_reached", "aperture_reached", "identity_hit",
                        "identity_detail", "contaminating", "contamination_detail")}
                    row["aperture"]["n_neighbours"] = len(arec.get("neighbours") or [])
                    aveto = aperture_veto(arec)
                    if aveto and str(row.get("tier")) in ("candidate", "interest"):
                        row["tier"] = "none"
                        row["first_veto"] = aveto
                        row["flags"] = ";".join(
                            [f for f in str(row.get("flags") or "").split(";") if f
                             and f != "None" and f != "nan"] + [aveto])
                        demoted_ap.append(f"{key}:{aveto}")
                else:
                    row.pop("aperture", None)
                rd = rd_by_key.get(key)
                row["redetect"] = rd or {"status": "not_attempted"}
                veto = _lightcurve_veto(rd)
                if veto and str(row.get("tier")) in ("candidate", "interest"):
                    row["tier"] = "none"
                    row["first_veto"] = veto
                    row["flags"] = ";".join(
                        [f for f in str(row.get("flags") or "").split(";") if f
                         and f != "None" and f != "nan"] + [veto])
                    demoted_lc.append(f"{key}:{veto}")
                rep = vets.get(key)
                if rep is not None:
                    row["vetstar"] = {k: rep.get(k) for k in (
                        "verdict", "generated_utc", "_file", "period", "catalogue",
                        "fetch_status", "n_flares_redetected", "n_catalogue_epochs",
                        "unreached", "surviving_explanations", "neighbour_period_matches",
                        "recovered_from")}
                    vveto = vetstar_veto(rep)
                    if vveto and str(row.get("tier")) in ("candidate", "interest"):
                        row["tier"] = "none"
                        row["first_veto"] = vveto
                        row["flags"] = ";".join(
                            [f for f in str(row.get("flags") or "").split(";") if f
                             and f != "None" and f != "nan"] + [vveto])
                        demoted_vet.append(f"{key}:{vveto}")
                else:
                    row.pop("vetstar", None)

    # the stage blocks describe the stage files, dated by the files themselves
    if red:
        summary["redetect"] = {
            "generated_utc": red.get("generated_utc"),
            "verdict": red.get("verdict"),
            "n_demoted": len(demoted_lc), "demoted": demoted_lc,
            "vetoes": ["catalogue_epochs_absent", "photometric_oscillation"],
            "per_star": rd_by_key,
            "note": ((summary.get("redetect") or {}).get("note")
                     or "the light curve has the last word; see redetect.json"),
        }
    if ap:
        summary["aperture_contamination"] = {
            k: ap.get(k) for k in (
                "generated_utc", "radius_arcsec", "identity_radius_arcsec", "sources",
                "aperture_tol", "identity_tol", "shortlist_definition", "n_shortlist",
                "n_positioned", "positions", "frac_identity_reached", "frac_aperture_reached",
                "n_flagged", "flagged", "flagged_detail", "n_with_variable_neighbour", "note")}
        summary["aperture_contamination"]["n_demoted"] = len(demoted_ap)
        summary["aperture_contamination"]["demoted"] = demoted_ap
        deg = [d for d in (summary.get("degraded") or [])
               if not str(d).startswith("aperture_stage:")]
        fa = ap.get("frac_aperture_reached")
        fi = ap.get("frac_identity_reached")
        if not (isinstance(fa, (int, float)) and fa >= 1.0
                and isinstance(fi, (int, float)) and fi >= 1.0):
            deg.append(f"aperture_stage:identity_reached_{float(fi or 0):.2f}"
                       f"_aperture_reached_{float(fa or 0):.2f}_of_shortlist")
        summary["degraded"] = deg
        summary.setdefault("provenance", {})["aperture_generated_utc"] = ap.get("generated_utc")
    summary["vetstar"] = {
        "n_vetted": len(vets),
        "n_demoted": len(demoted_vet), "demoted": demoted_vet,
        "per_star": {k: {"verdict": v.get("verdict"), "veto": vetstar_veto(v),
                         "generated_utc": v.get("generated_utc"), "file": v.get("_file"),
                         "recovered_from": v.get("recovered_from")}
                     for k, v in sorted(vets.items())},
        "note": ("one file per vetted star (vetstar_<mission>_<id>.json); every file present "
                 "is re-applied on every reconciliation, so parallel vets cannot overwrite "
                 "one another's demotions.  A count after vetting is not an occurrence limit "
                 "and per CLAUDE.md is not written up"),
    }
    if vets:
        summary.setdefault("provenance", {})["vetstar_generated_utc"] = max(
            str(v.get("generated_utc") or "") for v in vets.values()) or None

    if isinstance(cj, dict):
        cp.write_text(json.dumps(cj, indent=2, default=_json_default))
    rebuild_summary(out, summary, stage=stage)
    if isinstance(cj, dict):
        # candidates.json names the same verdict and the same moment as summary
        cj["verdict"] = summary.get("verdict")
        cj["generated_utc"] = summary.get("generated_utc")
        cj["assess_generated_utc"] = (summary.get("provenance") or {}).get(
            "assess_generated_utc")
        cp.write_text(json.dumps(cj, indent=2, default=_json_default))
    sp.write_text(json.dumps(summary, indent=2, default=_json_default))
    res.update({"status": "OK", "n_rows": n_rows, "demoted_aperture": demoted_ap,
                "demoted_lightcurve": demoted_lc,
                "demoted_vetstar": demoted_vet, "n_vetstar_files": len(vets),
                "verdict": summary.get("verdict")})
    return res


def check_consistency(out: Path) -> list[str]:
    """Every way ``summary.json`` can disagree with its own records, as a list
    of problems (empty = consistent).  Used by the tests and by the workflow
    after every commit-back."""
    from .redetect import current_tiers, demotions_by_stage

    out = Path(out)
    probs: list[str] = []
    try:
        s = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"summary unreadable: {exc!r}"]
    tiers, prov = current_tiers(out)
    counts = {t: 0 for t in ("none", "watch", "interest", "candidate")}
    for t in tiers.values():
        counts[t] = counts.get(t, 0) + 1
    if s.get("tiers") != counts:
        probs.append(f"tiers {s.get('tiers')} != records {counts}")
    for k, t in (("n_candidates", "candidate"), ("n_interest", "interest"),
                 ("n_watch", "watch")):
        if int(s.get(k, -1)) != counts[t]:
            probs.append(f"{k}={s.get(k)} != {counts[t]}")
    f = s.get("funnel") or {}
    for k, t in (("stars_candidate", "candidate"), ("stars_interest", "interest"),
                 ("stars_watch", "watch")):
        if int(f.get(k, -1)) != counts[t]:
            probs.append(f"funnel.{k}={f.get(k)} != {counts[t]}")
    dem = demotions_by_stage(prov.get("vetoes") or {})
    if int(f.get("stars_demoted_by_vetstar", -1)) != len(dem["vetstar"]):
        probs.append("funnel.stars_demoted_by_vetstar disagrees with the records")
    if int(f.get("stars_demoted_by_lightcurve", -1)) != len(dem["lightcurve"]):
        probs.append("funnel.stars_demoted_by_lightcurve disagrees with the records")
    v = str(s.get("verdict") or "")
    for tok in re.findall(r"\b(REDETECT|VETSTAR)_DEMOTED_(\d+)\b", v):
        want = len(dem["lightcurve" if tok[0] == "REDETECT" else "vetstar"])
        if int(tok[1]) != want:
            probs.append(f"verdict token {tok[0]}_DEMOTED_{tok[1]} != records ({want})")
    for name in ("REDETECT", "VETSTAR"):
        if len(re.findall(rf"\b{name}_DEMOTED_\d+\b", v)) > 1:
            probs.append(f"verdict carries {name}_DEMOTED more than once")
    pend = "CLOCK_CANDIDATES_PENDING_VET" in v
    if pend != bool(counts["candidate"] or counts["interest"]):
        probs.append("verdict pending-state disagrees with the tiers")
    vs = s.get("vetstar") or {}
    reps = load_vetstar_reports(out)
    if vs and int(vs.get("n_vetted", len(reps))) != len(reps):
        probs.append(f"vetstar.n_vetted={vs.get('n_vetted')} != {len(reps)} files")
    return probs


__all__ = ["check_consistency", "load_vetstar_reports", "reconcile_all", "vetstar_filename"]
