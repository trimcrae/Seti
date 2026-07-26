"""MIDDEN deep-dive: HD 217522 multi-instrument epoch audit.

The full-corpus MIDDEN run left exactly one surviving flag: HD 217522 (roAp),
single HARPS epoch, coherent Tc I triplet (z = 4.0/2.7/2.8), no control veto.
The discriminating physics: technetium at 4.2 Myr half-life is *static* on any
observational baseline, while roAp rare-earth blends modulate with pulsation
phase (5-15 min periods) and rotation.  This module therefore:

1. discovers EVERY high-resolution ESO Phase-3 spectrum of HD 217522 plus a
   panel of well-studied roAp comparison stars (any instrument covering the
   Tc I 4238/4262/4297 A triplet at R >= 30,000 — not just the HARPS/FEROS
   pull of the survey stage);
2. measures every epoch with the same NIST-gated line list and census
   machinery as the survey;
3. scores two things the survey could not:
     - epoch stability of the target's triplet depths (chi^2 against a
       constant; Tc must be constant, blends need not be), and
     - the target's standing against the roAp panel itself — the harshest
       available null model, because every panel star has the same
       rare-earth-forest confuser.

No detection is claimed from a flag that fails either test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .acquire import _retry, process_corpus, resolve_anchor
from .measure import Z_TC_EACH, Z_TC_QUAD, census_z

_ESO_TAP = "https://archive.eso.org/tap_obs"

TARGET = "HD 217522"

# roAp / cool-Ap comparison panel (name, fallback ra/dec deg — Sesame is
# authoritative on the runner; fallbacks are only a net for an outage — and
# approximate Teff in K for census windowing / the record).
PANEL = [
    ("HD 217522", 345.629, -44.846, 6750, 0),   # the target (priority 0)
    ("HD 128898", 220.627, -64.975, 7420, 1),   # alpha Cir
    ("HD 137949", 232.729, -17.441, 7550, 1),   # 33 Lib
    ("HD 24712", 58.817, -12.099, 7250, 1),     # DO Eri
    ("HD 101065", 174.404, -46.710, 6600, 1),   # Przybylski — blend extreme
    ("HD 201601", 317.585, 10.132, 7700, 1),    # gamma Equ
    ("HD 176232", 284.696, 13.907, 7550, 1),    # 10 Aql
    ("HD 83368", 144.208, -51.551, 7650, 1),    # HR 3831
    ("HD 60435", 111.230, -57.986, 8100, 1),
    ("HD 122970", 211.152, 5.626, 6930, 1),
    ("HD 19918", 46.999, -81.099, 7100, 1),
    ("HD 42659", 92.319, -16.198, 7900, 1),
]

# Full-triplet coverage with margin, in ObsCore metres.
_EM_MIN_MAX_M = 4.20e-7
_EM_MAX_MIN_M = 4.31e-7
_RES_MIN = 30000.0
_RES_WHITELIST = {"HARPS", "FEROS", "UVES", "ESPRESSO"}

_STAR_QUERY = """
SELECT dp_id, access_url, instrument_name, obs_collection,
       s_ra, s_dec, snr, t_min, em_min, em_max, em_res_power, target_name
FROM ivoa.ObsCore
WHERE dataproduct_type = 'spectrum'
  AND s_ra BETWEEN {ra_lo:.6f} AND {ra_hi:.6f}
  AND s_dec BETWEEN {dec_lo:.6f} AND {dec_hi:.6f}
  AND em_min < {em_lo:.4e} AND em_max > {em_hi:.4e}
"""


def star_box(ra: float, dec: float, radius_arcsec: float) -> dict:
    """RA/Dec box bounds around a position (RA widened by 1/cos dec)."""
    r_deg = radius_arcsec / 3600.0
    cosd = max(np.cos(np.radians(dec)), 1e-3)
    return {"ra_lo": ra - r_deg / cosd, "ra_hi": ra + r_deg / cosd,
            "dec_lo": dec - r_deg, "dec_hi": dec + r_deg,
            "em_lo": _EM_MIN_MAX_M, "em_hi": _EM_MAX_MIN_M}


def resolution_ok(df: pd.DataFrame) -> pd.Series:
    """R >= 30k, trusting the instrument whitelist when em_res_power is null."""
    res = pd.to_numeric(df.get("em_res_power"), errors="coerce")
    inst = df.get("instrument_name", pd.Series("", index=df.index))
    inst = inst.astype(str).str.upper().str.strip()
    whitelisted = inst.str.split().str[0].isin(_RES_WHITELIST)
    return (res >= _RES_MIN) | (res.isna() & whitelisted)


def discover_panel(out_dir: Path, radius_arcsec: float = 20.0) -> pd.DataFrame:
    """All triplet-covering high-resolution ESO spectra of the panel stars."""
    out_path = out_dir / "deepdive_obscore.parquet"
    if out_path.exists():
        print(f"[midden-deep] discovery checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo

    tap = pyvo.dal.TAPService(_ESO_TAP)
    chunk_dir = out_dir / "deepdive_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, fra, fdec, teff, priority in PANEL:
        part = chunk_dir / f"star_{name.replace(' ', '_')}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            continue
        pos = resolve_anchor(name, fra, fdec)
        q = _STAR_QUERY.format(**star_box(pos["ra"], pos["dec"],
                                          radius_arcsec))

        def _go(q=q):
            res = tap.run_async(q, maxrec=5000)
            df = res.to_table().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})

        df = _retry(_go, retries=3, label=f"deepdive obscore {name}")
        df["star"] = name
        df["teff"] = float(teff)
        df["priority"] = int(priority)
        df.to_parquet(part, index=False)
        print(f"[midden-deep] {name}: {len(df)} triplet-covering spectra "
              f"({pos.get('source', 'resolved')})")
        frames.append(df)
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    obs.to_parquet(out_path, index=False)
    return obs


def select_epochs(obs: pd.DataFrame, epochs_per_star: int = 30,
                  target_epochs: int = 80) -> pd.DataFrame:
    """One file per epoch, resolution-gated; the target keeps a deeper stack."""
    if not len(obs):
        return obs
    df = obs[resolution_ok(obs)].copy()
    df = df.sort_values("dp_id").drop_duplicates("dp_id")
    df["snr"] = pd.to_numeric(df.get("snr"), errors="coerce").fillna(0.0)
    df = df.sort_values(["star", "snr"], ascending=[True, False])
    df = df.drop_duplicates(["star", "t_min"])
    df["epoch_rank"] = df.groupby("star").cumcount()
    cap = np.where(df["star"] == TARGET, target_epochs, epochs_per_star)
    df = df[df["epoch_rank"] < cap].copy()
    df["tid"] = df.groupby("star", sort=True).ngroup()
    df["source"] = "deepdive"
    df = df.sort_values(["priority", "star", "epoch_rank"]).reset_index(drop=True)
    counts = df.groupby("star")["dp_id"].count().to_dict()
    print(f"[midden-deep] corpus: {len(df)} epochs across {len(counts)} stars: "
          f"{counts}")
    return df


def _census_by_instrument(meas: pd.DataFrame, min_group: int = 10) -> pd.DataFrame:
    """Census z within each instrument when populated, else panel-wide.

    Line depth at fixed abundance varies with spectrograph resolution, so an
    epoch is ranked against same-instrument peers whenever there are enough
    of them; the whole panel is one Teff class by construction, so the Teff
    window is effectively disabled.
    """
    out = census_z(meas, teff_window=1e9)          # pooled baseline
    inst = meas["instrument"].astype(str).str.upper()
    for _, g in meas.groupby(inst, sort=False):
        if g["dp_id"].nunique() >= min_group:
            zg = census_z(g, teff_window=1e9)
            out.loc[zg.index, "z"] = zg["z"]
    return out


def _stability(target_meas: pd.DataFrame) -> dict:
    """Chi^2-against-constant for each Tc line's depth across epochs."""
    from scipy.stats import chi2 as chi2_dist

    out = {}
    tc = target_meas[target_meas["species"] == "Tc I"]
    for lam, g in tc.groupby("wavelength"):
        d = g["depth"].to_numpy(float)
        e = np.clip(g["depth_err"].to_numpy(float), 1e-4, None)
        ok = np.isfinite(d) & np.isfinite(e)
        d, e = d[ok], e[ok]
        rec = {"n_epochs": int(len(d))}
        if len(d) >= 2:
            w = 1.0 / e ** 2
            mean = float(np.sum(w * d) / np.sum(w))
            chi2 = float(np.sum(((d - mean) / e) ** 2))
            dof = len(d) - 1
            rec.update({"weighted_mean_depth": mean, "chi2": chi2, "dof": dof,
                        "chi2_dof": chi2 / dof,
                        "p_constant": float(chi2_dist.sf(chi2, dof)),
                        "depth_span": float(np.ptp(d))})
        out[f"{lam:.2f}"] = rec
    return out


def score_deepdive(cfg: Config, meas: pd.DataFrame) -> dict:
    out_dir = cfg.root / "results" / "midden_deepdive"
    out_dir.mkdir(parents=True, exist_ok=True)

    meas = meas[meas["role"].isin(["radionuclide", "rv_ref", "control"])].copy()
    if meas.empty:
        raise RuntimeError("midden-deep: zero usable measurements")
    meas_z2 = _census_by_instrument(meas)
    meas_z2.to_parquet(out_dir / "deepdive_measurements.parquet", index=False)

    # Per-star median z per Tc line (panel standing).
    tc = meas_z2[meas_z2["species"] == "Tc I"]
    per_star = tc.pivot_table(index="star", columns="wavelength", values="z",
                              aggfunc="median")
    per_star["tc_quad_med"] = np.sqrt((per_star.clip(lower=0) ** 2).sum(axis=1))
    panel_rank = per_star["tc_quad_med"].rank(pct=True)

    target_meas = meas_z2[meas_z2["star"] == TARGET]
    n_target_epochs = int(target_meas["dp_id"].nunique())
    stability = _stability(target_meas)

    # Per-epoch coherent-triplet verdicts for the target.
    epoch_rows = []
    for dp, g in target_meas.groupby("dp_id"):
        tcg = g[g["species"] == "Tc I"].sort_values("wavelength")
        z = tcg["z"].to_numpy(float)
        ctrl = g[g["role"] == "control"]["z"].to_numpy(float)
        epoch_rows.append({
            "dp_id": dp, "instrument": str(g["instrument"].iloc[0]),
            "t_min": float(pd.to_numeric(g["t_min"].iloc[0], errors="coerce")),
            "z_4238": float(z[0]) if len(z) > 0 else np.nan,
            "z_4262": float(z[1]) if len(z) > 1 else np.nan,
            "z_4297": float(z[2]) if len(z) > 2 else np.nan,
            "tc_coherent": bool(len(z) == 3 and np.all(np.isfinite(z))
                                and np.all(z >= Z_TC_EACH)
                                and np.sqrt(np.sum(z ** 2)) >= Z_TC_QUAD),
            "control_veto": bool(np.any(np.isfinite(ctrl) & (ctrl >= 4.0))),
        })
    cols = ["dp_id", "instrument", "t_min", "z_4238", "z_4262", "z_4297",
            "tc_coherent", "control_veto"]
    epochs = (pd.DataFrame(epoch_rows).sort_values("t_min")
              if epoch_rows else pd.DataFrame(columns=cols))
    epochs.to_csv(out_dir / "target_epochs.csv", index=False)

    n_coherent = int(epochs["tc_coherent"].sum()) if len(epochs) else 0
    summary = {
        "target": TARGET,
        "n_target_epochs": n_target_epochs,
        "n_panel_epochs": int(meas_z2["dp_id"].nunique()),
        "n_panel_stars": int(meas_z2["star"].nunique()),
        "target_epochs_coherent": n_coherent,
        "target_epochs_vetoed": int(epochs["control_veto"].sum()) if len(epochs) else 0,
        "target_panel_percentile_tc_quad": float(panel_rank.get(TARGET, np.nan)),
        "per_star_tc_quad_median": {s: float(v) for s, v in
                                    per_star["tc_quad_med"].items()},
        "stability": stability,
        "verdict_rules": {
            "sustained": "coherent in >=2 epochs AND every-line p_constant > 0.01 "
                         "AND panel percentile > 0.9",
            "note": "Tc (4.2 Myr half-life) must be epoch-static; roAp blends "
                    "modulate. A flag failing stability or panel standing is "
                    "attributed to line-forest blending.",
        },
    }
    (out_dir / "deepdive_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    lines = ["# MIDDEN deep-dive: HD 217522", "",
             f"Panel: {summary['n_panel_stars']} roAp/cool-Ap stars, "
             f"{summary['n_panel_epochs']} epochs; target epochs: "
             f"{n_target_epochs}.", "",
             f"Coherent-triplet epochs: {n_coherent}/{n_target_epochs}; "
             f"panel percentile (median Tc quad z): "
             f"{summary['target_panel_percentile_tc_quad']:.2f}", "",
             "## Stability (chi^2 against constant depth)", ""]
    for lam, rec in stability.items():
        if "chi2_dof" in rec:
            lines.append(f"- Tc I {lam}: {rec['n_epochs']} epochs, "
                         f"chi2/dof {rec['chi2_dof']:.2f}, "
                         f"p(constant) {rec['p_constant']:.3g}, "
                         f"depth span {rec['depth_span']:.4f}")
        else:
            lines.append(f"- Tc I {lam}: {rec['n_epochs']} epoch(s) — "
                         "stability not testable")
    lines += ["", "## Panel standing (median Tc quadrature z per star)", ""]
    for s, v in sorted(summary["per_star_tc_quad_median"].items(),
                       key=lambda kv: -kv[1]):
        mark = " <-- target" if s == TARGET else ""
        lines.append(f"- {s}: {v:.2f}{mark}")
    (out_dir / "DEEPDIVE.md").write_text("\n".join(lines) + "\n")
    print(f"[midden-deep] scored: {json.dumps(summary)[:400]}")
    return summary


def deepdive_run(cfg: Config | None = None, stage: str = "all",
                 epochs_per_star: int = 30, target_epochs: int = 80,
                 radius_arcsec: float = 20.0, batch_size: int = 25,
                 scratch_dir: str | Path | None = None) -> dict:
    import tempfile

    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "midden_deepdive"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = meas = None
    if stage in ("discover", "all"):
        obs = discover_panel(out_dir, radius_arcsec=radius_arcsec)
        corpus_path = out_dir / "deepdive_corpus.parquet"
        if corpus_path.exists():
            corpus = pd.read_parquet(corpus_path)
            print(f"[midden-deep] corpus checkpoint exists: {corpus_path}")
        else:
            corpus = select_epochs(obs, epochs_per_star=epochs_per_star,
                                   target_epochs=target_epochs)
            corpus.to_parquet(corpus_path, index=False)
        if stage == "discover":
            return {"stage": stage, "n_epochs": int(len(corpus))}
    if stage in ("measure", "all"):
        corpus = corpus if corpus is not None else \
            pd.read_parquet(out_dir / "deepdive_corpus.parquet")
        scratch = Path(scratch_dir) if scratch_dir else \
            Path(tempfile.gettempdir()) / "midden_deep_scratch"
        import hashlib
        key = hashlib.sha1(",".join(corpus["dp_id"].astype(str))
                           .encode()).hexdigest()[:10]
        meas = process_corpus(corpus, out_dir / f"meas_{key}", scratch,
                              batch_size=batch_size)
        meas.to_parquet(out_dir / "deepdive_raw_measurements.parquet",
                        index=False)
    if stage in ("score", "all"):
        meas = meas if meas is not None else \
            pd.read_parquet(out_dir / "deepdive_raw_measurements.parquet")
        meas = meas.rename(columns={"instrument_name": "instrument"}) \
            if "instrument_name" in meas.columns else meas
        if "instrument" not in meas.columns:
            meas["instrument"] = ""
        return score_deepdive(cfg, meas)
    return {"stage": stage, "status": "done"}
