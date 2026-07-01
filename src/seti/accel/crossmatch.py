"""Cross-match the astrometric dark-companion shortlist against the *published*
Gaia compact-companion catalogues --- the decisive novelty test.

The AMRF triage that flags our class-3 systems is exactly the method Shahaf et
al. (2023, MNRAS 518, 2991) and its DR3 sequels applied to the same Gaia
``nss_two_body_orbit`` table.  Recovering their catalogue validates the
pipeline; the only *remarkable* outcome is a class-3, single-lined, nearby
system that is **absent** from every published list.  This module runs that
comparison: it pulls the published catalogues from VizieR, matches our
candidates to them by Gaia source_id (with a positional fallback), and labels
each candidate ``in_literature`` / ``ABSENT``.

Blocked in the sandbox (VizieR egress); runs on the GitHub runner.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# Published Gaia astrometric compact-/dark-companion catalogues on VizieR.  Each
# entry: VizieR catalogue id -> the column(s) that carry the Gaia DR3 source_id.
# We read *every* table in the catalogue and scan all of these column names, so a
# rename between tables does not silently drop a match.
_LIT_CATALOGS = {
    # Shahaf, Bashi, Mazeh et al. 2023 -- "Triage of Gaia astrometric binaries:
    # the search for compact companions".  The direct methodological twin.
    "J/MNRAS/518/2991": {
        "label": "Shahaf+2023",
        "id_cols": ["Source", "GaiaDR3", "DR3", "SourceID", "Gaia"],
    },
    # Shahaf, El-Badry, Mazeh et al. 2024 -- extended DR3 compact-companion set.
    "J/MNRAS/529/3466": {
        "label": "Shahaf+2024",
        "id_cols": ["Source", "GaiaDR3", "DR3", "SourceID", "Gaia"],
    },
    # Gaia Collaboration / Halbwachs+2023 astrometric-binary orbit validation and
    # El-Badry+2023 dormant-BH catalogue land here if resolvable; kept generic.
    "J/A+A/674/A9": {
        "label": "GaiaDR3-NSS-validated",
        "id_cols": ["Source", "GaiaDR3", "DR3", "SourceID"],
    },
}


def _fetch_literature_ids(verbose: bool = True) -> dict[str, set]:
    """Return {catalogue_label: set(source_id)} for every published catalogue we
    can reach.  Failures are logged and skipped, never fatal."""
    from astroquery.vizier import Vizier

    v = Vizier(columns=["**"], row_limit=-1)
    out: dict[str, set] = {}
    for cat, spec in _LIT_CATALOGS.items():
        label = spec["label"]
        try:
            tables = v.get_catalogs(cat)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[xmatch] {label} ({cat}) fetch failed: {exc!r}")
            continue
        ids: set[int] = set()
        for tbl in tables:
            df = tbl.to_pandas()
            for col in spec["id_cols"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    ids.update(int(x) for x in vals if x > 0)
        out[label] = ids
        if verbose:
            print(f"[xmatch] {label}: {len(ids)} Gaia source_ids")
    return out


def _fetch_literature_positions(verbose: bool = True) -> pd.DataFrame:
    """Positional fallback: RA/Dec of every published entry, so a candidate whose
    source_id column is absent/renamed can still be matched on the sky."""
    from astroquery.vizier import Vizier

    v = Vizier(columns=["**"], row_limit=-1)
    rows = []
    for cat, spec in _LIT_CATALOGS.items():
        label = spec["label"]
        try:
            tables = v.get_catalogs(cat)
        except Exception:  # noqa: BLE001
            continue
        for tbl in tables:
            df = tbl.to_pandas()
            racol = next((c for c in ("RA_ICRS", "RAJ2000", "_RAJ2000", "RA", "ra")
                          if c in df.columns), None)
            decol = next((c for c in ("DE_ICRS", "DEJ2000", "_DEJ2000", "DE", "dec")
                          if c in df.columns), None)
            if racol and decol:
                sub = pd.DataFrame({
                    "lit_label": label,
                    "ra": pd.to_numeric(df[racol], errors="coerce"),
                    "dec": pd.to_numeric(df[decol], errors="coerce"),
                })
                rows.append(sub.dropna())
    if not rows:
        return pd.DataFrame(columns=["lit_label", "ra", "dec"])
    return pd.concat(rows, ignore_index=True)


def _positional_match(ra: float, dec: float, lit: pd.DataFrame,
                      radius_arcsec: float = 2.0) -> str | None:
    """Nearest published entry within ``radius_arcsec`` (great-circle), or None."""
    if not len(lit):
        return None
    dra = (lit["ra"].to_numpy() - ra) * np.cos(np.radians(dec))
    dde = lit["dec"].to_numpy() - dec
    sep_arcsec = np.hypot(dra, dde) * 3600.0
    i = int(np.argmin(sep_arcsec))
    if sep_arcsec[i] <= radius_arcsec:
        return str(lit["lit_label"].iloc[i])
    return None


def crossmatch_candidates(candidates: pd.DataFrame,
                          lit_ids: dict[str, set] | None = None,
                          lit_pos: pd.DataFrame | None = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Label each candidate with which published catalogues contain it.

    ``candidates`` needs ``source_id`` (and ``ra``/``dec`` for the positional
    fallback).  Returns a copy with ``lit_matches`` (comma list), ``in_literature``
    (bool) and ``match_kind`` ('source_id' / 'position' / '').  Pass ``lit_ids``/
    ``lit_pos`` to run fully offline (tests); otherwise they are fetched.
    """
    if lit_ids is None:
        lit_ids = _fetch_literature_ids(verbose=verbose)
    if lit_pos is None:
        lit_pos = _fetch_literature_positions(verbose=verbose)

    out = candidates.copy()
    matches, kinds = [], []
    for _, row in out.iterrows():
        sid = int(row["source_id"])
        hit = [label for label, ids in lit_ids.items() if sid in ids]
        kind = "source_id" if hit else ""
        if not hit and {"ra", "dec"} <= set(out.columns):
            pm = _positional_match(float(row["ra"]), float(row["dec"]), lit_pos)
            if pm:
                hit = [pm]
                kind = "position"
        matches.append(",".join(hit))
        kinds.append(kind)
    out["lit_matches"] = matches
    out["match_kind"] = kinds
    out["in_literature"] = [bool(m) for m in matches]
    # Coverage guard: if NO published catalogue returned any ids (VizieR egress
    # failed), an "absent" label is meaningless -- record it so a null fetch is
    # never mistaken for a genuine novelty detection.
    total_lit = sum(len(ids) for ids in lit_ids.values())
    out.attrs["lit_ids_total"] = int(total_lit)
    out.attrs["lit_catalogs_loaded"] = int(sum(1 for v in lit_ids.values() if v))
    return out


def run_crossmatch(candidates_csv, out_dir, known_bh: set | None = None,
                   verbose: bool = True) -> dict:
    """Load the class-3 shortlist, cross-match against the literature, and write
    ``literature_crossmatch.csv`` + a summary highlighting any ABSENT candidate.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(candidates_csv)
    xm = crossmatch_candidates(cand, verbose=verbose)

    known_bh = known_bh or set()
    xm["known_gaia_bh"] = xm["source_id"].isin(known_bh)
    # A genuinely new object: not a validation BH, and absent from every catalogue.
    xm["novel_candidate"] = (~xm["in_literature"]) & (~xm["known_gaia_bh"])

    xm.to_csv(out_dir / "literature_crossmatch.csv", index=False)
    novel = xm[xm["novel_candidate"]]
    summary = {
        "n_candidates": int(len(xm)),
        "n_in_literature": int(xm["in_literature"].sum()),
        "n_known_bh": int(xm["known_gaia_bh"].sum()),
        "n_novel_absent": int(len(novel)),
        "novel_source_ids": [int(s) for s in novel["source_id"]],
        "novel": novel.to_dict("records"),
    }
    (out_dir / "literature_crossmatch_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"[xmatch] {len(xm)} candidates: {summary['n_in_literature']} in "
              f"literature, {summary['n_known_bh']} known BH, "
              f"{summary['n_novel_absent']} ABSENT (novel).")
        if len(novel):
            print(novel.to_string(index=False))
    return summary


__all__ = ["crossmatch_candidates", "run_crossmatch", "_fetch_literature_ids",
           "_fetch_literature_positions"]
