"""End-to-end Gaia XP anomaly search over a sky chunk.

Pulls XP spectra for a cone, fits the self-calibrating stellar-shape locus, scores
every source for a global or localised spectral anomaly, runs the survivors
through a contamination funnel (DSC quasar/galaxy, white dwarf, known variable),
and writes the ranked clean anomalies + an occurrence-style summary.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..stats.upper_limit import occurrence_upper_limit
from .anomaly import anomaly_score, fit_locus, normalize_spectrum


def classify_xp_anomaly(row: dict) -> str:
    """Why an anomalous XP spectrum might be mundane, or 'clean' if none apply."""
    if float(row.get("classprob_dsc_combmod_quasar", 0) or 0) > 0.5:
        return "quasar"
    if float(row.get("classprob_dsc_combmod_galaxy", 0) or 0) > 0.5:
        return "galaxy"
    # White dwarf: very sub-luminous + blue (mundane peculiar SED).
    g = row.get("phot_g_mean_mag")
    plx = row.get("parallax")
    bp_rp = row.get("bp_rp")
    if (g is not None and plx is not None and bp_rp is not None
            and np.isfinite(g) and np.isfinite(plx) and plx > 0):
        m_g = float(g) + 5.0 * np.log10(float(plx) / 100.0)
        if m_g > 10.0 and float(bp_rp) < 1.5:
            return "white_dwarf"
    if str(row.get("phot_variable_flag", "")).upper().startswith("VARIABLE"):
        return "known_variable"
    if int(row.get("non_single_star", 0) or 0) > 0:
        return "non_single_star"
    return "clean"


def xp_run(cfg: Config | None = None, ra: float = 180.0, dec: float = 30.0,
           radius_deg: float = 1.0, g_max: float = 17.5, limit: int = 20000,
           global_sigma_min: float = 8.0, feature_resid_min: float = 6.0,
           chunk: dict | None = None) -> dict:
    """Search one XP chunk for spectral-shape anomalies.  ``chunk`` may be passed
    directly (offline tests) as ``{'wave','flux','meta'}``; otherwise it is fetched
    from the Gaia archive.  Returns the summary."""
    cfg = cfg or load_config()

    if chunk is None:
        from .acquire import assemble_chunk, fetch_xp_metadata, fetch_xp_spectra
        meta = fetch_xp_metadata(ra, dec, radius_deg=radius_deg, g_max=g_max,
                                 limit=limit)
        spec = fetch_xp_spectra(meta["source_id"].tolist())
        chunk = assemble_chunk(meta, spec)

    flux = np.asarray(chunk["flux"], dtype=float)
    meta = chunk["meta"].reset_index(drop=True)
    n_searched = int(flux.shape[0])
    if n_searched < 50:
        print(f"[xp] too few spectra ({n_searched}) to model a locus")
        return {"n_searched": n_searched, "n_anomalies": 0}

    norm = np.vstack([normalize_spectrum(f) for f in flux])
    colors = pd.to_numeric(meta.get("bp_rp"), errors="coerce").to_numpy()
    locus = fit_locus(norm, colors)
    scores = [anomaly_score(norm[i], colors[i], locus) for i in range(n_searched)]

    rows = []
    for i, sc in enumerate(scores):
        gs = sc["global_sigma"]
        fr = sc["feature_resid"]
        is_anom = (np.isfinite(gs) and gs >= global_sigma_min) or \
                  (np.isfinite(fr) and fr >= feature_resid_min)
        if not is_anom:
            continue
        m = meta.iloc[i].to_dict()
        reason = classify_xp_anomaly(m)
        rows.append({**{k: m.get(k) for k in (
            "source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax",
            "teff_gspphot", "classprob_dsc_combmod_quasar",
            "classprob_dsc_combmod_galaxy", "phot_variable_flag")},
            "global_sigma": float(gs), "feature_resid": float(fr),
            "feature_index": int(sc["feature_index"]),
            "class": reason, "_spec_index": i})

    clean = [r for r in rows if r["class"] == "clean"]
    clean.sort(key=lambda r: max(r["global_sigma"], r["feature_resid"]),
               reverse=True)

    out_dir = cfg.root / "results" / "xp"
    tag = f"f{ra:+06.1f}{dec:+05.1f}".replace(".", "p").replace("+", "p").replace("-", "m")
    field_dir = out_dir / tag
    field_dir.mkdir(parents=True, exist_ok=True)

    # Save the top clean anomalies' spectra so the actual shapes can be examined.
    from .anomaly import _bin_of
    windows = []
    for r in clean[:30]:
        idx = r["_spec_index"]
        b = _bin_of(float(colors[idx]) if np.isfinite(colors[idx]) else 0.0,
                    locus.bin_edges)
        windows.append({k: r[k] for k in r if not k.startswith("_")}
                       | {"wave": list(map(float, chunk["wave"])),
                          "flux_norm": list(map(float, norm[idx])),
                          "model": list(map(float, locus.medians[b]))})
    (field_dir / "top_anomalies.json").write_text(json.dumps(windows))
    if rows:
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows]).to_csv(field_dir / "xp_anomalies.csv",
                                             index=False)

    from collections import Counter
    counts = dict(Counter(r["class"] for r in rows))
    lim = occurrence_upper_limit(k=len(clean), n_eff=max(n_searched, 1),
                                 confidence=cfg.thresholds["stats"]["upper_limit_confidence"])
    # Reliability guard: the colour-conditional locus needs a well-populated
    # sample.  Too few sources, or an implausibly high anomaly fraction, means the
    # per-bin scatter is underestimated and the "anomalies" are locus noise, not
    # real outliers -- do not treat such a run as a candidate list.
    anomaly_fraction = len(rows) / max(n_searched, 1)
    reliable = (n_searched >= 1500) and (anomaly_fraction <= 0.15)
    summary = {
        "field": {"ra": ra, "dec": dec, "radius_deg": radius_deg},
        "n_searched": n_searched, "n_anomalies_raw": len(rows),
        "n_clean_anomalies": len(clean),
        "anomaly_fraction": round(anomaly_fraction, 4),
        "reliable": bool(reliable),
        "rejection_counts": counts,
        "thresholds": {"global_sigma_min": global_sigma_min,
                       "feature_resid_min": feature_resid_min},
        "top_clean": [{k: r[k] for k in ("source_id", "ra", "dec", "global_sigma",
                                         "feature_resid", "phot_g_mean_mag",
                                         "bp_rp", "teff_gspphot")}
                      for r in clean[:20]],
        "occurrence_limit": {"k": lim.k, "n_eff": lim.n_eff,
                             "f_upper": lim.f_upper, "f_point": lim.f_point},
    }
    (field_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[xp] summary:", json.dumps({k: summary[k] for k in
          ("n_searched", "n_anomalies_raw", "n_clean_anomalies", "anomaly_fraction",
           "reliable", "rejection_counts")}))
    if not reliable:
        print(f"[xp] WARNING: run UNRELIABLE (n={n_searched}, "
              f"anomaly_fraction={anomaly_fraction:.2f}) -- locus undersampled; "
              f"these are not candidates. Use a denser/larger field.")
    return summary


__all__ = ["xp_run", "classify_xp_anomaly"]
