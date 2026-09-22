#!/usr/bin/env python
"""Print METRONOME's single-star vet verbatim.

The point of the vet is that every catalogue's answer, and every light-curve
number behind the verdict, is legible in the run log without opening the JSON.
"NOT_LISTED" and "UNREACHED" are printed as they are stored, because they are
different claims: one is about the star, the other is about the archive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GAIA_FIELDS = (
    "source_id", "sep_arcsec_at_epoch", "sep_arcsec_gaia_epoch", "phot_g_mean_mag",
    "phot_bp_mean_mag", "phot_rp_mean_mag", "ruwe", "non_single_star",
    "astrometric_excess_noise", "astrometric_excess_noise_sig", "astrometric_gof_al",
    "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude", "duplicated_source",
    "phot_variable_flag", "radial_velocity", "radial_velocity_error", "rv_nb_transits",
    "rv_amplitude_robust", "rv_chisq_pvalue", "rv_renormalised_gof", "parallax",
    "parallax_error", "pmra", "pmdec", "teff_gspphot", "logg_gspphot", "distance_gspphot",
)

FOLD_FIELDS = ("period", "amplitude_ptp", "amplitude_sigma", "depth", "height",
               "extremum_is_dip", "frac_below_half_depth", "frac_above_half_height",
               "phase_of_min", "phase_of_max", "n_bins_filled", "n_points")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else Path("results/metronome/vetstar.json")
    if not path.exists():
        print(f"::warning::no {path} — the vet produced nothing")
        return 0
    d = json.loads(path.read_text())

    print("VERDICT:", d.get("verdict"))
    print("star", d.get("star_key"), "P", d.get("period"), "2P", d.get("period_double"))
    print("ra/dec", d.get("ra"), d.get("dec"), "from", d.get("position_source"))
    print("fetch", d.get("fetch_status"), d.get("fetch_route"),
          "lc", json.dumps(d.get("lightcurve", {}))[:400])
    print("n_flares_redetected", d.get("n_flares_redetected"),
          "masked_fraction", d.get("masked_fraction"),
          "n_catalogue_epochs", d.get("n_catalogue_epochs"),
          "from", d.get("catalogue_epochs_source"),
          json.dumps(d.get("catalogue_epochs_query") or {})[:400])

    print("\n--- what every catalogue says, verbatim ---")
    g = d.get("gaia") or {}
    print("gaia_source:", g.get("status"), "n_sources", g.get("n_sources"),
          "err", g.get("error"))
    for k in GAIA_FIELDS:
        print("   ", k, "=", (g.get("match") or {}).get(k))
    for n in (g.get("neighbours") or [])[:10]:
        sep = n.get("sep_arcsec_at_epoch")
        print("    neighbour", n.get("source_id"),
              "sep", None if sep is None else round(float(sep), 2),
              "G", n.get("phot_g_mean_mag"), "var", n.get("phot_variable_flag"))
    for t, r in (d.get("gaia_variability") or {}).items():
        print(f"{t}: {r.get('status')} rows={r.get('n_rows')}"
              + (f" err={r.get('error')}" if r.get("error") else ""))
        if r.get("row"):
            print("   ", json.dumps(r["row"])[:1200])
    for n, r in (d.get("vizier_cones") or {}).items():
        print(f"vizier {n} ({r.get('table')}): {r.get('status')} rows={r.get('n_rows')}"
              + (f" err={r.get('error')}" if r.get("error") else ""))
        for row in (r.get("rows") or [])[:3]:
            print("   ", json.dumps(row)[:800])
    for n, r in (d.get("vizier_by_id") or {}).items():
        print(f"vizier-by-id {n} ({r.get('seed')}): {r.get('status')} "
              f"rows={r.get('n_rows')} tables={r.get('tables_seen')}"
              + (f" err={r.get('error')}" if r.get("error") else ""))
        for row in (r.get("rows") or [])[:3]:
            print("   ", json.dumps(row)[:800])
    print("catalogued_binary_hits:", json.dumps(d.get("catalogued_binary_hits"))[:2000])

    print("\n--- the folded light curve ---")
    for k in ("fold_at_period", "fold_at_period_flares_masked", "fold_at_2p",
              "fold_at_2p_flares_masked"):
        f = d.get(k) or {}
        print(k, {kk: f.get(kk) for kk in FOLD_FIELDS})
    for k in ("fold_significance", "fold_significance_flares_masked",
              "fold_significance_2p", "harmonics_at_period", "harmonics_at_2p"):
        print(k, d.get(k))
    print("odd_even", d.get("odd_even"))
    print("odd_even_flares_masked", d.get("odd_even_flares_masked"))
    print("periodogram_detrended", d.get("periodogram_detrended"))
    print("periodogram_detrended_unmasked", d.get("periodogram_detrended_unmasked"))

    print("\n--- flare phase ---")
    for k in ("event_phase_rayleigh_at_p", "event_phase_rayleigh_at_2p",
              "catalogue_phase_rayleigh_at_p", "catalogue_phase_rayleigh_at_2p"):
        print(k, d.get(k))
    print("offset from photometric max", d.get("event_phase_offset_from_photometric_max"),
          "(binned)", d.get("event_phase_offset_from_photometric_max_binned"))

    print("\n--- neighbours catalogued AT the clock period ---")
    for h in (d.get("neighbour_period_matches") or []):
        print(f"  gaia {h.get('source_id')} at {h.get('sep_arcsec')}\" "
              f"G={h.get('phot_g_mean_mag')}")
        for m in h.get("matches") or []:
            print(f"     {m['source']}.{m['column']} = {m['period']} "
                  f"(x{m['harmonic']}) type={m['type']!r} name={m['name']!r}")

    print("\n--- the neighbours that could be the real source ---")
    for n in (d.get("neighbours") or []):
        print(f"  {n.get('source_id')} sep={n.get('sep_arcsec')} G={n.get('phot_g_mean_mag')} "
              f"flag={n.get('phot_variable_flag')} why={n.get('why')}")
        for t, r in (n.get("gaia_variability") or {}).items():
            print(f"     {t}: {r.get('status')} rows={r.get('n_rows')}")
            if r.get("row"):
                print("       ", json.dumps(r["row"])[:900])
        for cn, r in (n.get("vizier_cones") or {}).items():
            print(f"     vizier {cn}: {r.get('status')} rows={r.get('n_rows')}")
            for row in (r.get("rows") or [])[:2]:
                print("       ", json.dumps(row)[:700])

    print("\n--- per quarter ---")
    for r in (d.get("per_segment") or []):
        print("   ", r)
    print("amplitude ratio", d.get("per_segment_amplitude_ratio"))
    print("roll_season", json.dumps(d.get("roll_season"))[:1500])
    print("reconciliation:", json.dumps(d.get("reconciliation"))[:800])
    print("unreached:", d.get("unreached"))
    print("surviving:", d.get("surviving_explanations"))
    print("elapsed_s", d.get("elapsed_s"))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
