"""CENOTAPH stage orchestration.

The funnel is an energy-conservation argument executed as a sequence of vetoes.
Each stage writes its own checkpoint immediately, so a killed runner loses
minutes rather than the whole pull.

    sample ─▶ twins ─▶ grey ─▶ midir ─▶ farir ─▶ reduce

``sample``  Gaia DR3 GSP-Spec dwarfs + 2MASS + AllWISE, parallax zero-point.
``twins``   ΔM_Ks against ≥50 parameter twins; twin-median intrinsic colours.
``grey``    joint GLS for (grey ``g``, ``A_V``) on the per-band residuals, with
            the distance modulus entering as a fully correlated component
            because it is *exactly* degenerate with grey.  **Leg 1.**
``midir``   the residual in W3/W4 *after* removing the fitted grey and
            reddening; a significantly negative residual is a mid-IR excess and
            disqualifies the star.  Measured here from the photometry rather
            than inherited from a published excess catalogue — those are ~92%
            false positive (Silverberg et al. 2018), so inheriting one would
            import someone else's error rate.  **Leg 2.**
``farir``   AKARI/FIS + IRAS association and the closure ratio
            ``ρ = f_IR/f_dim``.  **Leg 3.**
``reduce``  vetting, funnel counts, summary.json.

Every stage runs offline against a synthetic table when ``--synthetic`` is
given, which is how the whole funnel is exercised in CI with no network.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import (
    FAR_IR_BANDS,
    close_budget,
    coverage_table,
    radius_for_temperature,
    wise_temperature_ceilings,
)
from .extinction import EXCESS_BANDS, FIT_BANDS, a_over_av
from .greyfit import fit_grey_reddening, minimum_detectable_f
from .twins import TwinConfig, twin_colour_medians, twin_statistics
from .vet import (
    FAR_IR_SOURCE_DENSITY_PER_SQDEG,
    VetThresholds,
    background_galaxy_probability,
    expected_false_matches,
    vet_coverage,
    vet_table,
)

STAGES = ("sample", "twins", "grey", "midir", "farir", "reduce")

# Solar-luminosity proxy used when no bolometric luminosity is available. The
# closure test needs L only through f·L/4πd², so a 20% error in L moves the
# closure ratio by 20% — well inside the factor-3 tolerance.
_SUN_MKS = 3.28


def _out_dir(cfg=None, out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        p = Path(out_dir)
    elif cfg is not None:
        p = Path(cfg.path("results_dir")) / "cenotaph"
    else:
        p = Path("results/cenotaph")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(x):
    if is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2))


def luminosity_lsun(m_ks: np.ndarray) -> np.ndarray:
    """Crude L/L_⊙ from M_Ks. Only ever used inside the closure ratio.

    ``M_Ks`` is nearly a bolometric proxy for FGK dwarfs (BC_Ks varies by
    ~0.1 mag over 4000–7000 K), so this is good to ~15% — far better than the
    factor-3 tolerance the closure test applies.
    """
    return 10.0 ** (-0.4 * (np.asarray(m_ks, float) - _SUN_MKS))


# --------------------------------------------------------------------------
# Stage 0: probe — one minimal query, live, before spending a 5-hour pull
# --------------------------------------------------------------------------
def stage_probe(out: Path, **kw) -> dict:
    """Run ONE minimal archive query and print what came back.

    Cheap enough to iterate on (seconds, not the 110 minutes the full sample
    stage burned before failing), and it answers the only questions that
    matter after an acquisition failure: which transport works, what
    ``COUNT(*)`` says, and whether a plain ``SELECT`` returns that many rows.
    """
    from .acquire import probe_gaia

    res = probe_gaia(poe_min=kw.get("poe_min", 20.0),
                     ruwe_max=kw.get("ruwe_max", 1.4),
                     logg_min=kw.get("logg_min", 3.8),
                     teff_lo=kw.get("teff_lo", 4000.0),
                     teff_hi=kw.get("teff_hi", 7000.0),
                     plx_lo=kw.get("probe_plx_lo", 2.0),
                     plx_hi=kw.get("probe_plx_hi", 2.5))
    _write_json(out / "probe.json", res)
    print(json.dumps(_jsonable({k: v for k, v in res.items()
                                if k not in ("first_rows", "columns")}), indent=2))
    return res


def _empty_sample_summary(out: Path) -> dict:
    """The verdict for an empty parent sample — read off the ledger, not guessed.

    ``NO_DATA_REACHED`` is now reserved for its literal meaning: no query
    executed anywhere. A query that ran and matched nothing is
    ``QUERY_RETURNED_ZERO_ROWS``, which is a statement about the cuts and is
    actionable in a completely different way.
    """
    meta_path = out / "sample_meta.json"
    acq = {}
    verdict = "NO_DATA_REACHED"
    note = ("no sample.parquet and no acquisition ledger: the sample stage did "
            "not run, or its output never reached this job")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            acq = meta.get("acquisition", {}) or {}
            verdict = meta.get("verdict") or acq.get("acquisition_verdict") \
                or "NO_DATA_REACHED"
            note = acq.get("note", note)
        except Exception as exc:  # noqa: BLE001
            note = f"sample_meta.json unreadable: {exc!r}"
    return {"verdict": verdict, "funnel": {"n_sample": 0},
            "acquisition": acq, "note": note}


# --------------------------------------------------------------------------
# Stage 1: sample
# --------------------------------------------------------------------------
def stage_sample(out: Path, synthetic: bool = False, n_synth: int = 4000,
                 seed: int = 7, **kw) -> pd.DataFrame:
    if synthetic:
        from .synth import add_subdwarfs, inject_grey, inject_reddening, make_population

        df = make_population(n=n_synth, seed=seed)
        rng = np.random.default_rng(seed + 1)
        idx_grey = rng.choice(df.index, size=max(4, n_synth // 200), replace=False)
        df = inject_grey(df, idx_grey, grey_mag=0.45)
        rest = df.index.difference(idx_grey)
        idx_red = rng.choice(rest, size=max(4, n_synth // 100), replace=False)
        df = inject_reddening(df, idx_red, av=0.55)
        df = add_subdwarfs(df, n=max(50, n_synth // 20), seed=seed + 2)
        meta = {"mode": "synthetic", "n": int(len(df)), "seed": seed,
                "n_grey_injected": int(len(idx_grey)),
                "n_reddened_injected": int(len(idx_red))}
    else:
        from .acquire import (
            apply_parallax_zero_point,
            fetch_external_photometry,
            fetch_gspspec_sample,
            filter_gspspec_flags,
        )

        where = {"poe_min": kw.get("poe_min", 20.0),
                 "ruwe_max": kw.get("ruwe_max", 1.4),
                 "logg_min": kw.get("logg_min", 3.8),
                 "teff_lo": kw.get("teff_lo", 4000.0),
                 "teff_hi": kw.get("teff_hi", 7000.0)}
        from .acquire import summarise_acquisition

        ck = out / "checkpoints"
        ledger: list = []
        df = fetch_gspspec_sample(
            poe_min=where["poe_min"], ruwe_max=where["ruwe_max"],
            logg_min=where["logg_min"], teff_lo=where["teff_lo"],
            teff_hi=where["teff_hi"], plx_min_mas=kw.get("plx_min_mas", 1.0),
            checkpoint_dir=ck, ledger_out=ledger)
        acq = summarise_acquisition(ledger)
        if df.empty:
            # The three ways to have no stars are three different facts, and
            # the archive tells us which. Reporting "NO_DATA_REACHED" for a
            # query that ran fine and matched nothing — or for one that
            # returned 703k rows and then hit a dead shell — is the bug this
            # guard exists to prevent.
            _write_json(out / "sample_meta.json",
                        {"mode": "archive", "n": 0,
                         "verdict": acq["acquisition_verdict"],
                         "cuts": where, "acquisition": acq})
            print(f"[cenotaph] stage sample: 0 stars, "
                  f"verdict {acq['acquisition_verdict']}")
            return df
        df = filter_gspspec_flags(df)
        df = apply_parallax_zero_point(df)
        phot_acq = {}
        for kind in ("twomass", "allwise"):
            pled: list = []
            try:
                ph = fetch_external_photometry(
                    kind, where, plx_min_mas=kw.get("plx_min_mas", 1.0),
                    checkpoint_dir=ck, ledger_out=pled)
                df = df.merge(ph, on="source_id", how="left")
            except Exception as exc:  # noqa: BLE001
                pled.append({"chunk": kind, "status": "QUERY_FAILED",
                             "n_rows": 0, "error": repr(exc)})
                print(f"[cenotaph] {kind} photometry unavailable: {exc!r}")
            phot_acq[kind] = summarise_acquisition(pled)
        df = df.rename(columns={"phot_g_mean_mag": "g_mag",
                                "phot_bp_mean_mag": "bp_mag",
                                "phot_rp_mean_mag": "rp_mag",
                                "teff_gspspec": "teff", "logg_gspspec": "logg",
                                "mh_gspspec": "mh", "alphafe_gspspec": "alphafe",
                                "b": "b_gal"})
        for b, foe in (("g", "phot_g_mean_flux_over_error"),
                       ("bp", "phot_bp_mean_flux_over_error"),
                       ("rp", "phot_rp_mean_flux_over_error")):
            if foe in df.columns:
                df[f"{b}_mag_error"] = (2.5 / np.log(10.0)) / pd.to_numeric(
                    df[foe], errors="coerce")
        meta = {"mode": "archive", "n": int(len(df)),
                "verdict": acq["acquisition_verdict"],
                "parallax_zp_method": str(df.get("parallax_zp_method",
                                                 pd.Series(["unknown"])).iloc[0]),
                "cuts": where,
                "acquisition": acq,
                "photometry_acquisition": phot_acq}

    df.to_parquet(out / "sample.parquet", index=False)
    _write_json(out / "sample_meta.json", meta)
    print(f"[cenotaph] stage sample: {len(df)} stars -> {out / 'sample.parquet'}")
    return df


# --------------------------------------------------------------------------
# Stage 2: twins
# --------------------------------------------------------------------------
def stage_twins(df: pd.DataFrame, out: Path, cfg_twin: TwinConfig | None = None,
                bands: list[str] | None = None) -> pd.DataFrame:
    cfg_twin = cfg_twin or TwinConfig()
    bands = bands or [b for b in list(FIT_BANDS) + [e.name for e in EXCESS_BANDS]
                      if f"{b}_mag" in df.columns]
    tw = twin_statistics(df, cfg_twin)
    cm = twin_colour_medians(df, cfg_twin, bands=bands)
    res = pd.concat([df.reset_index(drop=True), tw.reset_index(drop=True),
                     cm.reset_index(drop=True)], axis=1)
    res.to_parquet(out / "twins.parquet", index=False)
    ok = int((res["twin_verdict"] == "ok").sum())
    print(f"[cenotaph] stage twins: {ok}/{len(res)} stars have a usable twin set; "
          f"median twin scatter {np.nanmedian(res['twin_scatter']):.3f} mag")
    return res


# --------------------------------------------------------------------------
# Stage 3+4: grey fit (leg 1) and the mid-IR excess veto (leg 2)
# --------------------------------------------------------------------------
def _per_band_residuals(row, bands: list[str], dm_ref: float, ref: str):
    dm, sig, names = [], [], []
    for b in bands:
        if b == ref:
            dm.append(dm_ref)
            sig.append(float(row.get(f"{ref}_mag_error", 0.02) or 0.02))
            names.append(b)
            continue
        mb = row.get(f"{b}_mag")
        mr = row.get(f"{ref}_mag")
        med = row.get(f"{b}_col_med")
        if mb is None or mr is None or med is None:
            continue
        if not (np.isfinite(mb) and np.isfinite(mr) and np.isfinite(med)):
            continue
        dm.append(dm_ref + (float(mb) - float(mr) - float(med)))
        eb = float(row.get(f"{b}_mag_error", 0.02) or 0.02)
        sc = row.get(f"{b}_col_scatter")
        sc = float(sc) if sc is not None and np.isfinite(sc) else 0.03
        sig.append(float(np.hypot(eb, sc)))
        names.append(b)
    return names, np.array(dm), np.array(sig)


def stage_grey(df: pd.DataFrame, out: Path, cfg_twin: TwinConfig | None = None,
               z_min: float = 3.0, max_fit: int | None = None,
               excess_nsigma: float = 3.0) -> pd.DataFrame:
    """Fit (g, A_V) for every star with a usable twin set; then veto mid-IR excess.

    The fit is run on the *whole* usable sample, not only the underluminous
    tail, because the distribution of fitted ``g`` over stars with no deficit is
    the empirical null: it is what tells us whether a 3σ tail is real or is the
    tail of a mis-estimated error model.
    """
    cfg_twin = cfg_twin or TwinConfig()
    ref = cfg_twin.band
    fit_bands = [b for b in FIT_BANDS if f"{b}_mag" in df.columns]
    exc_bands = [e.name for e in EXCESS_BANDS if f"{e.name}_mag" in df.columns]

    usable = df["twin_verdict"] == "ok"
    idx = np.flatnonzero(usable.to_numpy())
    if max_fit is not None and idx.size > max_fit:
        rng = np.random.default_rng(3)
        keep = set(idx[np.argsort(-df["z_twin"].to_numpy()[idx])][:max_fit // 2])
        keep |= set(rng.choice(idx, size=max_fit // 2, replace=False).tolist())
        idx = np.array(sorted(keep))

    recs = []
    for i in idx:
        row = df.iloc[i]
        dm_ref = float(row["dm_twin"])
        if not np.isfinite(dm_ref):
            continue
        names, dm, sig = _per_band_residuals(row, fit_bands, dm_ref, ref)
        if len(names) < 3:
            continue
        sig_mu = float(row.get("dm_twin_err", 0.1))
        # dm_twin_err already contains the distance term; the GLS needs it
        # separately, so pass the parallax part and use the twin scatter for
        # the per-band diagonal.
        scat = float(row.get("twin_scatter", 0.05) or 0.05)
        n_tw = max(int(row.get("n_twins", 50) or 50), 1)
        sig_mu_only = (5.0 / np.log(10.0)) * float(
            row.get("parallax_error", 0.0)) / max(float(row.get("parallax", 1.0)), 1e-9)
        # The twin scatter enters *every* band identically, because every band's
        # residual is built from the same reference-band deficit dm_ref. It is
        # therefore common-mode, exactly like the distance modulus, and belongs
        # in the rank-1 correlated term. Putting it on the diagonal instead
        # would treat one shared draw as N independent ones and inflate every
        # significance in the channel by ~sqrt(N_bands).
        common = float(np.hypot(sig_mu_only, scat * np.sqrt(1.0 + 1.0 / n_tw)))
        diag = np.asarray(sig, dtype=float)
        av_prior = None
        if np.isfinite(row.get("azero_gspphot", np.nan)):
            av_prior = (float(row["azero_gspphot"]), 0.15)
        fit = fit_grey_reddening(names, dm, diag, dist_modulus_sigma=common,
                                 av_prior=av_prior)
        rec = {"source_id": row["source_id"], "row": int(i),
               "grey_mag": fit.grey_mag, "grey_err": fit.grey_err,
               "grey_sigma": fit.significance, "av_fit": fit.av,
               "av_err": fit.av_err, "rho_grey_av": fit.rho_grey_av,
               "grey_chi2": fit.chi2, "grey_dof": fit.dof,
               "grey_nbands": fit.n_bands, "grey_lever": fit.lever_arm,
               "grey_verdict": fit.verdict, "f_cov": fit.covering_fraction,
               "f_cov_err": fit.covering_fraction_err,
               "dm_twin": dm_ref, "dm_twin_err": sig_mu}
        # Leg 2 — mid-IR excess measured here, after removing the fitted
        # grey + reddening. An excess makes the star *brighter* than its twins,
        # i.e. a significantly NEGATIVE residual.
        rec["midir_excess"] = False
        rec["midir_worst_sigma"] = np.nan
        worst = np.nan
        for b in exc_bands:
            mb, mr = row.get(f"{b}_mag"), row.get(f"{ref}_mag")
            med = row.get(f"{b}_col_med")
            if not all(np.isfinite([mb if mb is not None else np.nan,
                                    mr if mr is not None else np.nan,
                                    med if med is not None else np.nan])):
                continue
            dmb = dm_ref + (float(mb) - float(mr) - float(med))
            model = fit.grey_mag + fit.av * a_over_av(b)
            eb = float(row.get(f"{b}_mag_error", 0.05) or 0.05)
            sc = row.get(f"{b}_col_scatter")
            sc = float(sc) if sc is not None and np.isfinite(sc) else 0.05
            z = (dmb - model) / float(np.hypot(np.hypot(eb, sc), fit.grey_err))
            rec[f"dm_{b}_resid_sigma"] = z
            if not np.isfinite(worst) or z < worst:
                worst = z
        rec["midir_worst_sigma"] = worst
        rec["midir_excess"] = bool(np.isfinite(worst) and worst < -excess_nsigma)
        recs.append(rec)

    fits = pd.DataFrame(recs)
    fits.to_parquet(out / "greyfit.parquet", index=False)
    if len(fits):
        n_ok = int((fits["grey_verdict"] == "ok").sum())
        n_sig = int((fits["grey_sigma"] > z_min).sum())
        print(f"[cenotaph] stage grey: {len(fits)} fitted, {n_ok} clean, "
              f"{n_sig} with grey > {z_min} sigma; "
              f"{int(fits['midir_excess'].sum())} vetoed for mid-IR excess")
    else:
        print("[cenotaph] stage grey: no fits produced")
    return fits


# --------------------------------------------------------------------------
# Stage 5: far-IR closure (leg 3)
# --------------------------------------------------------------------------
def stage_farir(df: pd.DataFrame, fits: pd.DataFrame, out: Path,
                synthetic: bool = False, t_assumed_k: float = 50.0,
                far_ir_table: pd.DataFrame | None = None,
                max_beam_checks: int = 400) -> pd.DataFrame:
    """Associate surviving candidates with far-IR catalogues and close the budget."""
    if fits.empty:
        _write_json(out / "farir_meta.json",
                    {"verdict": "no_candidates_to_test", "n": 0})
        return pd.DataFrame()

    merged = fits.merge(
        df[["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "m_abs"]],
        on="source_id", how="left")

    matches = pd.DataFrame()
    coverage = {}
    if far_ir_table is not None:
        matches = far_ir_table
        coverage["source"] = "supplied"
    elif synthetic:
        coverage["source"] = "synthetic_none"
    else:
        from .acquire import FAR_IR_CATALOGS, crossmatch_far_ir, fetch_far_ir_catalog

        parts = []
        for name in FAR_IR_CATALOGS:
            try:
                cat = fetch_far_ir_catalog(
                    name, checkpoint=out / "checkpoints" / f"{name}.parquet")
                m = crossmatch_far_ir(merged, cat, name)
                if len(m):
                    parts.append(m)
                coverage[name] = int(len(m))
            except Exception as exc:  # noqa: BLE001
                print(f"[cenotaph] far-IR catalogue {name} unavailable: {exc!r}")
                coverage[name] = "unreachable"
        if parts:
            matches = parts[0]
            for p in parts[1:]:
                matches = matches.merge(p, on="source_id", how="outer")

        # Beam crowding, measured rather than assumed. A far-IR flux attributed
        # to a star that shares its beam with another catalogued source is not
        # attributable to that star -- and with a 25-40" beam that is common.
        # Only the handful of stars with an actual association need the cone
        # search, so this is affordable; it is a funnel stage, not follow-up,
        # because background-galaxy confusion destroyed every Hephaistos
        # candidate and the far-IR beams here are 10^2-10^4x larger in area.
        if len(matches):
            from .acquire import FAR_IR_CATALOGS as _CATS
            from .acquire import count_beam_neighbours

            radius = max(float(c["match_radius_arcsec"]) for c in _CATS.values())
            pos = merged.set_index("source_id")[["ra", "dec"]]
            counts = {}
            for sid in list(matches["source_id"])[:max_beam_checks]:
                try:
                    p = pos.loc[sid]
                    counts[sid] = count_beam_neighbours(
                        float(p["ra"]), float(p["dec"]), radius,
                        exclude_source_id=int(sid))
                except Exception as exc:  # noqa: BLE001
                    print(f"[cenotaph] beam count failed for {sid}: {exc!r}")
            if counts:
                matches["n_beam_neighbours"] = matches["source_id"].map(counts)
                coverage["beam_counts_measured"] = int(len(counts))
                coverage["beam_radius_arcsec"] = radius

    rows = []
    for _, r in merged.iterrows():
        dpc = 1000.0 / float(r["parallax"]) if r.get("parallax", 0) else np.nan
        lsun = float(luminosity_lsun(np.array([r.get("m_abs", np.nan)]))[0])
        if not np.isfinite(lsun) or not np.isfinite(dpc):
            continue
        fluxes, errs = {}, {}
        extra = {}
        if len(matches):
            hit = matches[matches["source_id"] == r["source_id"]]
            if len(hit):
                h = hit.iloc[0]
                for k in ("n_beam_neighbours",):
                    if k in h.index:
                        extra[k] = h.get(k)
                for k in h.index:
                    if k.endswith("_n_within") or k.endswith("_sep_arcsec"):
                        extra[k] = h.get(k)
                for b in FAR_IR_BANDS:
                    v = h.get(b.name)
                    if v is not None and np.isfinite(v):
                        fluxes[b.name] = float(v)
                        e = h.get(f"{b.name}_err")
                        errs[b.name] = float(e) if e is not None and np.isfinite(e) \
                            else 0.2 * float(v)
        cl = close_budget(max(float(r["f_cov"]), 0.0), float(r["f_cov_err"]),
                          lsun, dpc, t_assumed_k=t_assumed_k,
                          far_ir_fluxes_jy=fluxes, far_ir_flux_errs_jy=errs)
        rows.append({"source_id": r["source_id"], "d_pc": dpc, "l_lsun": lsun,
                     **{k: v for k, v in cl.to_dict().items() if k != "notes"},
                     **extra, "notes": "; ".join(cl.notes)})
    res = pd.DataFrame(rows)
    if len(res):
        res.to_parquet(out / "farir.parquet", index=False)
    _write_json(out / "farir_meta.json",
                {"coverage": coverage, "n_tested": int(len(res)),
                 "t_assumed_k": t_assumed_k,
                 "n_decidable": int(res["decidable"].sum()) if len(res) else 0,
                 "verdicts": (res["verdict"].value_counts().to_dict()
                              if len(res) else {})})
    print(f"[cenotaph] stage farir: {len(res)} tested, "
          f"{int(res['decidable'].sum()) if len(res) else 0} inside a far-IR horizon")
    return res


# --------------------------------------------------------------------------
# Stage 6: reduce
# --------------------------------------------------------------------------
def stage_reduce(df: pd.DataFrame, fits: pd.DataFrame, farir: pd.DataFrame,
                 out: Path, z_min: float = 3.0,
                 thr: VetThresholds | None = None) -> dict:
    thr = thr or VetThresholds()
    funnel = {"n_sample": int(len(df))}

    if fits.empty:
        summary = {"verdict": "NO_DATA_REACHED", "funnel": funnel,
                   "note": "no grey fits were produced; nothing can be claimed"}
        _write_json(out / "summary.json", summary)
        return summary

    merged = df.merge(fits, on="source_id", how="inner", suffixes=("", "_fit"))
    funnel["n_fitted"] = int(len(merged))

    # Beam-neighbour counts are measured in the far-IR stage, only for stars
    # that actually have an association; fold them in before vetting so the
    # crowding test is a real measurement rather than a missing column.
    if len(farir) and "n_beam_neighbours" in farir.columns:
        merged = merged.merge(farir[["source_id", "n_beam_neighbours"]],
                              on="source_id", how="left")

    vet = vet_table(merged, thr)
    merged = pd.concat([merged.reset_index(drop=True),
                        vet.reset_index(drop=True)], axis=1)

    step = merged
    funnel["n_twin_ok"] = int((step["twin_verdict"] == "ok").sum())
    step = step[step["twin_verdict"] == "ok"]
    funnel["n_grey_fit_clean"] = int((step["grey_verdict"] == "ok").sum())
    step = step[step["grey_verdict"] == "ok"]
    funnel["n_leg1_grey_significant"] = int((step["grey_sigma"] > z_min).sum())
    step = step[step["grey_sigma"] > z_min]
    funnel["n_leg2_no_midir_excess"] = int((~step["midir_excess"]).sum())
    step = step[~step["midir_excess"]]
    funnel["n_not_param_edge"] = int((~step["param_edge"]).sum())
    step = step[~step["param_edge"]]
    funnel["n_vet_core"] = int(step["pass_core"].sum())
    step = step[step["pass_core"]]

    # The candidate list is legs 1 and 2 plus the core vetting. Leg 3 is an
    # *annotation* on that list, not a gate applied before it: a star with no
    # far-IR coverage, or one outside the horizon for its f and distance, has
    # not failed the energy test — the test simply could not be run, and
    # dropping it silently would turn missing data into a rejection.
    candidates = step.copy()
    funnel["n_leg3_closes"] = 0
    if len(farir):
        keep = [c for c in ["source_id", "verdict", "closure_ratio", "f_ir",
                            "decidable", "horizon_pc", "far_ir_band",
                            "far_ir_flux_jy", "n_beam_neighbours"]
                if c in farir.columns]
        candidates = candidates.merge(
            farir[keep].rename(columns={"verdict": "closure_verdict"}),
            on="source_id", how="left", suffixes=("", "_far"))
        cv = candidates.get("closure_verdict")
        if cv is not None:
            funnel["n_leg3_decidable"] = int(
                candidates.get("decidable", pd.Series(dtype=bool)).fillna(False).sum())
            funnel["n_leg3_closes"] = int((cv == "closes").sum())
            # A closure is only believable in an uncrowded beam at high
            # latitude: background-galaxy confusion killed every Hephaistos
            # candidate, and these beams are 10^2-10^4x larger in area.
            clean_ctx = candidates["pass_far_ir_context"].fillna(False)
            funnel["n_leg3_closes_clean_beam"] = int(
                ((cv == "closes") & clean_ctx).sum())
            funnel["closure_verdicts"] = {
                str(k): int(v) for k, v in cv.value_counts().items()}

    # Empirical null: the fitted-grey distribution over the *whole* fitted set.
    g = pd.to_numeric(fits["grey_sigma"], errors="coerce").to_numpy(float)
    g = g[np.isfinite(g)]
    null = {
        "n": int(g.size),
        "median": float(np.median(g)) if g.size else None,
        "mad_sigma": float(1.4826 * np.median(np.abs(g - np.median(g))))
        if g.size else None,
        "n_above_3": int((g > 3).sum()), "n_below_minus3": int((g < -3).sum()),
    }
    if null["n_below_minus3"]:
        null["asymmetry_ratio"] = null["n_above_3"] / null["n_below_minus3"]
        null["note"] = (
            "the overluminous tail (grey < 0) is the binary/blend control: an "
            "occulter population shows up as an EXCESS of the underluminous tail "
            "over the overluminous one, so a ratio near 1 means no signal "
            "regardless of how many stars pass 3 sigma"
        )

    # What the channel could ever have detected, computed rather than asserted.
    scat = float(np.nanmedian(df.get("twin_scatter", pd.Series([0.06]))))
    sens = {
        "median_twin_scatter_mag": scat if np.isfinite(scat) else None,
        "min_detectable_f_poe20": minimum_detectable_f(scat if np.isfinite(scat)
                                                       else 0.06, 20.0),
        "min_detectable_f_poe50": minimum_detectable_f(scat if np.isfinite(scat)
                                                       else 0.06, 50.0),
        "min_detectable_f_poe100": minimum_detectable_f(scat if np.isfinite(scat)
                                                        else 0.06, 100.0),
        "zackrisson2018_floor_f": 0.75,
        "note": ("Zackrisson et al. (2018) state f_cov > 0.75 is required for a "
                 "factor-2 distance discrepancy with 20-30% spectrophotometric "
                 "distances; the twin estimator replaces that error budget with "
                 "the twin scatter plus the parallax term"),
    }

    confusion = {
        "expected_chance_akari_matches": expected_false_matches(
            funnel["n_fitted"], FAR_IR_SOURCE_DENSITY_PER_SQDEG["akari"], 25.0),
        "expected_chance_iras_matches": expected_false_matches(
            funnel["n_fitted"], FAR_IR_SOURCE_DENSITY_PER_SQDEG["iras"], 40.0),
        "p_background_galaxy_in_akari_beam": background_galaxy_probability(25.0),
        "p_background_galaxy_in_iras_beam": background_galaxy_probability(40.0),
        "note": ("background-galaxy confusion destroyed every Project Hephaistos "
                 "candidate; with a 25-40 arcsec far-IR beam the chance-match "
                 "expectation is large, so a positional association is never "
                 "evidence on its own -- only the closure ratio is"),
    }

    verdict = "no_candidates"
    if funnel.get("n_leg3_closes_clean_beam", 0) > 0:
        verdict = "candidates_with_energy_closure"
    elif funnel.get("n_leg3_closes", 0) > 0:
        verdict = "closure_but_crowded_beam"
    elif len(candidates):
        verdict = "leg1_leg2_survivors_far_ir_pending"

    cols = [c for c in ["source_id", "ra", "dec", "b_gal", "teff", "logg", "mh",
                        "alphafe", "parallax", "parallax_over_error", "ruwe",
                        "dm_twin", "dm_twin_err", "z_twin", "n_twins",
                        "twin_scatter", "grey_mag", "grey_err", "grey_sigma",
                        "av_fit", "av_err", "f_cov", "f_cov_err",
                        "midir_worst_sigma", "closure_verdict", "closure_ratio",
                        "f_ir", "horizon_pc", "v_tan_kms", "label"]
            if c in candidates.columns]
    if len(candidates):
        candidates[cols].to_csv(out / "candidates.csv", index=False)
        candidates.to_parquet(out / "candidates.parquet", index=False)

    # The acquisition ledger travels into every summary, not just the failing
    # ones. Row counts at each stage of the pull, the transport that served
    # them, and any chunk that failed or was truncated are part of the result:
    # a funnel quoted without its parent-sample completeness is unauditable.
    acq_block = {}
    meta_path = out / "sample_meta.json"
    if meta_path.exists():
        try:
            _m = json.loads(meta_path.read_text())
            acq_block = {"sample_verdict": _m.get("verdict"),
                         "n_sample_rows": _m.get("n"),
                         "cuts": _m.get("cuts"),
                         "parent": _m.get("acquisition", {}),
                         "photometry": _m.get("photometry_acquisition", {})}
        except Exception as exc:  # noqa: BLE001
            acq_block = {"error": f"sample_meta.json unreadable: {exc!r}"}

    summary = {
        "verdict": verdict,
        "acquisition": acq_block,
        "funnel": funnel,
        "null_distribution": null,
        "sensitivity": sens,
        "confusion_budget": confusion,
        "vet_coverage": vet_coverage(vet),
        "n_candidates": int(len(candidates)),
        "wise_wien_ceilings_k": wise_temperature_ceilings(),
        "cold_regime": {
            "t_k": [30.0, 50.0, 80.0, 100.0],
            "radius_au_solar": [radius_for_temperature(1.0, t)
                                for t in (30.0, 50.0, 80.0, 100.0)],
            "note": ("at stellar scale the searched regime is 100-1000 K, "
                     "capped by WISE W4 at 22 um. Cirkovic & Bradbury (2006) "
                     "argue on Landauer/Brillouin grounds that computation is "
                     "more efficient at a colder reservoir and that ATCs migrate "
                     "to cold regions (they anchor on molecular clouds, ~10 K); "
                     "they quote no specific shell temperature. The 30-100 K "
                     "window searched here is set by what the far-IR all-sky "
                     "catalogues can reach, not by a quotation"),
        },
        "far_ir_horizons": coverage_table(),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps({"verdict": verdict, "funnel": funnel,
                      "n_candidates": int(len(candidates))}, indent=2))
    return summary


# --------------------------------------------------------------------------
def cenotaph_run(cfg=None, stage: str = "all", out_dir: str | Path | None = None,
                 synthetic: bool = False, n_synth: int = 4000, seed: int = 7,
                 z_min: float = 3.0, t_assumed_k: float = 50.0,
                 max_fit: int | None = None, **kw) -> dict:
    """Run one or all CENOTAPH stages, reading checkpoints where they exist."""
    out = _out_dir(cfg, out_dir)
    stages = list(STAGES) if stage == "all" else [stage]

    def _load(name: str) -> pd.DataFrame:
        p = out / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    if stage == "probe":
        return stage_probe(out, **kw)

    df = pd.DataFrame()
    if "sample" in stages:
        df = stage_sample(out, synthetic=synthetic, n_synth=n_synth, seed=seed, **kw)
    else:
        df = _load("sample")
    if df.empty:
        summary = _empty_sample_summary(out)
        _write_json(out / "summary.json", summary)
        print(json.dumps({"verdict": summary["verdict"]}, indent=2))
        return summary

    tw = _load("twins")
    if "twins" in stages:
        tw = stage_twins(df, out)
    if tw.empty:
        tw = df

    fits = _load("greyfit")
    if "grey" in stages or "midir" in stages:
        fits = stage_grey(tw, out, z_min=z_min, max_fit=max_fit)

    farir = _load("farir")
    if "farir" in stages:
        farir = stage_farir(tw, fits, out, synthetic=synthetic,
                            t_assumed_k=t_assumed_k,
                            max_beam_checks=kw.get("max_beam_checks", 400))

    if "reduce" in stages:
        return stage_reduce(tw, fits, farir, out, z_min=z_min)
    return {"verdict": "stage_complete", "stages": stages}


__all__ = ["STAGES", "cenotaph_run", "luminosity_lsun", "stage_farir",
           "stage_grey", "stage_probe", "stage_reduce", "stage_sample",
           "stage_twins"]
