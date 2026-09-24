"""RELAY target list: short links whose beam contains Earth.

Why this stage exists
---------------------
The hit re-cut (``stage_assess``) is empty by construction: the Gaia
kinematic drift is 10^-8 Hz/s, so the drift window is the pipeline resolution
around the Earth term, and at the wide beams ~every BL target is already the
transmitter of some qualifying pair (run 35745111146: 99.7 % at 1.48 deg).
Neither test selects anything -- ``chance.py`` measures that.

This stage asks a question that is NOT empty.  Suppose nodes talk to their
*nearest neighbours* (link cost grows as distance^2, so a network does).  For
a link T -> R only a few parsecs long, Earth -- tens of parsecs away -- is in
the beam only if R lies almost exactly on the T -> Earth sightline: the
probability for a random link is (1 - cos(theta/2)) / 2, 4e-7 for a
100-m L-band dish.  Over the ~1.3 million nearest-neighbour links of the
100 pc sample that is ~0.5 expected links at the narrow beam and ~630 at 5 deg.
So which stars have such a link is a sharp, finite, falsifiable list -- the
specific systems whose ordinary neighbour traffic would cross Earth -- and it
can be set against what Breakthrough Listen has pointed at.

Rigor
-----
* The transmitter angle is exact (``geometry.transmitter_angle``), and the
  probability that the link really contains Earth is a Monte Carlo over both
  parallaxes (``p_in_beam`` per beam).  The ordering of T and R along the
  sightline -- which decides whether the link points AT Earth or away from it
  -- is exactly what a parallax error can flip, and the Monte Carlo carries it.
* A pair whose parallax difference is not significant has an UNMEASURED radial
  separation; its apparent alignment is parallax noise.  A comoving pair (small
  tangential-velocity difference at small projected separation) is most likely
  a bound system whose true separation is far below the parallax-noise depth.
  Both are counted and excluded from the ranked list, never ranked.
* The isotropic expectation sum_links (1 - cos(theta/2))/2 is printed beside
  every count.  A count at expectation says the list is geometric coincidence
  -- which is what it is expected to be.  The list is a TARGET list, not
  evidence.
* "Observed by BL" is the resolved BL target list (and the telescope's
  half-power beam around each).  BL names that did not resolve to Gaia cannot
  be checked; the count is stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as geo


def p_iso(theta_rad: float) -> float:
    """Probability that a random direction lies within theta/2 of a given one."""
    return 0.5 * (1.0 - np.cos(0.5 * float(theta_rad)))


def neighbour_links(xyz: np.ndarray, k: int, link_max_pc: float, min_sep_pc: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed links T -> R for each T's k nearest neighbours within link_max_pc.

    Returns (t_idx, r_idx, length_pc, rank) where rank 1 is the nearest.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    kk = min(int(k) + 1, len(xyz))
    d, j = tree.query(xyz, k=kk, distance_upper_bound=float(link_max_pc))
    d = np.atleast_2d(d)
    j = np.atleast_2d(j)
    n = len(xyz)
    t = np.repeat(np.arange(n), kk).reshape(n, kk)
    ok = np.isfinite(d) & (j < n) & (j != t) & (d >= float(min_sep_pc))
    # rank among the kept neighbours (self and binaries removed)
    rk = np.cumsum(ok, axis=1)
    return t[ok], j[ok], d[ok], rk[ok]


def link_table(xyz: np.ndarray, t: np.ndarray, r: np.ndarray, length: np.ndarray,
               rank: np.ndarray) -> pd.DataFrame:
    alpha = geo.transmitter_angle(xyz[t], xyz[r])
    d_t = np.linalg.norm(xyz[t], axis=1)
    d_r = np.linalg.norm(xyz[r], axis=1)
    # the beam passes Earth before reaching R when the link's projection on
    # T->Earth is longer than |T|
    proj = np.sum((xyz[r] - xyz[t]) * (-xyz[t]), axis=1) / np.maximum(d_t, 1e-12)
    between = proj > d_t
    return pd.DataFrame({"t_idx": t, "r_idx": r, "link_pc": length, "nn_rank": rank,
                         "alpha_rad": alpha, "d_t_pc": d_t, "d_r_pc": d_r,
                         "geometry": np.where(between, geo.GEOM_BETWEEN, geo.GEOM_SPILLOVER),
                         "flux_ratio_earth_over_receiver": (length / np.maximum(d_t, 1e-12)) ** 2})


def mc_in_beam(sample: pd.DataFrame, t: np.ndarray, r: np.ndarray, thetas: dict,
               n_mc: int, seed: int) -> dict[str, np.ndarray]:
    """P(alpha <= theta/2) per link per beam, over Gaussian parallax errors."""
    rng = np.random.default_rng(int(seed))
    ra = sample["ra"].to_numpy(float)
    dec = sample["dec"].to_numpy(float)
    plx = sample["parallax"].to_numpy(float)
    eplx = np.nan_to_num(sample["parallax_error"].to_numpy(float), nan=0.0)
    u = geo.unit_vectors(ra, dec)
    hits = {name: np.zeros(len(t)) for name in thetas}
    for _ in range(int(n_mc)):
        pt = plx[t] + eplx[t] * rng.standard_normal(len(t))
        pr = plx[r] + eplx[r] * rng.standard_normal(len(t))
        pt = np.where(pt > 0, pt, np.nan)
        pr = np.where(pr > 0, pr, np.nan)
        a = geo.transmitter_angle(u[t] * (1000.0 / pt)[:, None], u[r] * (1000.0 / pr)[:, None])
        for name, th in thetas.items():
            hits[name] += np.nan_to_num(a <= 0.5 * th, nan=False)
    return {name: v / float(n_mc) for name, v in hits.items()}


def comoving_flags(sample: pd.DataFrame, t: np.ndarray, r: np.ndarray, *, z_min: float,
                   dv_tan_max_kms: float, proj_sep_max_pc: float, m_tot_msun: float = 2.0,
                   dv_margin_kms: float = 1.0, min_sky_sep_arcsec: float = 0.0) -> pd.DataFrame:
    plx = sample["parallax"].to_numpy(float)
    eplx = sample["parallax_error"].to_numpy(float)
    z = np.abs(plx[t] - plx[r]) / np.sqrt(eplx[t] ** 2 + eplx[r] ** 2)
    u = geo.unit_vectors(sample["ra"].to_numpy(float), sample["dec"].to_numpy(float))
    sep = np.radians(geo.sky_separation_deg(u[t], u[r]))
    d_mean = 0.5 * (1000.0 / plx[t] + 1000.0 / plx[r])
    proj = d_mean * sep
    pmra = sample["pmra"].to_numpy(float)
    pmdec = sample["pmdec"].to_numpy(float)
    dmu = np.hypot(pmra[t] - pmra[r], pmdec[t] - pmdec[r])
    dv = geo.K_TAN * dmu * d_mean / 1000.0
    comoving = (dv <= float(dv_tan_max_kms)) & (proj <= float(proj_sep_max_pc))
    # bound-consistent: the tangential velocity difference is below the escape
    # speed at the projected separation for a system of mass m_tot (plus a
    # margin for the measurement).  Run 35992222801's first list was 13 pairs
    # 1-2" apart (20-150 AU projected) with dv_tan 3-5 km/s -- orbital motion,
    # not field pairs -- whose 3-6 sigma parallax differences are the known
    # close-pair astrometry bias.  v_esc = 42.1 km/s sqrt(M/Msun / (s/AU)).
    s_au = np.maximum(proj * 206264.806, 1e-6)
    v_esc = 42.1 * np.sqrt(float(m_tot_msun) / s_au)
    bound = dv <= v_esc + float(dv_margin_kms)
    close = (sep / geo.ARCSEC) < float(min_sky_sep_arcsec)
    return pd.DataFrame({"parallax_diff_sigma": z, "sky_sep_arcsec": sep / geo.ARCSEC,
                         "proj_sep_pc": proj, "dv_tan_kms": dv, "v_esc_kms": v_esc,
                         "bound_consistent": bound, "close_pair_astrometry": close,
                         "radial_separation_resolved": z >= float(z_min),
                         "comoving_likely_bound": comoving | bound | close})


def observed_mask(sample: pd.DataFrame, bl_idx, radius_arcmin: float) -> np.ndarray:
    """True for sample rows within radius of any resolved BL target."""
    from scipy.spatial import cKDTree

    n = len(sample)
    m = np.zeros(n, bool)
    idx = np.asarray([int(i) for i in bl_idx if int(i) >= 0], int)
    if not len(idx):
        return m
    u = geo.unit_vectors(sample["ra"].to_numpy(float), sample["dec"].to_numpy(float))
    tree = cKDTree(u[idx])
    d, _ = tree.query(u, k=1)
    return d <= geo.chord(float(radius_arcmin) * geo.ARCMIN)


def build_targetlist(sample: pd.DataFrame, beams: list[dict], tconf: dict, *,
                     bl_idx=(), bl_names_unresolved: int | None = None,
                     observed_radius_arcmin: float = 4.35) -> tuple[pd.DataFrame, dict]:
    """The ranked list and its report.  Pure: no I/O, no network."""
    xyz = geo.positions_pc(sample["ra"].to_numpy(float), sample["dec"].to_numpy(float),
                           sample["parallax"].to_numpy(float))
    radio = [b for b in beams if float(b["theta_rad"]) >= float(tconf.get("theta_min_rad", 1e-4))]
    radio = sorted(radio, key=lambda b: float(b["theta_rad"]))
    thetas = {str(b["name"]): float(b["theta_rad"]) for b in radio}
    th_max = max(thetas.values()) if thetas else 0.0
    t, r, length, rank = neighbour_links(xyz, int(tconf["k_neighbours"]), float(tconf["link_max_pc"]),
                                         float(tconf["min_separation_pc"]))
    n_links = int(len(t))
    lt = link_table(xyz, t, r, length, rank)
    # keep every link whose MEASURED alpha is inside the widest beam, or within
    # a parallax-error margin of it (the Monte Carlo then decides)
    keep = lt["alpha_rad"].to_numpy() <= float(tconf.get("alpha_prefilter_factor", 3.0)) * 0.5 * th_max
    lt = lt[keep].reset_index(drop=True)
    rep = {"n_stars": int(len(sample)), "n_links": n_links,
           "k_neighbours": int(tconf["k_neighbours"]), "link_max_pc": float(tconf["link_max_pc"]),
           "min_separation_pc": float(tconf["min_separation_pc"]),
           "link_length_p50_pc": float(np.median(length)) if n_links else None,
           "n_links_prefiltered": int(len(lt)), "n_mc": int(tconf["n_mc"]), "beams": {}}
    if len(lt):
        p = mc_in_beam(sample, lt["t_idx"].to_numpy(), lt["r_idx"].to_numpy(), thetas,
                       int(tconf["n_mc"]), int(tconf.get("seed", 60)))
        for name, v in p.items():
            lt[f"p_in_beam:{name}"] = v
        fl = comoving_flags(sample, lt["t_idx"].to_numpy(), lt["r_idx"].to_numpy(),
                            z_min=float(tconf["parallax_diff_sigma_min"]),
                            dv_tan_max_kms=float(tconf["comoving_dv_tan_max_kms"]),
                            proj_sep_max_pc=float(tconf["comoving_proj_sep_max_pc"]),
                            m_tot_msun=float(tconf.get("bound_m_tot_msun", 2.0)),
                            dv_margin_kms=float(tconf.get("bound_dv_margin_kms", 1.0)),
                            min_sky_sep_arcsec=float(tconf.get("min_sky_sep_arcsec", 0.0)))
        lt = pd.concat([lt, fl], axis=1)
        obs = observed_mask(sample, bl_idx, observed_radius_arcmin)
        lt["t_bl_observed"] = obs[lt["t_idx"].to_numpy()]
        lt["r_bl_observed"] = obs[lt["r_idx"].to_numpy()]
        for c in ("source_id", "phot_g_mean_mag", "bp_rp", "ra", "dec", "parallax", "parallax_error"):
            if c in sample:
                lt[f"t_{c}"] = sample[c].to_numpy()[lt["t_idx"].to_numpy()]
        lt["r_source_id"] = sample["source_id"].to_numpy()[lt["r_idx"].to_numpy()]
        lt["rankable"] = lt["radial_separation_resolved"] & ~lt["comoving_likely_bound"]
        # narrowest beam with P >= 0.5: the sharper the beam that still hits
        # Earth, the fewer links a random network would need to produce it
        order = list(thetas.keys())
        best = np.full(len(lt), len(order), int)
        for i, name in reversed(list(enumerate(order))):
            best = np.where(lt[f"p_in_beam:{name}"].to_numpy() >= 0.5, i, best)
        lt["narrowest_beam_p50"] = [order[i] if i < len(order) else None for i in best]
        lt["_best"] = best
    for name, th in thetas.items():
        col = f"p_in_beam:{name}"
        exp = n_links * p_iso(th)
        rec = {"theta_rad": th, "p_iso_per_link": p_iso(th), "n_expected_isotropic": exp}
        if len(lt):
            pv = lt[col].to_numpy()
            ok = lt["rankable"].to_numpy()
            rec.update({"sum_p_all": float(pv.sum()), "sum_p_rankable": float(pv[ok].sum()),
                        "n_links_p50": int((pv >= 0.5).sum()),
                        "n_links_p50_rankable": int(((pv >= 0.5) & ok).sum()),
                        "n_links_p50_rankable_t_unobserved": int(((pv >= 0.5) & ok &
                                                                  ~lt["t_bl_observed"].to_numpy()).sum())})
        rep["beams"][name] = rec
    if len(lt):
        rep["n_excluded_radial_unresolved"] = int((~lt["radial_separation_resolved"]).sum())
        rep["n_excluded_comoving"] = int(lt["comoving_likely_bound"].sum())
        rep["n_excluded_bound_consistent"] = int(lt["bound_consistent"].sum())
        rep["n_excluded_close_pair"] = int(lt["close_pair_astrometry"].sum())
        first = order[0] if order else None
        lt = lt.sort_values(["rankable", "_best", f"p_in_beam:{first}", "d_t_pc"],
                            ascending=[False, True, False, True]).drop(columns=["_best"]).reset_index(drop=True)
    rep["bl_targets_resolved_in_sample"] = int(len([i for i in bl_idx if int(i) >= 0]))
    rep["bl_names_unresolved"] = bl_names_unresolved
    rep["observed_radius_arcmin"] = observed_radius_arcmin
    return lt, rep
