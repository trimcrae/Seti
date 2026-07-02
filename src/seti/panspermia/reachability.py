"""Directed (technological) dispersal: reachability + destination quality.

The passive channels (gravitational capture, geometric interception) treat life as
cargo that has to be *caught*, so they are killed by relative velocity and
encounter distance.  A **technological** disperser is different: it chooses a
target, aims, and decelerates on arrival, so relative velocity is not a barrier at
all.  Two quantities then decide where a K2-18 civilisation could go:

* **Reachability** -- the crossing time at a given cruise speed.  Light crosses
  1 pc in 3.262 yr, so a ship at fraction ``f`` of light speed crosses ``d`` pc in
  ``3.262 d / f`` yr.  Because the neighbourhood reshuffles, the cheapest moment to
  make the trip is each star's *closest approach* (minimum crossing distance
  ``d_min`` at epoch ``t_enc``) -- the optimal launch window.  At any plausible
  cruise speed every neighbour is reachable in astronomically trivial time, so
  reachability does not discriminate; it is reported, not used as a cut.

* **Destination quality** -- which is therefore the real filter.  A good target is
  a long-lived, stable host star (not a short-lived hot star, not an evolved giant,
  not a stellar remnant), ideally already known to host a habitable-zone planet.
  Here we score it from Gaia photometry alone (main-sequence F/G/K/M);
  known-planet-host information is layered on by the runner-side Exoplanet-Archive
  cross-match (:mod:`seti.panspermia.exohosts`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LY_PER_PC = 3.2615638             # light-years per parsec == light crossing yr per pc
DEFAULT_SPEEDS_C = (0.001, 0.01, 0.1)     # chemical/ion, fusion/sail, laser-sail regimes

# Main-sequence ridge: absolute Gaia G as a function of BP-RP colour, from a few
# well-known anchors (A0 ~ 0.0/+1.1; F ~ 0.5/+3.5; Sun ~ 0.82/+4.67; K ~ 1.5/+7.0;
# M ~ 2.5/+11 ... 3.5/+13.5).  Used only to separate dwarfs from giants/remnants.
_MS_BPRP = np.array([-0.1, 0.0, 0.5, 0.82, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
_MS_ABSG = np.array([0.6, 1.1, 3.5, 4.67, 5.4, 7.0, 8.8, 11.0, 12.5, 13.6, 14.5])


def crossing_times(df: pd.DataFrame, speeds_c=DEFAULT_SPEEDS_C,
                   distance_col: str = "d_min_pc") -> pd.DataFrame:
    """Add crossing time (yr) at each cruise speed for the optimal (closest) window.

    ``distance_col`` defaults to ``d_min_pc`` (the minimum crossing distance at
    closest approach); pass ``sep_now_pc`` for a launch-today figure.
    """
    out = df.copy()
    d = pd.to_numeric(out[distance_col], errors="coerce").to_numpy(float)
    for f in speeds_c:
        out[f"cross_yr_{f:g}c"] = _LY_PER_PC * d / f
    return out


def _ms_residual(bp_rp: np.ndarray, abs_g: np.ndarray) -> np.ndarray:
    """abs_G minus the main-sequence ridge at the same colour (mag).

    ~0 => on the main sequence; strongly negative => over-luminous (giant);
    strongly positive => under-luminous (white dwarf / subdwarf)."""
    ridge = np.interp(bp_rp, _MS_BPRP, _MS_ABSG, left=np.nan, right=np.nan)
    return abs_g - ridge


def destination_quality(df: pd.DataFrame, target: str = "hycean") -> pd.DataFrame:
    """Score each star as a *destination* from Gaia photometry alone.

    Adds ``abs_g``, ``ms_residual``, ``lum_class`` (main_sequence / giant /
    remnant / hot_shortlived / unknown) and ``dest_score`` in [0, 1].  It is a
    *host-star* prior, made sharper later by the known-planet cross-match.

    ``target`` sets whose notion of habitability to encode:

    * ``"hycean"`` (default) -- the traveller evolved on K2-18 b, so it seeks other
      hycean worlds, which are found around **cool K/M dwarfs** (like K2-18, an
      M2.5 dwarf).  The colour preference peaks in the K7-M range and stays high
      through the mid-M dwarfs.
    * ``"classical"`` -- an Earth-analog prior peaking at Sun-like F/G/K stars.

    Either way, over-luminous giants, hot short-lived stars and remnants score ~0.
    """
    out = df.copy()
    g = pd.to_numeric(out.get("phot_g_mean_mag"), errors="coerce").to_numpy(float)
    dist = pd.to_numeric(out.get("dist_pc"), errors="coerce").to_numpy(float)
    bp_rp = pd.to_numeric(out.get("bp_rp"), errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_g = g - 5.0 * np.log10(dist) + 5.0
    resid = _ms_residual(bp_rp, abs_g)

    lum = np.full(len(out), "unknown", dtype=object)
    on_ms = np.abs(resid) <= 1.5
    lum[np.isfinite(resid) & (resid < -1.5)] = "giant"
    lum[np.isfinite(resid) & (resid > 3.0)] = "remnant"
    lum[on_ms] = "main_sequence"
    # Hot, short-lived MS stars (blue of ~F5, bp_rp<0.45) are poor long-term hosts.
    lum[on_ms & (bp_rp < 0.45)] = "hot_shortlived"

    # Colour desirability.  For a hycean traveller, peak on cool K/M dwarfs (the
    # hycean-world hosts, like K2-18 itself), broad enough to span K7-mid M; for
    # the classical Earth-analog prior, peak on Sun-like F/G/K.
    if target == "classical":
        centre, width = 1.0, 0.7          # F/G/K
    else:
        centre, width = 2.6, 1.1          # K7-M (hycean hosts); K2-18 ~ bp_rp 2.9
    colour_pref = np.exp(-((bp_rp - centre) ** 2) / (2 * width ** 2))
    score = np.where(lum == "main_sequence", colour_pref, 0.0)
    score = np.where(lum == "hot_shortlived", 0.1 * colour_pref, score)
    out["abs_g"] = abs_g
    out["ms_residual"] = resid
    out["lum_class"] = lum
    out["dest_score"] = np.clip(score, 0.0, 1.0)
    return out


def rank_targets(df: pd.DataFrame, speeds_c=DEFAULT_SPEEDS_C,
                 past_only: bool = True, target: str = "hycean") -> pd.DataFrame:
    """Full directed-travel target ranking: crossing times + destination quality.

    ``past_only`` keeps stars whose optimal (closest-approach) window is in the
    past -- those a K2-18 civilisation could *already* have reached.  ``target``
    selects the habitability prior (``"hycean"`` default, or ``"classical"``)."""
    out = crossing_times(df, speeds_c=speeds_c)
    out = destination_quality(out, target=target)
    if past_only and "t_enc_myr" in out.columns:
        out = out[pd.to_numeric(out["t_enc_myr"], errors="coerce") < 0]
    # Best destination first; among equally-good destinations prefer the closer
    # optimal-window crossing (a shorter trip).
    sort_cols = ["dest_score"] + (["d_min_pc"] if "d_min_pc" in out.columns else [])
    asc = [False] + ([True] if "d_min_pc" in out.columns else [])
    return out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


__all__ = ["crossing_times", "destination_quality", "rank_targets",
           "DEFAULT_SPEEDS_C"]
