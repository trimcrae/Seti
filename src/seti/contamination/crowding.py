"""Crowding / blending cuts -- the Project Hephaistos failure mode.

A blended background source inside the WISE PSF injects fake infrared excess.
We reject WDs with more than one IR counterpart within the match radius (from
``gaiadr3.allwise_neighbourhood``) or a second Gaia source within the WISE PSF,
and require the independent unWISE forced photometry to agree with CatWISE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crowding_pass(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    c = thresholds["contamination"]["crowding"]

    n_wise = df.get("n_wise_neighbours", pd.Series(1, index=df.index)).fillna(1)
    gaia_nn = df.get("gaia_nn_arcsec", pd.Series(np.inf, index=df.index)).fillna(np.inf)

    neigh_ok = n_wise <= c["max_wise_neighbours"]
    isolated_ok = gaia_nn >= c["gaia_neighbour_radius_arcsec"]

    # unWISE vs CatWISE W1 agreement (fractional), when both present.
    w1_cat = df.get("W1mag", pd.Series(np.nan, index=df.index))
    w1_unw = df.get("W1mag_unwise", pd.Series(np.nan, index=df.index))
    have_both = np.isfinite(w1_cat) & np.isfinite(w1_unw)
    # |Delta flux| / flux ~ 0.4 ln10 |Delta mag|.
    frac_diff = 0.4 * np.log(10.0) * np.abs(w1_cat - w1_unw)
    agree_ok = pd.Series(True, index=df.index)
    agree_ok[have_both] = frac_diff[have_both] <= c["unwise_catwise_max_frac_diff"]

    return (neigh_ok & isolated_ok & agree_ok).fillna(False)
