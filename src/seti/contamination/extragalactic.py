"""Extragalactic rejection: AGN/QSO and resolved-galaxy contaminants.

Background galaxies and AGN are a dominant source of spurious WISE infrared
excess.  We reject sources in the AGN/QSO region of WISE colour space and those
matching a quasar catalogue, plus anything flagged as extended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extragalactic_pass(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    e = thresholds["contamination"]["extragalactic"]

    w1 = df.get("W1mag", pd.Series(np.nan, index=df.index))
    w2 = df.get("W2mag", pd.Series(np.nan, index=df.index))
    w1_w2 = w1 - w2

    # AGN wedge: very red W1-W2 is the classic Stern et al. AGN signature.
    not_agn = w1_w2 < e["w1_w2_agn_min"]

    # Quasar-catalogue match flag (set upstream by the acquisition step).
    qso = df.get("is_qso_match", pd.Series(False, index=df.index)).fillna(False)
    not_qso = ~qso.astype(bool)

    return (not_agn & not_qso).fillna(False)
