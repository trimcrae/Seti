"""Thiele-Innes -> geometric orbital elements for Gaia DR3 NSS solutions.

DR3 ``Orbital`` solutions publish the photocentre orbit as Thiele-Innes
coefficients (A, B, F, G) in mas.  The pole direction needs (i, Omega), via
the classical inversion:

    A = a0 ( cos w cos O - sin w sin O cos i)
    B = a0 ( cos w sin O + sin w cos O cos i)
    F = a0 (-sin w cos O - cos w sin O cos i)
    G = a0 (-sin w sin O + cos w cos O cos i)

    A + G = a0 (1 + cos i) cos(w + O)      B - F = a0 (1 + cos i) sin(w + O)
    A - G = a0 (1 - cos i) cos(w - O)      B + F = -a0 (1 - cos i) sin(w - O)

so with k1 = hypot(A+G, B-F) = a0 (1 + cos i) and
        k2 = hypot(A-G, B+F) = a0 (1 - cos i):

    a0 = (k1 + k2) / 2,   cos i = (k1 - k2) / (k1 + k2),
    w + O = atan2(B - F, A + G),   w - O = atan2(-(B + F), A - G),
    Omega = ((w+O) - (w-O)) / 2  (mod 180 -- the astrometric axial ambiguity).
"""

from __future__ import annotations

import numpy as np


def thiele_innes_to_geometric(a_ti, b_ti, f_ti, g_ti):
    """(a0, inclination_deg, node_deg, omega_deg) from Thiele-Innes A,B,F,G.

    Vectorised; angles in degrees, inclination in [0, 180], node in [0, 180)
    (axial), omega in [0, 360).
    """
    a = np.asarray(a_ti, float)
    b = np.asarray(b_ti, float)
    f = np.asarray(f_ti, float)
    g = np.asarray(g_ti, float)

    k1 = np.hypot(a + g, b - f)          # a0 (1 + cos i)
    k2 = np.hypot(a - g, b + f)          # a0 (1 - cos i)
    a0 = 0.5 * (k1 + k2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_i = np.where(a0 > 0, (k1 - k2) / np.where(a0 > 0, k1 + k2, 1.0),
                         np.nan)
    inc = np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))

    wpo = np.arctan2(b - f, a + g)       # w + Omega
    wmo = np.arctan2(-(b + f), a - g)    # w - Omega
    node = np.degrees(0.5 * (wpo - wmo)) % 180.0
    omega = np.degrees(0.5 * (wpo + wmo)) % 360.0
    return a0, inc, node, omega


def geometric_to_thiele_innes(a0, inc_deg, node_deg, omega_deg):
    """Forward construction (for tests and mocks)."""
    a0 = np.asarray(a0, float)
    i = np.radians(np.asarray(inc_deg, float))
    o = np.radians(np.asarray(node_deg, float))
    w = np.radians(np.asarray(omega_deg, float))
    ci = np.cos(i)
    a = a0 * (np.cos(w) * np.cos(o) - np.sin(w) * np.sin(o) * ci)
    b = a0 * (np.cos(w) * np.sin(o) + np.sin(w) * np.cos(o) * ci)
    f = a0 * (-np.sin(w) * np.cos(o) - np.cos(w) * np.sin(o) * ci)
    g = a0 * (-np.sin(w) * np.sin(o) + np.cos(w) * np.cos(o) * ci)
    return a, b, f, g
