"""Axial statistics for orbital-pole fields.

Astrometric orbits determine the pole only up to sign (the i vs 180-i and
Omega mod 180 ambiguities collapse the pole onto an AXIS, p == -p), so every
statistic here lives on the projective sphere: orientation tensors and the
Bingham test, never vector means.

Conventions
-----------
For a star at (ra, dec) the plane-of-sky tangent basis is (east, north, LOS)
with LOS pointing from the observer to the star.  An orbit with inclination
``i`` (0 = face-on) and position angle of the node ``Omega`` (from north
through east) has pole axis

    p = sin(i) sin(Omega) * east + sin(i) cos(Omega) * north + cos(i) * LOS

(the sign of the LOS term is unknowable without radial velocities — harmless,
because p is an axis).  Axes are then rotated into Galactic Cartesian so that
patches can be compared across the sky.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ICRS -> Galactic rotation (IAU 1958 pole/center, standard matrix).
_ICRS_TO_GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [+0.4941094279, -0.4448296300, +0.7469822445],
    [-0.8676661490, -0.1980763734, +0.4559837762],
])


def tangent_basis(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """(east, north, los) unit vectors in ICRS Cartesian, each (N, 3)."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    cos_d, sin_d = np.cos(dec), np.sin(dec)
    cos_a, sin_a = np.cos(ra), np.sin(ra)
    los = np.stack([cos_d * cos_a, cos_d * sin_a, sin_d], 1)
    east = np.stack([-sin_a, cos_a, np.zeros_like(ra)], 1)
    north = np.stack([-sin_d * cos_a, -sin_d * sin_a, cos_d], 1)
    return east, north, los


def pole_axes(ra_deg, dec_deg, inclination_deg, node_deg) -> np.ndarray:
    """Unit pole AXES in Galactic Cartesian, shape (N, 3).

    Sign is meaningless (axial data); the returned vectors are normalised and
    canonicalised to a non-negative Galactic z component for reproducibility.
    """
    east, north, los = tangent_basis(ra_deg, dec_deg)
    i = np.radians(np.asarray(inclination_deg, float))
    om = np.radians(np.asarray(node_deg, float))
    p_icrs = (np.sin(i)[:, None] * np.sin(om)[:, None] * east
              + np.sin(i)[:, None] * np.cos(om)[:, None] * north
              + np.cos(i)[:, None] * los)
    p_gal = p_icrs @ _ICRS_TO_GAL.T
    p_gal /= np.linalg.norm(p_gal, axis=1, keepdims=True)
    flip = p_gal[:, 2] < 0
    p_gal[flip] *= -1.0
    return p_gal


def orientation_tensor(axes: np.ndarray) -> np.ndarray:
    """Scatter tensor T = mean(p p^T); trace 1; isotropy -> eigenvalues 1/3."""
    a = np.asarray(axes, float)
    return a.T @ a / len(a)


def bingham_stat(axes: np.ndarray) -> float:
    """Bingham test statistic against axial isotropy.

    S = 15 N / 2 * sum_i (lambda_i - 1/3)^2, asymptotically chi^2 with 5
    degrees of freedom under isotropy.  The asymptotic null is used only as a
    sanity scale — real significance always comes from scanning-law-matched
    shuffles (the Gaia NSS inclination field is not isotropic by construction).
    """
    lam = np.linalg.eigvalsh(orientation_tensor(axes))
    return float(15.0 * len(axes) / 2.0 * np.sum((lam - 1.0 / 3.0) ** 2))


def principal_axis(axes: np.ndarray) -> np.ndarray:
    """Dominant eigenvector of the orientation tensor (the patch axis)."""
    t = orientation_tensor(axes)
    w, v = np.linalg.eigh(t)
    ax = v[:, int(np.argmax(w))]
    return ax if ax[2] >= 0 else -ax


def scan_coherence(pos_pc: np.ndarray, axes: np.ndarray, radius_pc: float,
                   n_min: int = 8) -> list[dict]:
    """Bingham statistic of every >=n_min neighbourhood (each star a centre).

    Returns per-neighbourhood records sorted by descending statistic; overlap
    dedup is the caller's job (same-members Jaccard, as in HERDSMAN).
    """
    tree = cKDTree(pos_pc)
    groups = tree.query_ball_point(pos_pc, r=radius_pc)
    out = []
    seen: set[tuple] = set()
    for ci, members in enumerate(groups):
        if len(members) < n_min:
            continue
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        sub = axes[members]
        lam = np.linalg.eigvalsh(orientation_tensor(sub))
        s = float(15.0 * len(sub) / 2.0 * np.sum((lam - 1.0 / 3.0) ** 2))
        out.append({"center": int(ci), "members": [int(m) for m in members],
                    "n": len(members), "stat": s,
                    "lambda1": float(lam[-1]),
                    "axis": [float(x) for x in principal_axis(sub)]})
    out.sort(key=lambda r: -r["stat"])
    return out


def shuffle_axes_within_bands(axes: np.ndarray, band_id: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Permute pole axes among stars sharing a scan-coverage band.

    The Gaia scanning law imprints ecliptic-latitude-dependent biases on NSS
    inclinations; shuffling only within bands builds a null that preserves the
    imprint while destroying any real spatial coherence.
    """
    out = axes.copy()
    for b in np.unique(band_id):
        idx = np.flatnonzero(band_id == b)
        out[idx] = axes[idx[rng.permutation(len(idx))]]
    return out


def ecliptic_latitude_deg(ra_deg, dec_deg) -> np.ndarray:
    """Ecliptic latitude (deg) — the scanning-law banding coordinate."""
    eps = np.radians(23.439281)
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    sb = (np.sin(dec) * np.cos(eps)
          - np.cos(dec) * np.sin(eps) * np.sin(ra))
    return np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))


def group_matrix(pos_pc: np.ndarray, radius_pc: float, n_min: int = 8):
    """Sparse membership matrix + member counts for all >=n_min neighbourhoods.

    Built once per radius; shuffle nulls then reuse it, because permuting
    axes among stars changes tensors but not neighbourhoods.
    """
    from scipy.sparse import csr_matrix

    tree = cKDTree(pos_pc)
    groups = tree.query_ball_point(pos_pc, r=radius_pc)
    seen: set[tuple] = set()
    rows, cols, centers = [], [], []
    gi = 0
    for ci, members in enumerate(groups):
        if len(members) < n_min:
            continue
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        rows.extend([gi] * len(members))
        cols.extend(members)
        centers.append(ci)
        gi += 1
    if gi == 0:
        return None, np.array([], int), np.array([], int)
    m = csr_matrix((np.ones(len(rows)), (rows, cols)),
                   shape=(gi, len(pos_pc)))
    counts = np.asarray(m.sum(axis=1)).ravel().astype(int)
    return m, counts, np.array(centers, int)


def bingham_stats_batch(m, counts: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Bingham statistic for every group at once (batched 3x3 eigenvalues)."""
    a = np.asarray(axes, float)
    outer = np.stack([a[:, 0] * a[:, 0], a[:, 1] * a[:, 1], a[:, 2] * a[:, 2],
                      a[:, 0] * a[:, 1], a[:, 0] * a[:, 2],
                      a[:, 1] * a[:, 2]], axis=1)
    s = m @ outer                                   # (n_groups, 6) sums
    n = counts.astype(float)[:, None]
    t = np.empty((len(counts), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = (s[:, 0] / n[:, 0],
                                          s[:, 1] / n[:, 0],
                                          s[:, 2] / n[:, 0])
    t[:, 0, 1] = t[:, 1, 0] = s[:, 3] / n[:, 0]
    t[:, 0, 2] = t[:, 2, 0] = s[:, 4] / n[:, 0]
    t[:, 1, 2] = t[:, 2, 1] = s[:, 5] / n[:, 0]
    lam = np.linalg.eigvalsh(t)
    return 15.0 * counts / 2.0 * ((lam - 1.0 / 3.0) ** 2).sum(axis=1)
