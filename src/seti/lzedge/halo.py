"""Local dark-matter velocity distributions and their lab-frame halo integrals.

Every component is defined in the Galactic rest frame (x toward the Galactic
centre, y along rotation, z toward the north Galactic pole) and carries its own
density fraction.  The lab-frame speed distribution f_lab(v, t) and the
inverse-speed integral g(v_min, t) = ∫_{|v|>v_min} f_lab(v)/|v| d³v are what
the recoil rate needs.

The isotropic truncated Maxwellian has a closed-form lab-frame speed
distribution (the angular integral over the spherical cap inside the escape
sphere is elementary); anisotropic Gaussians (the Gaia Sausage, streams, an
LMC-boosted tail) are integrated numerically with the escape-sphere cap handled
*exactly* — the Gauss–Legendre nodes in cos(theta) are mapped onto the allowed
sub-interval for each speed, so the hard edge never falls between nodes.  The
edge is where an endothermic interpretation of a 248 keV recoil lives, so its
treatment is not a detail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf

from .kinematics import C_KMS, trapz

SQRT_PI = math.sqrt(math.pi)


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    e3 = a / n if n > 0 else np.array([0.0, 0.0, 1.0])
    helper = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e3, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


@dataclass
class Maxwellian:
    """Isotropic Maxwellian, f ∝ exp(-v²/v0²), truncated at |v| = v_esc (None: untruncated)."""
    v0: float
    v_esc: float | None = 544.0
    fraction: float = 1.0
    name: str = "maxwellian"

    @property
    def norm(self) -> float:
        """N such that ∫ f d³v = 1 for f = exp(-v²/v0²)/N."""
        base = (SQRT_PI * self.v0) ** 3
        if self.v_esc is None:
            return base
        z = self.v_esc / self.v0
        return base * (erf(z) - 2.0 * z * math.exp(-z * z) / SQRT_PI)

    def density(self, v_xyz: np.ndarray) -> np.ndarray:
        v2 = np.sum(np.asarray(v_xyz, dtype=float) ** 2, axis=-1)
        f = np.exp(-v2 / self.v0 ** 2) / self.norm
        if self.v_esc is not None:
            f = np.where(v2 < self.v_esc ** 2, f, 0.0)
        return f

    def lab_speed_distribution(self, v_lab: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
        """Closed form: f_lab(v) = v² ∫dΩ f(v n̂ + v_lab), ∫ f_lab dv = 1."""
        v = np.asarray(v_grid, dtype=float)
        vl = float(np.linalg.norm(v_lab))
        v0 = self.v0
        out = np.zeros_like(v)
        pos = v > 0
        if vl < 1e-9:
            f = v[pos] ** 2 * 4.0 * math.pi * np.exp(-v[pos] ** 2 / v0 ** 2) / self.norm
            if self.v_esc is not None:
                f = np.where(v[pos] < self.v_esc, f, 0.0)
            out[pos] = f
            return out
        vv = v[pos]
        if self.v_esc is None:
            mu_c = np.ones_like(vv)
        else:
            mu_c = np.clip((self.v_esc ** 2 - vv ** 2 - vl ** 2) / (2.0 * vv * vl), -1.0, 1.0)
        # ∫_{-1}^{mu_c} exp(-(v²+vl²+2 v vl mu)/v0²) dmu = v0²/(2 v vl) [e^{-(v-vl)²/v0²} - e^{-(v²+vl²+2 v vl mu_c)/v0²}]
        a = np.exp(-(vv - vl) ** 2 / v0 ** 2)
        b = np.exp(-(vv ** 2 + vl ** 2 + 2.0 * vv * vl * mu_c) / v0 ** 2)
        ang = 2.0 * math.pi * v0 ** 2 / (2.0 * vv * vl) * (a - b)
        ang = np.where(mu_c > -1.0, ang, 0.0)
        out[pos] = vv ** 2 * ang / self.norm
        return out


@dataclass
class Gaussian:
    """Anisotropic Gaussian with principal axes along x, y, z of the Galactic frame.

    ``mean`` is the bulk velocity (a stream, LMC debris); ``sigma`` the three
    dispersions; ``v_esc`` truncates at the Galactic escape sphere (None for an
    unbound component).
    """
    mean: tuple[float, float, float]
    sigma: tuple[float, float, float]
    v_esc: float | None = 544.0
    fraction: float = 1.0
    name: str = "gaussian"
    _norm: float | None = field(default=None, repr=False)

    def density_untruncated(self, v_xyz: np.ndarray) -> np.ndarray:
        v = np.asarray(v_xyz, dtype=float)
        mu = np.asarray(self.mean, dtype=float)
        sg = np.asarray(self.sigma, dtype=float)
        z2 = np.sum(((v - mu) / sg) ** 2, axis=-1)
        return np.exp(-0.5 * z2) / ((2.0 * math.pi) ** 1.5 * np.prod(sg))

    @property
    def norm(self) -> float:
        """Fraction of the untruncated Gaussian inside the escape sphere."""
        if self.v_esc is None:
            return 1.0
        if self._norm is None:
            grid = np.arange(0.0, self.v_esc + 0.5, 1.0)
            f = _numeric_lab_speed_distribution(self.density_untruncated, np.zeros(3), grid,
                                                self.v_esc, n_mu=48, n_phi=64)
            self._norm = float(trapz(f, grid))
        return self._norm

    def density(self, v_xyz: np.ndarray) -> np.ndarray:
        f = self.density_untruncated(v_xyz) / self.norm
        if self.v_esc is not None:
            v2 = np.sum(np.asarray(v_xyz, dtype=float) ** 2, axis=-1)
            f = np.where(v2 < self.v_esc ** 2, f, 0.0)
        return f

    def lab_speed_distribution(self, v_lab: np.ndarray, v_grid: np.ndarray,
                               n_mu: int = 48, n_phi: int = 64) -> np.ndarray:
        return _numeric_lab_speed_distribution(self.density_untruncated, np.asarray(v_lab, float),
                                               np.asarray(v_grid, float), self.v_esc,
                                               n_mu=n_mu, n_phi=n_phi) / self.norm


def _numeric_lab_speed_distribution(density, v_lab: np.ndarray, v_grid: np.ndarray,
                                    v_esc: float | None, n_mu: int = 48, n_phi: int = 64) -> np.ndarray:
    """v² ∫dΩ density(v n̂ + v_lab) with the escape-sphere cap integrated exactly."""
    v = np.asarray(v_grid, dtype=float)
    vl = float(np.linalg.norm(v_lab))
    e1, e2, e3 = _orthonormal_frame(v_lab if vl > 0 else np.array([0.0, 0.0, 1.0]))
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_mu)          # on [-1, 1]
    phi = 2.0 * math.pi * (np.arange(n_phi) + 0.5) / n_phi
    out = np.zeros_like(v)
    pos = v > 0
    vv = v[pos]
    if v_esc is None:
        mu_c = np.ones_like(vv)
    elif vl < 1e-9:
        mu_c = np.where(vv < v_esc, 1.0, -1.0)
    else:
        mu_c = np.clip((v_esc ** 2 - vv ** 2 - vl ** 2) / (2.0 * vv * vl), -1.0, 1.0)
    # map nodes onto [-1, mu_c] for each speed: mu = (mu_c+1)/2 * x + (mu_c-1)/2
    half = 0.5 * (mu_c + 1.0)                                    # (n_v,)
    mu = half[:, None] * x_gl[None, :] + (half[:, None] - 1.0)   # (n_v, n_mu)
    w = half[:, None] * w_gl[None, :]                            # (n_v, n_mu)
    s = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))
    # n̂ = mu e3 + s (cos phi e1 + sin phi e2); build velocity vectors (n_v, n_mu, n_phi, 3)
    cph, sph = np.cos(phi), np.sin(phi)
    nhat = (mu[:, :, None, None] * e3[None, None, None, :]
            + s[:, :, None, None] * (cph[None, None, :, None] * e1[None, None, None, :]
                                     + sph[None, None, :, None] * e2[None, None, None, :]))
    vel = vv[:, None, None, None] * nhat + v_lab[None, None, None, :]
    dens = density(vel)                                          # (n_v, n_mu, n_phi)
    ang = np.sum(w[:, :, None] * dens, axis=(1, 2)) * (2.0 * math.pi / n_phi)
    ang = np.where(half > 0.0, ang, 0.0)
    out[pos] = vv ** 2 * ang
    return out


@dataclass
class Isotropic:
    """Isotropic component with an arbitrary radial profile ``profile(|v|)`` (Galactic frame).

    ``profile`` need not be normalised; the density is profile/N with N the
    3-D integral up to ``v_esc`` (None: the profile must vanish by itself).
    Used for the *shape* of the high-velocity tail — the sharp cut, the soft
    (King-like) cut, and the Gaia-style power-law tail (v_esc - v)^k — which
    is the physics an event at the kinematic edge is sensitive to.
    """
    profile: object
    v_esc: float | None
    fraction: float = 1.0
    name: str = "isotropic"
    v_grid_max: float = 1500.0
    _norm: float | None = field(default=None, repr=False)

    def _radial(self, speed: np.ndarray) -> np.ndarray:
        s = np.asarray(speed, dtype=float)
        f = np.asarray(self.profile(s), dtype=float)
        if self.v_esc is not None:
            f = np.where(s < self.v_esc, f, 0.0)
        return np.clip(f, 0.0, None)

    @property
    def norm(self) -> float:
        if self._norm is None:
            vmax = self.v_esc if self.v_esc is not None else self.v_grid_max
            grid = np.linspace(0.0, vmax, 4001)
            self._norm = float(trapz(4.0 * math.pi * grid ** 2 * self._radial(grid), grid))
        return self._norm

    def density(self, v_xyz: np.ndarray) -> np.ndarray:
        speed = np.sqrt(np.sum(np.asarray(v_xyz, dtype=float) ** 2, axis=-1))
        return self._radial(speed) / self.norm

    def lab_speed_distribution(self, v_lab: np.ndarray, v_grid: np.ndarray, n_mu: int = 96) -> np.ndarray:
        """v² ∫dΩ f(|v n̂ + v_lab|): phi is trivial; the cap in cos(theta) is integrated exactly."""
        v = np.asarray(v_grid, dtype=float)
        vl = float(np.linalg.norm(v_lab))
        out = np.zeros_like(v)
        pos = v > 0
        vv = v[pos]
        if vl < 1e-9:
            out[pos] = 4.0 * math.pi * vv ** 2 * self._radial(vv) / self.norm
            return out
        if self.v_esc is None:
            mu_c = np.ones_like(vv)
        else:
            mu_c = np.clip((self.v_esc ** 2 - vv ** 2 - vl ** 2) / (2.0 * vv * vl), -1.0, 1.0)
        x_gl, w_gl = np.polynomial.legendre.leggauss(n_mu)
        half = 0.5 * (mu_c + 1.0)
        mu = half[:, None] * x_gl[None, :] + (half[:, None] - 1.0)
        w = half[:, None] * w_gl[None, :]
        speed = np.sqrt(vv[:, None] ** 2 + vl ** 2 + 2.0 * vv[:, None] * vl * mu)
        ang = 2.0 * math.pi * np.sum(w * self._radial(speed), axis=1)
        ang = np.where(half > 0.0, ang, 0.0)
        out[pos] = vv ** 2 * ang / self.norm
        return out


def profile_sharp_maxwellian(v0: float):
    return lambda v: np.exp(-np.asarray(v, float) ** 2 / v0 ** 2)


def profile_soft_maxwellian(v0: float, v_esc: float):
    """exp(-v²/v0²) - exp(-v_esc²/v0²): continuous at the escape speed (King-like)."""
    c = math.exp(-(v_esc / v0) ** 2)
    return lambda v: np.exp(-np.asarray(v, float) ** 2 / v0 ** 2) - c


def profile_power_tail(v0: float, v_esc: float, k: float, v_join: float):
    """Maxwellian below v_join, then A (v_esc - v)^k joined continuously (Deason+2019 tail)."""
    fj = math.exp(-(v_join / v0) ** 2)
    a = fj / max(v_esc - v_join, 1e-9) ** k

    def f(v):
        v = np.asarray(v, float)
        tail = a * np.clip(v_esc - v, 0.0, None) ** k
        return np.where(v < v_join, np.exp(-v ** 2 / v0 ** 2), tail)
    return f


def shm_tail(v0: float = 238.0, v_esc: float = 544.0, tail: str = "sharp", k: float = 2.0,
             v_join: float | None = None, name: str | None = None) -> Halo:
    """A one-component isotropic halo with a chosen high-velocity-tail shape."""
    if tail == "sharp":
        prof = profile_sharp_maxwellian(v0)
    elif tail == "soft":
        prof = profile_soft_maxwellian(v0, v_esc)
    elif tail == "power":
        prof = profile_power_tail(v0, v_esc, k, v_join if v_join is not None else 0.8 * v_esc)
    else:
        raise ValueError(tail)
    return Halo([Isotropic(profile=prof, v_esc=v_esc, name=tail)], name=name or f"SHM-{tail}")


@dataclass
class Halo:
    """A mixture of components; fractions are density fractions and should sum to 1."""
    components: list
    name: str = "halo"
    v_grid_max: float = 1500.0
    v_grid_step: float = 1.0
    _eta_cache: dict = field(default_factory=dict, repr=False)

    def eta_at(self, t, v0_kms: float):
        """g(v_min) on date ``t`` for lab velocity v_lab(t; v0), cached per date."""
        from .earth import to_datetime, v_lab_kms
        key = (to_datetime(t).isoformat(), float(v0_kms))
        if key not in self._eta_cache:
            self._eta_cache[key] = self.eta(v_lab_kms(t, v0_kms))
        return self._eta_cache[key]

    @property
    def v_grid(self) -> np.ndarray:
        return np.arange(0.0, self.v_grid_max + 0.5 * self.v_grid_step, self.v_grid_step)

    @property
    def total_fraction(self) -> float:
        return float(sum(c.fraction for c in self.components))

    def density(self, v_xyz: np.ndarray) -> np.ndarray:
        return sum(c.fraction * c.density(v_xyz) for c in self.components)

    def lab_speed_distribution(self, v_lab: np.ndarray) -> np.ndarray:
        """f_lab(v) on ``self.v_grid``; ∫ f_lab dv = total_fraction."""
        vg = self.v_grid
        return sum(c.fraction * c.lab_speed_distribution(np.asarray(v_lab, float), vg)
                   for c in self.components)

    def eta_table(self, v_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(v_grid, g(v_min)) with g in s/km: g(v_min) = ∫_{v_min} f_lab(v)/v dv."""
        vg = self.v_grid
        f = self.lab_speed_distribution(v_lab)
        integrand = np.zeros_like(f)
        integrand[1:] = f[1:] / vg[1:]
        # reverse cumulative trapezoid
        seg = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(vg)
        g = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        return vg, g

    def eta(self, v_lab: np.ndarray):
        """Callable g(v_min) [s/km], linear interpolation, zero beyond the grid."""
        vg, g = self.eta_table(v_lab)

        def _g(v_min):
            return np.interp(np.asarray(v_min, dtype=float), vg, g, left=g[0], right=0.0)
        return _g

    def v_max(self, v_lab: np.ndarray) -> float:
        """Largest lab-frame speed with non-zero density (escape sphere + boost)."""
        vl = float(np.linalg.norm(v_lab))
        vmax = 0.0
        for c in self.components:
            if c.v_esc is not None:
                vmax = max(vmax, c.v_esc + vl)
            else:
                vmax = max(vmax, self.v_grid_max)
        return vmax


def eta_shm_analytic(v_min_kms, v_lab_kms: float, v0: float, v_esc: float):
    """Closed-form g(v_min) [s/km] for the truncated isotropic Maxwellian (Savage+2006)."""
    x = np.asarray(v_min_kms, dtype=float) / v0
    y = v_lab_kms / v0
    z = v_esc / v0
    nesc = erf(z) - 2.0 * z * math.exp(-z * z) / SQRT_PI
    pref = 1.0 / (2.0 * nesc * v0 * y)
    ez = math.exp(-z * z)
    g1 = pref * (erf(x + y) - erf(x - y) - 4.0 * y * ez / SQRT_PI)
    g2 = pref * (erf(z) - erf(x - y) - 2.0 * (y + z - x) * ez / SQRT_PI)
    out = np.where(x < z - y, g1, np.where(x < z + y, g2, 0.0))
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# Standard constructions
# ---------------------------------------------------------------------------
def shm(v0: float = 238.0, v_esc: float = 544.0, name: str = "SHM") -> Halo:
    return Halo([Maxwellian(v0=v0, v_esc=v_esc, fraction=1.0, name="round")], name=name)


def sausage_dispersions(v0: float, beta: float) -> tuple[float, float, float]:
    """(sigma_r, sigma_theta, sigma_phi) of the Sausage for anisotropy beta (Evans+2019)."""
    sr2 = 3.0 * v0 ** 2 / (2.0 * (3.0 - 2.0 * beta))
    st2 = sr2 * (1.0 - beta)
    return math.sqrt(sr2), math.sqrt(st2), math.sqrt(st2)


def shm_plus_plus(v0: float = 233.0, v_esc: float = 528.0, eta_s: float = 0.2, beta: float = 0.9,
                  name: str = "SHM++") -> Halo:
    """Evans, O'Hare & McCabe (2019): round halo + radially anisotropic Sausage.

    At the Sun the Galactocentric radial direction is -x, so sigma_r sits on x,
    sigma_phi on y (rotation) and sigma_theta on z.
    """
    sr, st, sp = sausage_dispersions(v0, beta)
    return Halo([
        Maxwellian(v0=v0, v_esc=v_esc, fraction=1.0 - eta_s, name="round"),
        Gaussian(mean=(0.0, 0.0, 0.0), sigma=(sr, sp, st), v_esc=v_esc, fraction=eta_s, name="sausage"),
    ], name=name)


def with_stream(base: Halo, mean_xyz, sigma: float, fraction: float, v_esc: float | None,
                name: str = "stream") -> Halo:
    """``base`` rescaled to (1 - fraction) plus a Gaussian stream of density ``fraction``."""
    comps = []
    for c in base.components:
        d = dict(c.__dict__)
        d.pop("_norm", None)
        d["fraction"] = c.fraction * (1.0 - fraction)
        comps.append(type(c)(**d))
    comps.append(Gaussian(mean=tuple(float(x) for x in mean_xyz), sigma=(sigma, sigma, sigma),
                          v_esc=v_esc, fraction=fraction, name=name))
    return Halo(comps, name=f"{base.name}+{name}")


def lmc_tail(base: Halo, speed_kms: float, sigma_kms: float, fraction: float,
             direction=(0.0, -1.0, 0.0), name: str = "lmc") -> Halo:
    """``base`` plus an unbound LMC-boosted component: a Gaussian in velocity space
    with bulk speed ``speed_kms`` along ``direction`` in the Galactic frame
    (default head-on against Galactic rotation, so it is fastest in the lab in
    June like the halo wind), isotropic dispersion ``sigma_kms`` and density
    fraction ``fraction``.  No escape-sphere truncation: these particles are not
    bound to the Milky Way."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return with_stream(base, mean_xyz=tuple(speed_kms * d), sigma=sigma_kms, fraction=fraction,
                       v_esc=None, name=name)


def g_dimensionless(g_s_per_km):
    """g̃ = c ∫ f/v d³v, dimensionless."""
    return np.asarray(g_s_per_km, dtype=float) * C_KMS
