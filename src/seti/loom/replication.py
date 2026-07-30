"""The replication signature: population tests on anomalous solar-system objects.

A Von Neumann probe is *defined* by self-replication.  That is the whole reason
this channel is not simply "look for one weird asteroid" — a single object with an
unexplained acceleration has a dozen mundane readings (a bad orbit, an
unmodelled perturbation, an undetected satellite, a comet nobody has seen
outgassing).  Replication predicts something a single object cannot provide: a
**population** sharing an origin.

Four things such a population should show, and no natural population should show
together:

1. **Over-clustering in orbital-element space** beyond what the ordinary minor
   planet population produces — tested against a null matched in the covariates
   that drive both detectability and residual size.
2. **Orbital-pole coherence.**  Nature orients the poles of an unrelated
   population essentially isotropically.  A set launched from one source, or
   holding a standardised trajectory, does not.  This is COMPASS's argument
   (``docs/next-question.md``) transposed from binary orbits to heliocentric
   ones, and it is the residue class the repository calls *shared geometry*.
3. **A size distribution inconsistent with fragmentation.**  A collisional family
   follows a steep power law, because breaking rock makes far more small pieces
   than large ones.  Manufactured objects are built to a specification, so their
   size distribution is *narrow*.  Reported rather than gated — see below.
4. **Concentration at dynamically privileged locations** — mean-motion
   resonances, Trojan points, quasi-satellite orbits.  Signature S29 of
   ``docs/necrosignatures.md``, "monuments at stable points": these are where an
   object parks if it intends to stay, and where a natural body is least likely
   to arrive by chance.

What is a gate and what is only reported
----------------------------------------
(1) and (2) are gates: both have well-defined nulls that the data itself
supplies.  (3) is **reported, not gated** — the anomalous set is *selected* by
exceeding a size-dependent envelope, so its size distribution is already shaped
by selection, and a narrowness statistic on it would partly measure that
selection rather than the population.  It is meaningful only within an
already-identified cluster, and even then the collisional reference slope varies
between real families, so it enters as evidence for a human to weigh rather than
as a threshold.  (4) is a gate only where the resonance library is complete
enough to define a null; otherwise reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Reuse rather than reimplement: the stratified-draw machinery is identical to
# the one TOCSIN uses for its population tests, and one implementation means one
# place for this class of bug to be found.
from ..tocsin.population import matched_draws

# Low-order mean-motion resonances with Jupiter, as semimajor axis in au.
# Computed from a_J = 5.2044 au and a = a_J * (p/q)^(2/3).  These are the
# Kirkwood gaps and the Hilda/Trojan niches -- the dynamically distinguished
# locations in the main belt.
JUPITER_A_AU = 5.2044
RESONANCES: tuple[tuple[str, float, float], ...] = tuple(
    (f"{p}:{q}", p, q) for p, q in
    ((4, 1), (3, 1), (5, 2), (7, 3), (2, 1), (5, 3), (3, 2), (4, 3), (1, 1))
)


def resonance_locations() -> dict[str, float]:
    """Semimajor axis (au) of each catalogued mean-motion resonance."""
    return {name: JUPITER_A_AU * (q / p) ** (2.0 / 3.0)
            for name, p, q in RESONANCES}


@dataclass
class Stat:
    """One population statistic with its degradation stated explicitly."""

    name: str
    statistic: float = float("nan")
    p_value: float = float("nan")
    z: float = float("nan")
    n: int = 0
    gate: bool = True
    ok: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"name": self.name, "statistic": self.statistic,
             "p_value": self.p_value, "z": self.z, "n": self.n,
             "is_gate": self.gate, "ok": self.ok, "reason": self.reason}
        d.update(self.detail)
        return d


# ---------------------------------------------------------------------------
# 1. Clustering in orbital-element space
# ---------------------------------------------------------------------------
def element_clustering(elements: np.ndarray, mask, labels, n_null: int = 500,
                       k: int = 1, rng: np.random.Generator | None = None) -> Stat:
    """Are the anomalous objects closer together in element space than matched draws?

    ``elements`` is typically ``(a, e, sin i)`` — the standard proper-element
    space in which collisional families are identified.  A left tail means
    over-clustered.  The matched null is what stops the statistic from firing on
    the obvious: anomalous objects are preferentially small and poorly observed,
    and small poorly-observed objects are not uniformly distributed in the belt.
    """
    st = Stat("element_clustering")
    X = np.asarray(elements, float)
    good = np.all(np.isfinite(X), axis=1)
    m = np.asarray(mask, bool) & good
    st.n = int(m.sum())
    if st.n <= k + 2:
        st.reason = "too_few_anomalies_with_elements"
        return st
    sd = np.nanstd(X[good], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - np.nanmean(X[good], axis=0)) / sd

    def stat(sel: np.ndarray) -> float:
        pts = Xs[sel]
        if pts.shape[0] <= k:
            return float("nan")
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(np.median(np.sort(d, axis=1)[:, k - 1]))

    s_obs = stat(m)
    rng = rng or np.random.default_rng(6060)
    draws = matched_draws(np.asarray(labels), m, int(n_null), rng)
    null = np.array([stat(draws[i] & good) for i in range(int(n_null))])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        st.reason = "null_did_not_populate"
        return st
    st.statistic = s_obs
    st.p_value = (int((null <= s_obs).sum()) + 1) / (null.size + 1)
    st.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    st.detail = {"null_median": float(np.median(null)),
                 "space": "standardised (a, e, sin i)"}
    st.ok = True
    return st


# ---------------------------------------------------------------------------
# 2. Orbital-pole coherence
# ---------------------------------------------------------------------------
def orbital_poles(inc_deg, node_deg) -> np.ndarray:
    """Unit orbital-pole vectors from inclination and longitude of ascending node."""
    i = np.radians(np.asarray(inc_deg, dtype=float))
    om = np.radians(np.asarray(node_deg, dtype=float))
    return np.column_stack([np.sin(i) * np.sin(om),
                            -np.sin(i) * np.cos(om),
                            np.cos(i)])


def pole_coherence(inc_deg, node_deg, mask, labels, n_null: int = 500,
                   rng: np.random.Generator | None = None) -> Stat:
    """Do the anomalous objects' orbital poles concentrate beyond the population's?

    The statistic is the largest eigenvalue of the **orientation tensor**
    ``sum(p p^T)/n``, which is the correct concentration measure for axes: an
    orbital pole is physically a direction but the observable is an axis, and a
    vector mean would partially cancel antipodal poles and understate real
    alignment.  This is the same axial-statistics care COMPASS takes with Gaia
    NSS orbits, where treating projective-sphere data as full vectors is wrong by
    construction.

    The null is matched draws from the same population, so the main belt's own
    strong inclination structure — which is emphatically not isotropic — cancels
    instead of being mistaken for coherence.
    """
    st = Stat("pole_coherence")
    P = orbital_poles(inc_deg, node_deg)
    good = np.all(np.isfinite(P), axis=1)
    m = np.asarray(mask, bool) & good
    st.n = int(m.sum())
    if st.n < 4:
        st.reason = "too_few_anomalies_with_i_and_node"
        return st

    def stat(sel: np.ndarray) -> float:
        p = P[sel]
        if p.shape[0] < 3:
            return float("nan")
        T = (p[:, :, None] * p[:, None, :]).mean(axis=0)
        return float(np.linalg.eigvalsh(T)[-1])

    s_obs = stat(m)
    rng = rng or np.random.default_rng(7171)
    draws = matched_draws(np.asarray(labels), m, int(n_null), rng)
    null = np.array([stat(draws[i] & good) for i in range(int(n_null))])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        st.reason = "null_did_not_populate"
        return st
    st.statistic = s_obs
    # Right tail: a LARGER leading eigenvalue means more concentrated.
    st.p_value = (int((null >= s_obs).sum()) + 1) / (null.size + 1)
    st.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    st.detail = {"null_median": float(np.median(null)),
                 "statistic_is": "largest_eigenvalue_of_orientation_tensor",
                 "isotropic_expectation": 1.0 / 3.0}
    st.ok = True
    return st


# ---------------------------------------------------------------------------
# 2b. Inclination isotropy: the memory of an interstellar arrival
# ---------------------------------------------------------------------------
def inclination_isotropy(inc_deg, mask, labels, n_null: int = 500,
                         rng: np.random.Generator | None = None) -> Stat:
    """Is the anomalous set's inclination distribution more isotropic than the population's?

    Everything that formed in the disc is dynamically cold: the main belt's median
    inclination is a few degrees, and that is a *memory of formation*.  A
    population captured from interstellar space has no such memory — its
    inclinations are drawn from something close to isotropic, which means a median
    ``sin i`` near 0.5 rather than near 0.1, and a substantial retrograde
    fraction.  This is the sharpest available statement about *origin* that bound
    orbital elements can still carry, because capture erases the arrival asymptote
    but not the plane.

    The statistic is the mean of ``sin i`` over the anomalous set, tested against
    matched draws.  One caveat is stated rather than buried: relative velocity
    rises with inclination, so a high-inclination object is detected and linked
    differently from a low-inclination one, and the matched null controls for
    ``H``, arc length and observation count but *not* for inclination-dependent
    linking efficiency.  A positive result here is therefore evidence about the
    sample's selection function until that efficiency is measured, and it is
    reported with that attached.
    """
    st = Stat("inclination_isotropy")
    inc = np.asarray(inc_deg, dtype=float)
    good = np.isfinite(inc)
    m = np.asarray(mask, bool) & good
    st.n = int(m.sum())
    if st.n < 5:
        st.reason = "too_few_anomalies_with_inclination"
        return st
    s = np.sin(np.radians(inc))

    def stat(sel: np.ndarray) -> float:
        v = s[sel & good]
        return float(np.mean(v)) if v.size else float("nan")

    s_obs = stat(m)
    rng = rng or np.random.default_rng(5959)
    draws = matched_draws(np.asarray(labels), m, int(n_null), rng)
    null = np.array([stat(draws[i]) for i in range(int(n_null))])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        st.reason = "null_did_not_populate"
        return st
    st.statistic = s_obs
    # Right tail: a LARGER mean sin i means dynamically hotter, i.e. closer to
    # isotropic, i.e. less like something that formed in this disc.
    st.p_value = (int((null >= s_obs).sum()) + 1) / (null.size + 1)
    st.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    st.detail = {
        "null_median": float(np.median(null)),
        "isotropic_expectation": math.pi / 4.0,
        "retrograde_fraction": float(np.mean(inc[m] > 90.0)),
        "caveat": ("matched on H, arc length and observation count but NOT on "
                   "inclination-dependent linking efficiency; a positive result "
                   "is evidence about selection until that efficiency is measured"),
    }
    st.ok = True
    return st


# ---------------------------------------------------------------------------
# 3. Size distribution (REPORTED, not gated)
# ---------------------------------------------------------------------------
def size_distribution(h_values, collisional_slope: float = 0.5) -> Stat:
    """Cumulative ``H`` slope and spread of a group.  Reported, never a gate.

    Breaking rock makes far more small fragments than large ones, so a collisional
    family's cumulative count rises steeply with ``H``; ``log10 N(<H) ≈ slope * H``
    with ``slope`` of order 0.5 for a Dohnanyi-like distribution.  Manufactured
    objects are built to a specification, so their sizes are *narrow* and the
    fitted slope is shallow.

    NOT a gate, for a reason that matters: the anomalous set is selected by
    exceeding a size-dependent envelope, so its size distribution is already
    shaped by that selection and a narrowness statistic would partly measure the
    selection function.  It is informative only inside an already-identified
    cluster, and even there real families scatter about the reference slope.  So
    this returns evidence for a human to weigh, with ``gate=False``.
    """
    st = Stat("size_distribution", gate=False)
    h = np.asarray(h_values, dtype=float)
    h = h[np.isfinite(h)]
    st.n = int(h.size)
    if h.size < 8:
        st.reason = "fewer_than_8_objects_with_H"
        return st
    hs = np.sort(h)
    n_cum = np.arange(1, hs.size + 1)
    # Fit log10 N(<H) = c + slope * H over the interior, avoiding both tails
    # where the cumulative count is least reliable.
    lo, hi = int(0.1 * hs.size), int(0.9 * hs.size)
    if hi - lo < 5:
        lo, hi = 0, hs.size
    x, y = hs[lo:hi], np.log10(n_cum[lo:hi])
    # Centre before fitting: an H range of a few magnitudes with an offset of ~20
    # makes the design matrix badly conditioned and numpy warns, for a fit that is
    # perfectly well determined once the pivot is moved to the data.
    x0 = float(np.mean(x))
    if float(np.ptp(x)) <= 0:
        # A set with a single H value has no size distribution.  This is not an
        # edge case to smooth over -- it is the honest answer, and fitting it would
        # crash the least-squares solve on a singular design matrix.
        st.reason = "all_objects_share_one_H__no_size_distribution_to_fit"
        st.detail = {"H_median": float(np.median(h)), "H_iqr": 0.0}
        return st
    slope, intercept0 = np.polyfit(x - x0, y, 1)
    intercept = float(intercept0 - slope * x0)
    st.statistic = float(slope)
    st.detail = {
        "cumulative_H_slope": float(slope),
        "collisional_reference_slope": float(collisional_slope),
        "slope_ratio_to_collisional": float(slope / collisional_slope)
        if collisional_slope else None,
        "H_iqr": float(np.subtract(*np.percentile(h, [75, 25]))),
        "H_median": float(np.median(h)),
        "intercept": float(intercept),
        "interpretation": ("shallow slope / narrow spread is what a built-to-spec "
                           "population looks like; steep is fragmentation"),
    }
    st.ok = True
    return st


# ---------------------------------------------------------------------------
# 3b. Photometric homogeneity, and what it can and cannot decide
# ---------------------------------------------------------------------------
def photometric_homogeneity(rows: list[dict], mask, labels, n_null: int = 500,
                            bands=("g", "r", "i", "z"),
                            rng: np.random.Generator | None = None) -> Stat:
    """Are the anomalous objects unusually alike in colour and phase function?

    Rubin's ``ssObject`` fits ``H`` and ``G12`` **per band** for all six filters,
    with covariances, so every object comes with a six-band phase curve.  A
    population built to a specification should be strikingly homogeneous in that
    space — same surface, same scattering behaviour — where a random draw from the
    minor-planet population spans the whole taxonomic range.

    What this test can decide, stated plainly: it distinguishes *a set with a
    common origin* from *a random subset of the belt*.  It does **not** distinguish
    manufactured from collisional, because a collisional family is also
    homogeneous — the fragments came off one parent body.  That separation is what
    :func:`size_distribution` is for, and it is the reason that function exists at
    all despite not being usable as a gate.

    Prior art to be honest about: SNAPS already performs unsupervised population
    outlier detection on minor-planet photometry at survey scale, over 15 features
    including per-band ``H``, ``G``, colour and rotation period.  The novelty here
    is not "look for photometric outliers among asteroids" — that is done — it is
    applying a *homogeneity* statistic to a set selected by its **dynamics**, which
    puts no photometric feature in the selection and so leaves this an independent
    test.
    """
    st = Stat("photometric_homogeneity")
    cols = []
    for b in bands:
        v = np.array([_f(r.get(f"h_{b}")) for r in rows])
        if np.isfinite(v).sum() > 0.5 * len(rows):
            cols.append(v)
    g12 = np.array([_f(r.get("g12_r")) for r in rows])
    if np.isfinite(g12).sum() > 0.5 * len(rows):
        cols.append(g12)
    if len(cols) < 2:
        st.reason = "fewer_than_two_usable_photometric_columns"
        return st
    # Colours rather than magnitudes: an absolute magnitude carries size, and this
    # test must not be a size test wearing a colour test's name.
    X = np.column_stack([cols[k] - cols[0] for k in range(1, len(cols))])
    good = np.all(np.isfinite(X), axis=1)
    m = np.asarray(mask, bool) & good
    st.n = int(m.sum())
    if st.n < 5:
        st.reason = "too_few_anomalies_with_photometry"
        return st
    sd = np.nanstd(X[good], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = X / sd

    def stat(sel: np.ndarray) -> float:
        pts = Xs[sel & good]
        if pts.shape[0] < 3:
            return float("nan")
        # Mean distance from the set's own centroid: a scale-free dispersion.
        return float(np.mean(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))

    s_obs = stat(m)
    rng = rng or np.random.default_rng(4747)
    draws = matched_draws(np.asarray(labels), m, int(n_null), rng)
    null = np.array([stat(draws[i]) for i in range(int(n_null))])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        st.reason = "null_did_not_populate"
        return st
    st.statistic = s_obs
    # Left tail: SMALLER dispersion means more homogeneous.
    st.p_value = (int((null <= s_obs).sum()) + 1) / (null.size + 1)
    st.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    st.detail = {"null_median": float(np.median(null)),
                 "n_colour_dimensions": int(Xs.shape[1]),
                 "cannot_decide": ("manufactured vs collisional -- a collisional "
                                   "family is homogeneous too; see size_distribution")}
    st.ok = True
    return st


# ---------------------------------------------------------------------------
# 3c. Mis-linkage: the contaminant that looks exactly like a family
# ---------------------------------------------------------------------------
def linkage_duplicates(rows: list[dict], mask, a_tol: float = 1e-3,
                       e_tol: float = 5e-3, i_tol: float = 0.05,
                       require_disjoint_epochs: bool = True) -> dict:
    """Find anomalous objects that are probably ONE object under several designations.

    This is the channel's most dangerous contaminant and it is not hypothetical.
    A strong non-gravitational acceleration (``|A_i|`` above ~1e-8 au/day^2) breaks
    the MPC linking algorithms outright: tracklets fail to link across apparitions
    and can be attached to *multiple* designations.  So a single genuinely
    accelerating object can enter the catalogue several times, and the result is a
    set of objects with near-identical orbital elements, anomalous residuals, and
    non-overlapping observation epochs — which is precisely the signature this
    channel calls replication.

    The two are separable, and the separator is the epoch coverage: real distinct
    objects in a family are observed *contemporaneously*, whereas the same object
    under two designations is observed in disjoint intervals.  Groups flagged here
    are collapsed to one representative before any clustering statistic runs, and
    the collapse is reported, because a "family" that evaporates under it is a
    finding about MPC linking rather than about the solar system.
    """
    m = np.asarray(mask, bool)
    idx = np.flatnonzero(m)
    out: dict = {"n_anomalous": int(idx.size), "groups": []}
    if idx.size < 2:
        out["n_groups"] = 0
        out["n_collapsed"] = 0
        return out
    a = np.array([_f(rows[i].get("a")) for i in idx])
    e = np.array([_f(rows[i].get("e")) for i in idx])
    inc = np.array([_f(rows[i].get("i")) for i in idx])
    t0 = np.array([_f(rows[i].get("mjd_min")) for i in idx])
    t1 = np.array([_f(rows[i].get("mjd_max")) for i in idx])

    assigned = -np.ones(idx.size, dtype=int)
    groups: list[list[int]] = []
    for p in range(idx.size):
        if assigned[p] >= 0:
            continue
        members = [p]
        for q in range(p + 1, idx.size):
            if assigned[q] >= 0:
                continue
            close = (abs(a[p] - a[q]) <= a_tol and abs(e[p] - e[q]) <= e_tol
                     and abs(inc[p] - inc[q]) <= i_tol)
            if not close:
                continue
            if require_disjoint_epochs:
                # Overlapping observation windows mean two objects really were on
                # the sky at once, so they are not the same object mis-linked.
                overlap = (math.isfinite(t0[p]) and math.isfinite(t1[p])
                           and math.isfinite(t0[q]) and math.isfinite(t1[q])
                           and not (t1[p] < t0[q] or t1[q] < t0[p]))
                if overlap:
                    continue
            members.append(q)
        if len(members) > 1:
            for q in members:
                assigned[q] = len(groups)
            groups.append(members)
    out["n_groups"] = len(groups)
    out["n_collapsed"] = sum(len(g) - 1 for g in groups)
    out["groups"] = [[int(idx[q]) for q in g] for g in groups]
    out["note"] = ("orbital elements matched within (a, e, i) tolerances with "
                   "disjoint observation epochs: consistent with one accelerating "
                   "object entered under several designations, not with a family")
    return out


def collapse_duplicates(mask, dup: dict) -> np.ndarray:
    """Keep one representative per mis-linkage group; drop the rest from the mask."""
    m = np.asarray(mask, bool).copy()
    for group in dup.get("groups", []):
        for j in group[1:]:
            m[int(j)] = False
    return m


# ---------------------------------------------------------------------------
# 4. Concentration at dynamically privileged locations
# ---------------------------------------------------------------------------
def resonance_concentration(a_au, mask, labels, n_null: int = 500,
                            width_au: float = 0.02,
                            rng: np.random.Generator | None = None) -> Stat:
    """Are anomalous objects nearer mean-motion resonances than matched draws?

    Signature S29, "monuments at stable points": a resonance or a Trojan point is
    where an object parks if it means to stay, and where a natural body is least
    likely to arrive by chance — the Kirkwood gaps exist precisely because
    resonances *remove* ordinary asteroids.  That last fact cuts both ways and is
    why the null must be matched: the belt's own density is depleted at exactly
    these locations, so a naive test would be biased against detection, making
    this conservative rather than credulous.
    """
    st = Stat("resonance_concentration")
    a = np.asarray(a_au, dtype=float)
    good = np.isfinite(a)
    m = np.asarray(mask, bool) & good
    st.n = int(m.sum())
    if st.n < 4:
        st.reason = "too_few_anomalies_with_semimajor_axis"
        return st
    locs = np.array(sorted(resonance_locations().values()))
    dist = np.min(np.abs(a[:, None] - locs[None, :]), axis=1)

    def stat(sel: np.ndarray) -> float:
        d = dist[sel & good]
        return float(np.median(d)) if d.size else float("nan")

    s_obs = stat(m)
    rng = rng or np.random.default_rng(8282)
    draws = matched_draws(np.asarray(labels), m, int(n_null), rng)
    null = np.array([stat(draws[i]) for i in range(int(n_null))])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        st.reason = "null_did_not_populate"
        return st
    st.statistic = s_obs
    # Left tail: SMALLER median distance to a resonance means more concentrated.
    st.p_value = (int((null <= s_obs).sum()) + 1) / (null.size + 1)
    st.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    st.detail = {"null_median_au": float(np.median(null)),
                 "resonance_width_au": float(width_au),
                 "n_within_width": int((dist[m] <= width_au).sum()),
                 "resonances": {k: round(v, 4)
                                for k, v in resonance_locations().items()}}
    st.ok = True
    return st


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def replication_tests(rows: list[dict], mask, labels, n_null: int = 500,
                      collisional_slope: float = 0.5,
                      seed: int = 20260730) -> dict:
    """Run every replication statistic on the anomalous subset.

    ``rows`` is one dict per screened solar-system object (the parent), ``mask``
    marks the ones whose non-gravitational behaviour exceeded the empirical
    envelope, and ``labels`` are the matching strata.
    """
    rng = np.random.default_rng(seed)
    m = np.asarray(mask, bool)
    out: dict = {"n_parent": len(rows), "n_anomaly_raw": int(m.sum()),
                 "n_null": int(n_null), "tests": []}

    # Mis-linkage FIRST, before any statistic sees the set.  A strongly
    # accelerating object can enter the catalogue several times under different
    # designations, producing near-identical elements with disjoint epochs -- which
    # is exactly what this channel would otherwise call a family.  Collapsing is
    # not optional and it is reported, because a cluster that evaporates under it
    # is a finding about MPC linking, not about the solar system.
    dup = linkage_duplicates(rows, m)
    out["linkage_duplicates"] = dup
    if dup.get("n_collapsed", 0):
        m = collapse_duplicates(m, dup)
    out["n_anomaly"] = int(m.sum())
    if len(rows) < 200 or m.sum() < 5:
        out["verdict"] = "INSUFFICIENT_POPULATION"
        out["note"] = ("replication tests need a parent of >=200 screened objects "
                       "and >=5 anomalies; below that the matched null cannot be "
                       "populated and any p-value would be noise")
        return out

    def col(name):
        return np.array([_f(r.get(name)) for r in rows])

    a, e, inc = col("a"), col("e"), col("i")
    node, h = col("node"), col("h")
    elements = np.column_stack([a, e, np.sin(np.radians(inc))])

    out["tests"].append(element_clustering(elements, m, labels, n_null=n_null,
                                           rng=rng).as_dict())
    out["tests"].append(pole_coherence(inc, node, m, labels, n_null=n_null,
                                       rng=rng).as_dict())
    out["tests"].append(inclination_isotropy(inc, m, labels, n_null=n_null,
                                             rng=rng).as_dict())
    out["tests"].append(resonance_concentration(a, m, labels, n_null=n_null,
                                                rng=rng).as_dict())
    out["tests"].append(photometric_homogeneity(rows, m, labels, n_null=n_null,
                                                rng=rng).as_dict())
    out["tests"].append(size_distribution(h[m],
                                          collisional_slope=collisional_slope).as_dict())

    gates = [t for t in out["tests"] if t.get("ok") and t.get("is_gate")]
    out["n_gates_usable"] = len(gates)
    if not gates:
        out["verdict"] = "NO_TEST_COULD_RUN"
        return out
    pmin = min(t["p_value"] for t in gates)
    out["p_min"] = pmin
    out["bonferroni_threshold"] = 0.05 / len(gates)
    out["p_resolution_floor"] = 1.0 / (int(n_null) + 1)
    if out["p_resolution_floor"] > 0.2 * out["bonferroni_threshold"]:
        out["verdict"] = "INSUFFICIENT_RESOLUTION"
        out["note"] = (f"n_null={n_null} resolves p only to "
                       f"{out['p_resolution_floor']:.4f}, not comfortably below "
                       f"the Bonferroni threshold {out['bonferroni_threshold']:.4f} "
                       f"for {len(gates)} gates; raise n_null before believing any "
                       f"detection")
        return out
    out["verdict"] = ("REPLICATION_STRUCTURE_DETECTED"
                      if pmin <= out["bonferroni_threshold"] else "NO_STRUCTURE")
    return out


def _f(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")
