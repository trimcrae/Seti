"""Run-time column discovery for Gaia epoch tables whose schema is not frozen.

Gaia DR4 is not public until 2026-12-02, and the only DR4-shaped file that
exists is the June-2026 epoch-astrometry prerelease (12 sources, a *draft*
data model, `release = "Gaia DR4_RC3"`).  Nothing here may hard-code a column
spelling the archive has not yet served.  Instead every reader asks this module
to resolve a *role* ("scan angle", "AL centroid", "G flux", ...) against the
columns actually present, through an ordered list of candidate spellings.  The
spellings come from three real sources:

* Gaia DR3 ``epoch_photometry`` (DataLink long form, one row per transit, and
  the CDN bulk wide form, one row per source with array cells);
* the DR4 epoch-astrometry prerelease VOTable (37 columns, per-CCD arrays);
* Gaia DR3 / FPR ``sso_observation`` (the only per-transit *astrometry* Gaia
  had published for anything before June 2026).

A role that resolves to nothing is reported, not guessed: ``resolve`` returns
``None`` and ``Resolution.missing`` names it, so a probe can say exactly which
role a new release renamed.

Array cells.  The CDN CSV serialises arrays as text.  Spellings seen across
Gaia products include ``[1.0, 2.0]``, ``(1.0,2.0)``, ``{1.0,2.0}`` and
``"1.0 2.0"``, with ``null``/``NaN``/empty for missing elements; booleans as
``true``/``false``/``1``/``0``.  ``parse_array_column`` accepts all of them
and returns one flat float array plus per-row lengths, which is the only way
to parse ~10^5 cells per file at a speed the runner can afford.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Role -> candidate spellings, most specific first.  Matching is
# case-insensitive and ignores '_' so camelCase (the DPAC CU9 naming that
# gaiasupdate converts to) also resolves.
# ---------------------------------------------------------------------------
PHOT_ROLES: dict[str, tuple[str, ...]] = {
    "source_id": ("source_id", "sourceid"),
    "transit_id": ("transit_id", "transitid", "observation_id"),
    "g_time": ("g_transit_time", "g_obs_time", "obs_time_g", "time", "g_time"),
    "g_flux": ("g_transit_flux", "g_flux", "flux_g"),
    "g_flux_error": ("g_transit_flux_error", "g_flux_error", "flux_error_g"),
    "g_mag": ("g_transit_mag", "g_mag"),
    "g_n_obs": ("g_transit_n_obs", "g_n_obs"),
    "bp_time": ("bp_obs_time", "bp_transit_time", "bp_time"),
    "bp_flux": ("bp_flux", "bp_transit_flux", "flux_bp"),
    "bp_flux_error": ("bp_flux_error", "bp_transit_flux_error", "flux_error_bp"),
    "rp_time": ("rp_obs_time", "rp_transit_time", "rp_time"),
    "rp_flux": ("rp_flux", "rp_transit_flux", "flux_rp"),
    "rp_flux_error": ("rp_flux_error", "rp_transit_flux_error", "flux_error_rp"),
    "noisy": ("photometry_flag_noisy_data",),
    "rejected": ("rejected_by_photometry",),
    "g_var_reject": ("variability_flag_g_reject",),
    "bp_var_reject": ("variability_flag_bp_reject",),
    "rp_var_reject": ("variability_flag_rp_reject",),
    "bp_reject": ("photometry_flag_bp_reject",),
    "rp_reject": ("photometry_flag_rp_reject",),
    "bp_unavailable": ("photometry_flag_bp_unavailable",),
    "rp_unavailable": ("photometry_flag_rp_unavailable",),
    "g_other_flags": ("g_other_flags",),
    "bp_other_flags": ("bp_other_flags",),
    "rp_other_flags": ("rp_other_flags",),
}

PHOT_REQUIRED = ("g_time", "g_flux", "g_flux_error", "bp_flux", "bp_flux_error",
                 "rp_flux", "rp_flux_error")

ASTRO_ROLES: dict[str, tuple[str, ...]] = {
    "source_id": ("source_id", "sourceid"),
    "transit_id": ("transit_id", "transitid", "observation_id"),
    "ra0": ("ra0", "ra_ref", "ra"),
    "dec0": ("dec0", "dec_ref", "dec"),
    "obs_time": ("obs_time_tcb", "obstime", "epoch", "obs_time"),
    "bary_corr": ("obs_time_bary_corr", "obstimebarycorr"),
    "scan_angle": ("scan_pos_angle", "scanposangle", "position_angle_scan", "scan_angle",
                   "scan_direction_angle"),
    "parallax_factor_al": ("parallax_factor_al", "parallaxfactoral"),
    "colour_factor_al": ("colour_factor_al", "colourfactoral"),
    "centroid_al": ("centroid_pos_al", "centroidposal", "al_position", "centroid_al"),
    "centroid_al_error": ("centroid_pos_error_al", "centroidposerroral", "al_position_error",
                          "centroid_al_error"),
    "used_al": ("used_by_agis_al", "usedbyagisal", "used_al"),
    "excess_noise": ("agis_source_excess_noise", "source_excess_noise"),
    "g_mag": ("g_mag", "gmag"),
    "multipeak": ("multipeak",),
    "blended": ("blended",),
    "ccd_proc_flags": ("ccd_proc_flags",),
    "transit_proc_flags": ("transit_proc_flags",),
    "dist_to_ci": ("source_dist_to_last_ci",),
    # SSO per-transit astrometry (gaiadr3/gaiafpr.sso_observation) spells the
    # AL/AC split as systematic+random RA/Dec errors instead.
    "ra_error_random": ("ra_error_random",),
    "dec_error_random": ("dec_error_random",),
    "g_flux": ("g_flux", "g_transit_flux"),
    "g_flux_error": ("g_flux_error", "g_transit_flux_error"),
}

ASTRO_REQUIRED = ("transit_id", "scan_angle", "centroid_al", "centroid_al_error",
                  "parallax_factor_al", "obs_time")


def _norm(name: str) -> str:
    return re.sub(r"[_\s]", "", str(name)).lower()


@dataclass
class Resolution:
    """Which real column each role resolved to, and which roles found nothing."""

    mapping: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)

    def get(self, role: str) -> str | None:
        return self.mapping.get(role)

    def has(self, *roles: str) -> bool:
        return all(r in self.mapping for r in roles)

    def as_dict(self) -> dict:
        return {"mapping": dict(self.mapping), "missing": list(self.missing),
                "unused": list(self.unused)}


def resolve(columns, roles: dict[str, tuple[str, ...]],
            required: tuple[str, ...] = ()) -> Resolution:
    """Map roles to the columns present.  Unknown extra columns are kept in
    ``unused`` (never an error: DR4 proper will add columns)."""
    cols = [str(c) for c in columns]
    by_norm: dict[str, str] = {}
    for c in cols:
        by_norm.setdefault(_norm(c), c)
    res = Resolution()
    taken: set[str] = set()
    for role, spellings in roles.items():
        for s in spellings:
            hit = by_norm.get(_norm(s))
            if hit is not None and hit not in taken:
                res.mapping[role] = hit
                taken.add(hit)
                break
    res.missing = [r for r in required if r not in res.mapping]
    res.unused = [c for c in cols if c not in taken]
    return res


# ---------------------------------------------------------------------------
# Array cells
# ---------------------------------------------------------------------------
_NULLS = re.compile(r"(?i)\b(?:null|none|nan|--|masked)\b")
_TRUE = re.compile(r"(?i)\btrue\b")
_FALSE = re.compile(r"(?i)\bfalse\b")


def _clean_cell(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and not np.isfinite(s):
        return ""
    t = str(s).strip()
    if len(t) >= 2 and t[0] in "[({\"'" and t[-1] in "])}\"'":
        t = t[1:-1].strip()
    if len(t) >= 2 and t[0] in "[({" and t[-1] in "])}":
        t = t[1:-1].strip()
    return t


def parse_array_column(cells) -> tuple[np.ndarray, np.ndarray]:
    """Parse a column of text array cells into (flat float64 values, lengths).

    Missing elements become NaN; booleans become 1.0/0.0.  Whitespace-only
    separation is accepted when a cell carries no commas.  An empty or null
    cell is a zero-length array (the prerelease's ``centroid_pos_ac`` is
    zero-length in every row, and that must not shift its neighbours).
    """
    parts: list[str] = []
    lengths = np.zeros(len(cells), dtype=np.int64)
    for i, c in enumerate(cells):
        t = _clean_cell(c)
        if not t:
            continue
        if "," not in t:
            toks = t.split()
        else:
            toks = [x.strip() for x in t.split(",")]
        lengths[i] = len(toks)
        parts.extend(toks)
    if not parts:
        return np.zeros(0, dtype=float), lengths
    joined = "\x1f".join(parts)
    joined = _TRUE.sub("1", joined)
    joined = _FALSE.sub("0", joined)
    joined = _NULLS.sub("nan", joined)
    toks = joined.split("\x1f")
    vals = np.empty(len(toks), dtype=float)
    try:
        vals[:] = np.asarray(toks, dtype=object).astype(float)
    except (TypeError, ValueError):
        for j, tk in enumerate(toks):
            try:
                vals[j] = float(tk) if tk.strip() else np.nan
            except ValueError:
                vals[j] = np.nan
    return vals, lengths


def split_flat(vals: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Inverse of the flattening: one array per row."""
    if len(lengths) == 0:
        return []
    edges = np.concatenate([[0], np.cumsum(lengths)])
    return [vals[edges[i]:edges[i + 1]] for i in range(len(lengths))]


def as_uint16_flags(x) -> np.ndarray:
    """Gaia flag columns are VOTable ``short`` holding UNSIGNED bitmasks
    (the prerelease has transit_proc_flags = -32208, bit 15 set).  Cast before
    any bit test: ``int16 & 0x8000`` overflows."""
    a = np.asarray(x)
    if a.dtype.kind == "f":
        a = np.where(np.isfinite(a), a, 0).astype(np.int64)
    return a.astype(np.int64).astype(np.uint16)
