"""Generic anomaly-catalogue + parent-sample interface.

TIDEMARK is deliberately *channel-agnostic*.  It does not know or care what an
anomaly is; it asks one question of any anomaly population --- **is its rate per
star spatially structured?** --- and that question needs exactly three things:

1. a **parent sample**: every star the channel actually searched, with sky
   position, distance, and the covariates that governed whether an anomaly
   *could* have been detected;
2. an **anomaly subset** of that parent;
3. a statement of which covariates control detectability.

The parent sample is the non-negotiable part.  A bare candidate list supports no
rate measurement at all --- with no denominator there is no rate --- so a
catalogue without a parent is not silently patched with a synthetic denominator
(that would be fabricating data).  It gets verdict ``NO_PARENT_SAMPLE`` and the
run reports the degradation as a first-class field.

Adapters are declarative: ``config/tidemark.yaml`` names, per channel, the parent
file, the candidate file, the id column, the score column and the threshold.  A
new channel is onboarded by adding a block, not by writing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..galactic.orbits import R0_KPC, Z_SUN_KPC
from ..panspermia.kinematics import _A_ICRS_TO_GAL

# --- verdicts ---------------------------------------------------------------
OK = "OK"
NO_PARENT_SAMPLE = "NO_PARENT_SAMPLE"
EMPTY_ANOMALY_SET = "EMPTY_ANOMALY_SET"
NO_DATA_REACHED = "NO_DATA_REACHED"
INSUFFICIENT_ANOMALIES = "INSUFFICIENT_ANOMALIES"
NO_POSITIONS = "NO_POSITIONS"

#: Below this many anomalies no spatial statistic is meaningful; the run says so
#: rather than reporting a p-value computed from a handful of objects.
MIN_ANOMALIES = 30

#: Covariates tried, in priority order, when a channel does not name its own.
#: Ordered most- to least-important: thin strata drop the *last* entries first.
DEFAULT_COVARIATES = (
    "phot_g_mean_mag",     # apparent magnitude -> photometric SNR
    "dist_pc",             # heliocentric distance (strict mode only)
    "bp_rp",               # colour -> spectral type, model validity
    "ebv",                 # extinction along the sightline
    "log_local_density",   # crowding / confusion
    "n_obs",               # epochs, coverage
)


def numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    """A numeric column as a float array, or an all-``default`` array if absent.

    Channels differ wildly in which columns they carry (some have no parallax,
    some no proper motion, some no sky position at all), so every access to an
    optional column goes through here rather than assuming it exists.
    """
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


# --- coordinates ------------------------------------------------------------
def add_galactic_frame(df: pd.DataFrame, *, parallax_floor_mas: float = 0.05,
                       dist_col: str | None = None) -> pd.DataFrame:
    """Attach Galactic ``l``/``b`` and Galactocentric ``R``/``z`` (kpc).

    Distance comes from ``dist_col`` if given, else an existing ``dist_pc``
    column, else ``1000/parallax``.  Angular coordinates are always defined even
    where the distance is not, so a footprint test survives a missing parallax.
    """
    out = df.copy()
    ra = numeric(out, "ra")
    dec = numeric(out, "dec")
    if ra.size == 0 or not np.isfinite(ra).any():
        return out
    r = np.radians(ra)
    d = np.radians(dec)
    u_icrs = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], axis=0)
    u_gal = _A_ICRS_TO_GAL @ u_icrs                     # X toward GC, Y rotation, Z NGP
    out["l_deg"] = np.degrees(np.arctan2(u_gal[1], u_gal[0])) % 360.0
    out["b_deg"] = np.degrees(np.arcsin(np.clip(u_gal[2], -1, 1)))
    out["abs_b_deg"] = np.abs(out["b_deg"])

    if dist_col and dist_col in out.columns:
        dist = numeric(out, dist_col)
    elif "dist_pc" in out.columns:
        dist = numeric(out, "dist_pc")
    else:
        plx = numeric(out, "parallax")
        dist = np.where(np.isfinite(plx) & (plx >= parallax_floor_mas), 1000.0 / plx, np.nan)
    out["dist_pc"] = dist

    out["X_pc"], out["Y_pc"], out["Z_pc"] = (u_gal[0] * dist, u_gal[1] * dist,
                                             u_gal[2] * dist)
    # Galactocentric cylindrical radius and height (kpc); GC at heliocentric X=+R0.
    x_gc = out["X_pc"].to_numpy(float) / 1e3 - R0_KPC
    y_gc = out["Y_pc"].to_numpy(float) / 1e3
    out["X_gal_kpc"], out["Y_gal_kpc"] = x_gc, y_gc
    out["R_gal_kpc"] = np.hypot(x_gc, y_gc)
    out["z_gal_kpc"] = out["Z_pc"].to_numpy(float) / 1e3 + Z_SUN_KPC
    out["abs_z_gal_kpc"] = np.abs(out["z_gal_kpc"])
    return out


def local_density(df: pd.DataFrame, *, n_side: int = 48) -> np.ndarray:
    """log10 sky-source-density of the parent per equal-area cell (a crowding /
    confusion proxy, and the only footprint descriptor the test ever needs)."""
    ra = numeric(df, "ra")
    dec = numeric(df, "dec")
    ok = np.isfinite(ra) & np.isfinite(dec)
    cell = np.full(len(df), -1, dtype=np.int64)
    # Equal-area: uniform in RA and in sin(dec).
    ri = np.clip((ra / 360.0 * (2 * n_side)).astype(np.int64), 0, 2 * n_side - 1)
    si = np.clip(((np.sin(np.radians(dec)) + 1.0) / 2.0 * n_side).astype(np.int64),
                 0, n_side - 1)
    cell[ok] = (ri * n_side + si)[ok]
    counts = np.bincount(cell[ok], minlength=2 * n_side * n_side)
    out = np.full(len(df), np.nan)
    out[ok] = np.log10(np.maximum(counts[cell[ok]], 1))
    return out


# --- the catalogue object ---------------------------------------------------
@dataclass
class AnomalyCatalogue:
    """A parent sample plus the anomaly subset of it, ready for a rate test."""

    name: str
    parent: pd.DataFrame
    anomaly_mask: np.ndarray
    id_col: str = "source_id"
    score: np.ndarray | None = None
    covariates: tuple = DEFAULT_COVARIATES
    verdict: str = OK
    notes: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    #: How the anomaly subset was defined.  A bare percentile of a score
    #: distribution is NOT a candidate population: it contains, by construction,
    #: exactly as many objects as you asked for, whatever the data looks like.
    #: Spatial structure in such a set most often traces the survey's footprint,
    #: cadence and depth.  Only ``vetted_candidate_list`` earns ``vetted=True``,
    #: and only a vetted population can support a DETECTION verdict.
    anomaly_definition: str = "unspecified"
    vetted: bool = False
    #: Free-text warning carried into the result string (e.g. a channel whose
    #: own ledger records it sits at a systematics floor).
    caveat: str = ""
    caveat_tag: str = ""

    # -- construction helpers -------------------------------------------
    def __post_init__(self):
        self.parent = self.parent.reset_index(drop=True)
        self.anomaly_mask = np.asarray(self.anomaly_mask, bool)
        if self.anomaly_mask.size != len(self.parent):
            raise ValueError(f"{self.name}: mask/parent length mismatch")
        self.covariates = tuple(c for c in self.covariates if c in self.parent.columns)

    @property
    def n_parent(self) -> int:
        return len(self.parent)

    @property
    def n_anomaly(self) -> int:
        return int(self.anomaly_mask.sum())

    @property
    def usable(self) -> bool:
        return self.verdict == OK

    def with_frame(self) -> AnomalyCatalogue:
        """Attach Galactic/Galactocentric coordinates and the crowding proxy."""
        p = add_galactic_frame(self.parent)
        if "log_local_density" not in p.columns:
            p["log_local_density"] = local_density(p)
        self.parent = p
        return self.validate()

    def validate(self) -> AnomalyCatalogue:
        """Set the verdict honestly.  Never invents a denominator."""
        if self.n_parent == 0:
            self.verdict = NO_DATA_REACHED
            self.notes.append("parent sample is empty")
            return self
        if self.n_anomaly == 0:
            self.verdict = EMPTY_ANOMALY_SET
            self.notes.append("no anomalies in the parent sample")
            return self
        if self.n_anomaly >= self.n_parent:
            self.verdict = NO_PARENT_SAMPLE
            self.notes.append(
                "every parent row is flagged: this is a candidate list, not a parent "
                "sample, and supports no rate measurement")
            return self
        need = {"ra", "dec"}
        if not need <= set(self.parent.columns):
            self.verdict = NO_POSITIONS
            self.notes.append("parent sample carries no sky position")
            return self
        if self.n_anomaly < MIN_ANOMALIES:
            self.verdict = INSUFFICIENT_ANOMALIES
            self.notes.append(
                f"{self.n_anomaly} anomalies < {MIN_ANOMALIES}: too few for a spatial "
                "rate statistic; reported, not tested")
            return self
        self.verdict = OK
        return self

    def coordinate(self, name: str) -> np.ndarray:
        if name not in self.parent.columns:
            raise KeyError(f"{self.name}: coordinate '{name}' not available")
        return pd.to_numeric(self.parent[name], errors="coerce").to_numpy(float)

    def summary(self) -> dict:
        return {"channel": self.name, "verdict": self.verdict,
                "anomaly_definition": self.anomaly_definition,
                "population_vetted": self.vetted,
                "caveat": self.caveat or None,
                "n_parent": self.n_parent, "n_anomaly": self.n_anomaly,
                "anomaly_rate": (self.n_anomaly / self.n_parent) if self.n_parent else None,
                "covariates": list(self.covariates), "notes": list(self.notes),
                "provenance": self.provenance}


# --- adapters ---------------------------------------------------------------
def from_frames(name: str, parent: pd.DataFrame, anomalies=None, *,
                id_col: str = "source_id", mask=None, score_col: str | None = None,
                score_min: float | None = None, covariates=DEFAULT_COVARIATES,
                provenance: dict | None = None, caveat: str = "",
                caveat_tag: str = "") -> AnomalyCatalogue:
    """Build a catalogue from a parent frame plus either an explicit mask, an
    anomaly frame/id list to join on ``id_col``, or a score column + threshold."""
    parent = parent.reset_index(drop=True)
    notes: list[str] = []
    definition, vetted = "unspecified", False
    if mask is not None:
        m = np.asarray(mask, bool)
        definition, vetted = "explicit_mask", False
    elif score_col is not None and score_col in parent.columns:
        v = pd.to_numeric(parent[score_col], errors="coerce").to_numpy(float)
        if score_min is not None:
            thr = float(score_min)
            definition = "score_threshold"
        else:
            thr = float(np.nanpercentile(v, 99.0))
            definition = "percentile_cut"
        m = np.isfinite(v) & (v >= thr)
        frac = float(m.mean()) if len(m) else 0.0
        notes.append(f"anomaly = {score_col} >= {thr:g} ({100 * frac:.3g}% of the parent)")
        if definition == "percentile_cut" or abs(frac - 0.01) < 2e-3:
            definition = "percentile_cut"
            notes.append(
                "this is a bare percentile of a score distribution, not a vetted "
                "candidate list: it selects a fixed FRACTION of the parent whatever "
                "the data looks like, so any spatial structure in it is at least as "
                "likely to trace survey footprint, cadence and depth as the sky")
    elif anomalies is not None:
        if isinstance(anomalies, pd.DataFrame):
            if id_col not in anomalies.columns or id_col not in parent.columns:
                ids = None
            else:
                ids = set(anomalies[id_col].astype(str))
        else:
            ids = {str(x) for x in np.ravel(anomalies)}
        if ids is None:
            cat = AnomalyCatalogue(name=name, parent=parent,
                                   anomaly_mask=np.zeros(len(parent), bool),
                                   id_col=id_col, covariates=covariates,
                                   provenance=provenance or {})
            cat.verdict = NO_PARENT_SAMPLE
            cat.notes.append(
                f"cannot join anomalies to the parent: id column '{id_col}' absent "
                "from one of them")
            return cat
        m = parent[id_col].astype(str).isin(ids).to_numpy()
        notes.append(f"anomaly = membership in the candidate list by {id_col}")
        definition, vetted = "vetted_candidate_list", True
    else:
        m = np.zeros(len(parent), bool)
        notes.append("no anomaly definition supplied")

    score = None
    if score_col and score_col in parent.columns:
        score = pd.to_numeric(parent[score_col], errors="coerce").to_numpy(float)
    cat = AnomalyCatalogue(name=name, parent=parent, anomaly_mask=m, id_col=id_col,
                           score=score, covariates=covariates,
                           provenance=provenance or {},
                           anomaly_definition=definition, vetted=vetted,
                           caveat=str(caveat or ""), caveat_tag=str(caveat_tag or ""))
    cat.notes.extend(notes)
    return cat.with_frame()


def _read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[tidemark] could not read {path}: {exc!r}")
        return None


def load_channel(root, name: str, spec: dict) -> AnomalyCatalogue:
    """Load one channel from a declarative spec (see ``config/tidemark.yaml``).

    Spec keys: ``parent`` (glob, relative to the repo root), ``candidates``
    (glob, optional if ``score_col`` is on the parent), ``id_col``,
    ``score_col``, ``score_min``, ``covariates``.  Every failure mode returns a
    catalogue carrying an explicit verdict --- nothing is silently skipped.
    """
    root = Path(root)
    prov = {"spec": dict(spec)}

    def _glob_concat(pattern) -> pd.DataFrame | None:
        if not pattern:
            return None
        frames = []
        for pat in ([pattern] if isinstance(pattern, str) else list(pattern)):
            for f in sorted(root.glob(pat)):
                t = _read_table(f)
                if t is not None and len(t):
                    t = t.copy()
                    t["_src"] = str(f.relative_to(root))
                    frames.append(t)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    parent = _glob_concat(spec.get("parent"))
    id_col = spec.get("id_col", "source_id")
    if parent is None:
        cat = AnomalyCatalogue(name=name, parent=pd.DataFrame(),
                               anomaly_mask=np.zeros(0, bool), id_col=id_col,
                               provenance=prov)
        cat.verdict = NO_PARENT_SAMPLE
        cat.notes.append(
            f"no parent-sample file matched {spec.get('parent')!r}: this channel "
            "publishes candidates but not the population it searched, so no rate "
            "can be formed")
        return cat
    if id_col in parent.columns:
        parent = parent.drop_duplicates(subset=[id_col], keep="first")
    cands = _glob_concat(spec.get("candidates"))
    prov["n_parent_rows_read"] = int(len(parent))
    prov["n_candidate_rows_read"] = int(len(cands)) if cands is not None else 0
    return from_frames(
        name, parent, anomalies=cands, id_col=id_col,
        score_col=spec.get("score_col"), score_min=spec.get("score_min"),
        covariates=tuple(spec.get("covariates") or DEFAULT_COVARIATES),
        provenance=prov, caveat=spec.get("caveat", ""),
        caveat_tag=spec.get("caveat_tag", ""))


def load_all(root, specs: dict) -> dict:
    """Load every configured channel; keys are channel names."""
    return {name: load_channel(root, name, spec) for name, spec in (specs or {}).items()}


def union_catalogue(cats, name: str = "union", *,
                    id_col: str = "source_id") -> AnomalyCatalogue:
    """Combine channels into one catalogue over the union of their parents.

    A star searched by three channels has three chances to be flagged, so
    ``n_channels_searched`` is carried as a **covariate** and matched on in the
    null --- otherwise the union test would simply rediscover which patch of sky
    happened to be covered by the most channels.  A star is an anomaly if any
    channel that searched it flagged it.
    """
    usable = [c for c in cats if c.n_parent > 0]
    if not usable:
        out = AnomalyCatalogue(name=name, parent=pd.DataFrame(),
                               anomaly_mask=np.zeros(0, bool), id_col=id_col)
        out.verdict = NO_DATA_REACHED
        out.notes.append("no channel supplied a parent sample")
        return out

    keep = ["ra", "dec", "parallax", "dist_pc", "phot_g_mean_mag", "bp_rp", "ebv",
            "pmra", "pmdec", "radial_velocity", "teff", "mh", "feh", "alpha_fe",
            "n_obs", "log_local_density"]
    frames, flags = [], []
    for c in usable:
        cols = [id_col] + [k for k in keep if k in c.parent.columns]
        if id_col not in c.parent.columns:
            c.parent = c.parent.copy()
            c.parent[id_col] = [f"{c.name}:{i}" for i in range(len(c.parent))]
            cols = [id_col] + [k for k in keep if k in c.parent.columns]
        sub = c.parent[cols].copy()
        sub[id_col] = sub[id_col].astype(str)
        sub[f"searched_{c.name}"] = True
        sub[f"anom_{c.name}"] = c.anomaly_mask
        frames.append(sub)
        flags.append(c.name)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=id_col, how="outer", suffixes=("", f"_{len(frames)}"))
    # Collapse duplicated physical columns produced by the outer merges.
    for k in keep:
        dupes = [c for c in merged.columns if c == k or c.startswith(k + "_")]
        if len(dupes) > 1:
            merged[k] = merged[dupes].bfill(axis=1).iloc[:, 0]
            merged = merged.drop(columns=[c for c in dupes if c != k])
    searched = [f"searched_{n}" for n in flags]
    anom = [f"anom_{n}" for n in flags]
    for c in searched + anom:
        merged[c] = merged[c].fillna(False).astype(bool)
    merged["n_channels_searched"] = merged[searched].sum(axis=1).astype(int)
    merged["n_channels_flagged"] = merged[anom].sum(axis=1).astype(int)
    mask = merged["n_channels_flagged"].to_numpy() > 0

    covs = tuple(list(DEFAULT_COVARIATES) + ["n_channels_searched"])
    cat = AnomalyCatalogue(name=name, parent=merged, anomaly_mask=mask, id_col=id_col,
                           covariates=covs,
                           provenance={"channels": flags,
                                       "per_channel_n_anomaly":
                                           {n: int(merged[f"anom_{n}"].sum()) for n in flags}})
    cat.notes.append("union over channels; n_channels_searched matched in the null")
    return cat.with_frame()


__all__ = ["AnomalyCatalogue", "add_galactic_frame", "local_density", "from_frames",
           "load_channel", "load_all", "union_catalogue", "DEFAULT_COVARIATES",
           "MIN_ANOMALIES", "OK", "NO_PARENT_SAMPLE", "EMPTY_ANOMALY_SET",
           "NO_DATA_REACHED", "INSUFFICIENT_ANOMALIES", "NO_POSITIONS"]
