"""The channel registry: which channels carry per-star scores, over what parent,
and which instrument their anomaly is measured with.

Every loader returns the same frame::

    key        str    "gaia:<source_id>" when a Gaia DR3 id is known, else "<ns>:<id>"
    source_id  Int64  Gaia DR3 source_id or <NA>
    ra, dec    float  degrees (NaN when the channel's file carries none)
    score      float  higher = more anomalous IN THE CHANNEL'S OWN SENSE; NaN = in the
                      parent but unscored (then only the flag can put it in the tail)
    flag       bool   the channel's own flagged set (its candidates / leg-1 tail)

A row is a member of the channel's PARENT sample.  A channel whose committed
files hold only its survivors has no parent here: it is listed with
``status="tail_only"`` and cannot enter the statistic until it emits per-star
scores (see ``TAIL_ONLY``).

Independence
------------
Two channels are independent only if no instrument measures the anomaly in
both.  ``primary`` names the instrument(s) whose measurement *is* the anomaly;
``support`` names instruments used only to normalise it (a photosphere
prediction, a luminosity, a sample cut).  The headline rule is strict:

    independent(A, B)  <=>  (A.primary ∪ A.support) ∩ (B.primary ∪ B.support) = ∅

with Gaia's *sample-definition* role (every channel's parent is a Gaia
selection) excluded, because the matched null conditions on exactly the Gaia
quantities (G, BP-RP, RUWE, scan coverage) that could carry a Gaia artefact
into both.  ``gaia_astrometry`` / ``gaia_phot`` / ``gaia_rvs`` appear as tags
only where the anomaly is *computed from* them (a parallax-based luminosity, an
acceleration, a spectroscopic twin).  A relaxed rule (primaries disjoint, and
neither primary used as the other's support) is reported beside it, never in
place of it.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(os.environ.get("SETI_RESULTS", "results"))
SCORES_DIR = RESULTS / "confluence" / "scores"

STD_COLUMNS = ["key", "source_id", "ra", "dec", "score", "flag"]


@dataclass
class ChannelSpec:
    name: str
    family: str
    primary: frozenset
    support: frozenset
    observable: str
    parent: str
    loader: Callable[[], pd.DataFrame] | None = None
    status: str = "scored"          # scored | flag_only | tail_only | pending_harvest
    tail_mode: str = "score"        # score | flag
    notes: str = ""
    source_files: list = field(default_factory=list)

    @property
    def tags(self) -> frozenset:
        return self.primary | self.support


def independent(a: ChannelSpec, b: ChannelSpec, rule: str = "strict") -> bool:
    if a.family == b.family:
        return False
    if rule == "strict":
        return not (a.tags & b.tags)
    return not (a.primary & b.primary) and not (a.primary & b.support) \
        and not (b.primary & a.support)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sid(x) -> pd.Series:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    s = s.where(s > 0)
    return s.round().astype("Int64")


def standardise(df: pd.DataFrame, *, sid=None, key_ns: str | None = None, key_col=None,
                ra="ra", dec="dec", score=None, flag=None) -> pd.DataFrame:
    """Build the standard frame from a channel table."""
    out = pd.DataFrame(index=df.index)
    out["source_id"] = _sid(df[sid]).to_numpy() if sid is not None and sid in df else pd.NA
    out["source_id"] = out["source_id"].astype("Int64")
    out["ra"] = pd.to_numeric(df[ra], errors="coerce") if ra in df else np.nan
    out["dec"] = pd.to_numeric(df[dec], errors="coerce") if dec in df else np.nan
    if callable(score):
        out["score"] = np.asarray(score(df), dtype=float)
    elif score is not None:
        out["score"] = pd.to_numeric(df[score], errors="coerce").astype(float)
    else:
        out["score"] = np.nan
    if callable(flag):
        out["flag"] = np.asarray(flag(df), dtype=bool)
    elif flag is not None and flag in df:
        f = df[flag]
        out["flag"] = f.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy() \
            if f.dtype == object else f.fillna(False).astype(bool).to_numpy()
    else:
        out["flag"] = False
    if key_col is not None:
        alt = key_ns + ":" + df[key_col].astype(str)
    else:
        alt = pd.Series(
            [f"pos:{a:.5f}_{b:+.5f}" if np.isfinite(a) and np.isfinite(b) else f"row:{i}"
             for i, (a, b) in enumerate(zip(out["ra"], out["dec"], strict=False))],
            index=df.index)
    has = out["source_id"].notna()
    out["key"] = np.where(has, "gaia:" + out["source_id"].astype(str), alt)
    out = out[STD_COLUMNS].reset_index(drop=True)
    # one row per star: keep the most anomalous
    out = out.sort_values("score", ascending=False, na_position="last")
    out = out.groupby("key", sort=False).agg(
        source_id=("source_id", "first"), ra=("ra", "first"), dec=("dec", "first"),
        score=("score", "first"), flag=("flag", "max")).reset_index()
    return out[STD_COLUMNS]


def _read(path) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def _concat(pattern: str, **kw) -> pd.DataFrame:
    fs = sorted(glob.glob(str(RESULTS / pattern)))
    if not fs:
        return pd.DataFrame()
    return pd.concat([_read(f) for f in fs], ignore_index=True, **kw)


# ---------------------------------------------------------------------------
# loaders over committed files
# ---------------------------------------------------------------------------


def _dimming(kind: str) -> pd.DataFrame:
    d = _concat("dimming/*/dimming_scored.csv")
    if d.empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    # Mixed id namespaces: most rows are ZTF object ids (15-16 digits); a few
    # fields keyed on Gaia DR3 ids (18-19 digits).  Only the latter are Gaia.
    sid_str = d["source_id"].astype(str)
    d["_gaia"] = np.where(sid_str.str.len() >= 17, d["source_id"], np.nan)
    if kind == "dip":
        return standardise(d, sid="_gaia", key_ns="ztf", key_col="source_id",
                           score="score", flag="is_candidate")
    slope = pd.to_numeric(d["secular_slope_mag_yr"], errors="coerce")
    sig = pd.to_numeric(d["secular_sigma"], errors="coerce")
    return standardise(d, sid="_gaia", key_ns="ztf", key_col="source_id",
                       score=lambda _: (np.sign(slope) * sig.abs()).to_numpy(),
                       flag="is_secular_fader")


def _baffle_bright(direction: str) -> pd.DataFrame:
    p = RESULTS / "baffle_bright" / "bright_residuals.csv"
    if not p.exists():
        return pd.DataFrame(columns=STD_COLUMNS)
    b = pd.read_csv(p, low_memory=False)
    s09 = pd.to_numeric(b["sig_s09"], errors="coerce")
    f12 = pd.to_numeric(b["sig_f12"], errors="coerce")
    s18 = pd.to_numeric(b["sig_s18"], errors="coerce")
    f25 = pd.to_numeric(b["sig_f25"], errors="coerce")
    if direction == "deficit":
        # resid = (Ks - m) - locus: negative = fainter than the photosphere.
        sc = -s09.fillna(f12)
        return standardise(b, sid="source_id", score=lambda _: sc.to_numpy(),
                           flag=lambda x: (x["veto"].astype(str) == "").to_numpy()
                           | x["is_candidate"].astype(str).str.lower().eq("true").to_numpy())
    sc = pd.concat([s18, f25], axis=1).max(axis=1)
    sc = sc.where(pd.concat([s18, f25], axis=1).notna().any(axis=1))
    return standardise(b, sid="source_id", score=lambda _: sc.to_numpy(), flag=None)


def _cenotaph_committed() -> pd.DataFrame:
    """Parent = the committed GSP-Spec dwarf shells; tail = the vet-core candidates."""
    sh = _concat("cenotaph/shell_*.parquet")
    if sh.empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    c = RESULTS / "cenotaph" / "candidates.csv"
    cand = set(pd.read_csv(c)["source_id"].astype("int64")) if c.exists() else set()
    sh = sh[["source_id", "ra", "dec"]].copy()
    sh["_flag"] = sh["source_id"].astype("int64").isin(cand)
    return standardise(sh, sid="source_id", score=None, flag="_flag")


def _slag() -> pd.DataFrame:
    m = RESULTS / "slag" / "misfit_list.csv"
    pw = RESULTS / "slag" / "data" / "pewdd_vizier.csv"
    if not (m.exists() and pw.exists()):
        return pd.DataFrame(columns=STD_COLUMNS)
    ml = pd.read_csv(m)
    p = pd.read_csv(pw, low_memory=False, usecols=lambda c: c in (
        "Star", "Gaia", "RAJ2000", "DEJ2000"))
    if "Star" not in p.columns:
        return pd.DataFrame(columns=STD_COLUMNS)
    # PEWDD writes the id as "Gaia DR3 <digits>"
    p["Gaia"] = pd.to_numeric(p.get("Gaia", pd.Series(dtype=str)).astype(str)
                              .str.extract(r"(\d{5,})", expand=False), errors="coerce")
    p = p.drop_duplicates("Star").rename(columns={"Star": "name"})
    x = ml.merge(p, on="name", how="left")
    x["_s"] = -np.log10(pd.to_numeric(x["p_misfit"], errors="coerce").clip(lower=1e-6))
    x["_f"] = x["misfit_class"].astype(str).eq("UNEXPLAINED")
    return standardise(x, sid="Gaia", ra="RAJ2000", dec="DEJ2000", key_ns="wd",
                       key_col="name", score="_s", flag="_f")


def _growth() -> pd.DataFrame:
    p = RESULTS / "growth" / "direct" / "measurements.csv"
    if not p.exists():
        return pd.DataFrame(columns=STD_COLUMNS)
    g = pd.read_csv(p, low_memory=False)
    g = g[pd.to_numeric(g["pdc_z"], errors="coerce").notna()]
    return standardise(g, sid=None, key_ns="kic", key_col="kepid", score="pdc_z", flag=None)


def _spark() -> pd.DataFrame:
    p = RESULTS / "spark" / "euclid" / "features_all.csv.gz"
    if not p.exists():
        return pd.DataFrame(columns=STD_COLUMNS)
    s = pd.read_csv(p, low_memory=False)
    # the channel's own anomaly is a line that is NOT a known stellar line
    stellar = s["stellar_line"].astype(str).str.lower().eq("true")
    s["_s"] = np.where(stellar, np.nan, pd.to_numeric(s["snr"], errors="coerce"))
    s["_s"] = s["_s"].fillna(-np.inf)   # in the parent, least anomalous
    return standardise(s, sid="gaia_source_id", key_ns="euclid", key_col="object_id",
                       score="_s", flag="survivor")


def _arc() -> pd.DataFrame:
    """ARC xi over its assessable stars; positions from the runner's id resolution."""
    p = RESULTS / "arc" / "xi_table.csv"
    pos = RESULTS / "confluence" / "arc_positions.csv"
    if not p.exists():
        return pd.DataFrame(columns=STD_COLUMNS)
    x = pd.read_csv(p, low_memory=False)
    x = x[x["assessable"].astype(str).str.lower().eq("true")].copy()
    if pos.exists():
        ps = pd.read_csv(pos)
        x = x.merge(ps[["star_key", "ra", "dec"]], on="star_key", how="left")
    x["_f"] = x["tier"].astype(str).isin(["candidate", "interest"])
    return standardise(x, sid=None, key_ns="arc", key_col="star_key",
                       score="xi_conservative_max", flag="_f")


def _accel_committed() -> pd.DataFrame:
    p = RESULTS / "accel" / "accel_candidates.csv"
    if not p.exists():
        return pd.DataFrame(columns=STD_COLUMNS)
    return standardise(pd.read_csv(p), sid="source_id", score="accel_significance",
                       flag="dark_companion")


def _harvested(name: str) -> Callable[[], pd.DataFrame]:
    def load() -> pd.DataFrame:
        p = SCORES_DIR / f"{name}.parquet"
        if not p.exists():
            return pd.DataFrame(columns=STD_COLUMNS)
        d = pd.read_parquet(p)
        for c in STD_COLUMNS:
            if c not in d:
                d[c] = np.nan if c in ("ra", "dec", "score") else (False if c == "flag" else pd.NA)
        d["source_id"] = _sid(d["source_id"]).to_numpy()
        d["source_id"] = d["source_id"].astype("Int64")
        d["flag"] = d["flag"].fillna(False).astype(bool)
        if "key" not in d or d["key"].isna().all():
            d["key"] = "gaia:" + d["source_id"].astype(str)
        return d[STD_COLUMNS]
    return load


F = frozenset
REGISTRY: list[ChannelSpec] = [
    ChannelSpec("dimming_dip", "dimming", F({"ztf"}), F(), "ZTF g/r dip score",
                "stars in 106 ZTF fields (committed dimming_scored.csv)",
                lambda: _dimming("dip"), source_files=["dimming/*/dimming_scored.csv"]),
    ChannelSpec("dimming_secular", "dimming", F({"ztf"}), F(), "ZTF signed secular fade sigma",
                "as dimming_dip", lambda: _dimming("secular"),
                source_files=["dimming/*/dimming_scored.csv"]),
    ChannelSpec("baffle_bright_deficit", "baffle_bright", F({"akari", "iras"}), F({"2mass"}),
                "AKARI 9um / IRAS 12um deficit below the Ks-anchored locus (sigma)",
                "29,080 Gaia G<7.5 stars", lambda: _baffle_bright("deficit"),
                source_files=["baffle_bright/bright_residuals.csv"]),
    ChannelSpec("baffle_bright_excess", "baffle_bright", F({"akari", "iras"}), F({"2mass"}),
                "AKARI 18um / IRAS 25um excess above the Ks-anchored locus (sigma)",
                "29,080 Gaia G<7.5 stars", lambda: _baffle_bright("excess"),
                notes="excess direction: not BAFFLE's own anomaly; kept as a positive-control "
                      "observable (Be stars, YSOs, AGB dust light it up)",
                source_files=["baffle_bright/bright_residuals.csv"]),
    ChannelSpec("cenotaph", "cenotaph",
                F({"gaia_phot", "gaia_astrometry", "gaia_rvs", "2mass"}), F({"wise", "galex"}),
                "grey (achromatic) luminosity deficit vs spectroscopic twins (sigma)",
                "Gaia DR3 GSP-Spec dwarfs, poe>20, RUWE<1.4 (674k fitted)",
                _harvested("cenotaph"), status="pending_harvest",
                source_files=["artifact cenotaph-analysis/greyfit.parquet (run 30212775555)"]),
    ChannelSpec("cenotaph_committed", "cenotaph",
                F({"gaia_phot", "gaia_astrometry", "gaia_rvs", "2mass"}), F({"wise", "galex"}),
                "vet-core flag only", "committed shells (703,555)", _cenotaph_committed,
                status="flag_only", tail_mode="flag",
                notes="used only when the harvested per-star grey fit is absent"),
    ChannelSpec("ignition", "ignition", F({"neowise"}), F({"wise"}),
                "two-band NEOWISE monotonic rise significance min(sigma_W1, sigma_W2)",
                "IGNITION tiles sweep parent (|b|>15, Gaia x AllWISE)", _harvested("ignition"),
                status="pending_harvest",
                source_files=["artifacts ignition-shard-* stars_s*.csv (run 35740159635)"]),
    ChannelSpec("cradle", "cradle", F({"wise"}), F({"2mass", "gaia_astrometry"}),
                "log(f/f_max) of the W3/W4 excess over the collisional maximum",
                "Gaia x AllWISE with W3 and W4 >= 5 sigma, |b|>10", _harvested("cradle"),
                status="pending_harvest",
                source_files=["artifact cradle-screen parent_screened.csv (run 35741356662)"]),
    ChannelSpec("ring_wd", "ring", F({"wise"}), F({"2mass", "gaia_phot"}),
                "white-dwarf W1/W2 excess significance", "Gentile Fusillo+2021 WDs x AllWISE",
                _harvested("ring_wd"), status="pending_harvest"),
    ChannelSpec("ossuary", "ossuary", F({"wise"}), F({"2mass", "gaia_rvs"}),
                "W1-W4 excess significance around metal-poor / halo stars",
                "Gaia metal-poor + halo tracks x AllWISE", _harvested("ossuary"),
                status="pending_harvest"),
    ChannelSpec("tailings", "tailings", F({"apogee", "galah"}), F(),
                "sparse single-element abundance outlier |z_max|",
                "GALAH DR4 + APOGEE DR17 cool dwarfs", _harvested("tailings"),
                status="pending_harvest"),
    ChannelSpec("accel_nss", "accel", F({"gaia_astrometry"}), F(),
                "Gaia DR3 nss_acceleration_astro significance",
                "every nss_acceleration_astro solution (acquired on the runner)",
                _harvested("accel_nss"), status="pending_harvest"),
    ChannelSpec("accel_committed", "accel", F({"gaia_astrometry"}), F(),
                "accel_significance of the committed ranked list", "7,047 ranked solutions",
                _accel_committed, status="tail_only",
                notes="a ranked cut of an unrecorded parent; superseded by accel_nss"),
    ChannelSpec("slag", "slag", F({"wd_spectroscopy"}), F(),
                "-log10 calibrated misfit p of the best natural parcel",
                "PEWDD polluted WDs with a >=5-element panel", _slag,
                source_files=["slag/misfit_list.csv", "slag/data/pewdd_vizier.csv"]),
    ChannelSpec("growth", "growth", F({"tess"}), F({"kepler"}),
                "TESS-vs-Kepler transit depth change z (PDCSAP)",
                "KOIs with a measured TESS depth", _growth,
                source_files=["growth/direct/measurements.csv"]),
    ChannelSpec("arc", "arc", F({"kepler", "tess"}), F(),
                "xi: flare energy above the starspot energy ceiling (conservative)",
                "4,206 assessable Kepler/TESS flare stars", _arc,
                notes="positions resolved from KIC/TIC on the runner (arc_positions.csv)",
                source_files=["arc/xi_table.csv", "confluence/arc_positions.csv"]),
    ChannelSpec("spark", "spark", F({"euclid"}), F(),
                "S/N of the strongest non-stellar emission feature",
                "Euclid Q1 point sources with >=1 line feature (Gaia-matched)", _spark,
                notes="parent = objects with at least one detected feature, not all spectra",
                source_files=["spark/euclid/features_all.csv.gz"]),
]

#: Channels whose committed outputs hold survivors only, what they would need to
#: emit, and whether their package is editable from here.
TAIL_ONLY: dict[str, dict] = {
    "rust": {"have": "top-500 stats per field (3,047 of 5,043)", "need": "all 5,043 rows",
             "instrument": "ztf", "editable": True},
    "knell": {"have": "cross candidates", "need": "per-star stats", "instrument": "ztf",
              "editable": True},
    "xp": {"have": "anomalies in 2 fields", "need": "per-star global_sigma over each field",
           "instrument": "gaia_xp", "editable": True},
    "baffle_deficit": {"have": "417,589 in-archive pre-selected deficit rows (committed 42,798 "
                       "vetoed + 74 candidates)", "need": "a parent-level residual (30M) is not "
                       "fetched at all -- the parent can only be reconstructed by selection",
                       "instrument": "wise", "editable": True},
    "science_wd": {"have": "170 candidates", "need": "per-WD chi_W1/chi_W2", "instrument": "wise",
                   "editable": True},
    "spectra": {"have": "350 triaged lines", "need": "per-spectrum max line significance over "
                "the SDSS/DESI stellar parent", "instrument": "sdss/desi", "editable": True},
    "fallout": {"have": "21 candidates", "need": "per-star template misfit over GALAH DR4",
                "instrument": "galah", "editable": True},
    "metronome": {"have": "screen counts", "need": "per-star clock statistic",
                  "instrument": "kepler/tess", "editable": True},
    "ember": {"have": "5,000 of 412,914 ladder verdicts", "need": "per-star early-epoch excess",
              "instrument": "iras/akari/wise", "editable": True},
    "century": {"have": "targets only", "need": "per-star DASCH stats", "instrument": "dasch",
                "editable": False},
    "tocsin_ztf": {"have": "220-star ledger", "need": "per-star event rate over 163,768 "
                   "star-nights", "instrument": "ztf_alerts", "editable": True},
}


def by_name() -> dict[str, ChannelSpec]:
    return {c.name: c for c in REGISTRY}
