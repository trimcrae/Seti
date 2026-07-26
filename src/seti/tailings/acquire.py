"""Archive pulls for TAILINGS (runner-only), with provenance and honest fallback.

Sandbox egress to VizieR/Gaia/SDSS is 403-blocked, so every function here is
written to be exercised on a GitHub Actions runner and to be *importable*
offline: all network libraries are imported inside the functions, and every
fetch takes an injectable transport so the offline tests can drive the same
code paths without a socket.

Source strategy
---------------
Catalogue names drift between data releases and between VizieR and the survey's
own services. Rather than hard-code one locator and fail the run when it moves,
each survey has an ordered list of **candidate sources**. The first one that
answers is used and the choice is recorded in the provenance block, so the
report always states which release the numbers actually came from instead of
asserting the one that was intended. A survey where nothing answers yields no
rows and an explicit degradation flag -- never a silently smaller sample.

Columns are resolved dynamically for the same reason. GALAH's native schema
(``mg_fe``, ``e_mg_fe``, ``flag_mg_fe``), APOGEE's (``MG_FE``, ``MG_FE_ERR``,
``MG_FE_FLAG``) and VizieR's mangling of both (``__Mg_Fe_``, ``e__Mg_Fe_``,
``f__Mg_Fe_``) all reduce to the same canonical element table by pattern, so a
schema change costs nothing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"


@dataclass(frozen=True)
class Source:
    """One candidate locator for a survey catalogue."""

    name: str
    kind: str  # "tap" | "file"
    locator: str  # TAP table name, or URL
    note: str = ""


#: Ordered candidates per survey. GALAH DR4 first, DR3 as the fallback that is
#: certain to exist on VizieR; the run states which one it used.
SOURCES: dict[str, tuple[Source, ...]] = {
    "GALAH": (
        Source("GALAH_DR4_vizier", "tap", "III/298/galahdr4", "GALAH DR4 main catalogue on VizieR"),
        Source("GALAH_DR4_vizier_alt", "tap", "III/297/galahdr4", "alternate DR4 VizieR number"),
        Source("GALAH_DR3_vizier", "tap", "III/283/allstar", "GALAH DR3 allstar (588k) -- certain"),
    ),
    "APOGEE": (
        Source("APOGEE_DR17_vizier", "tap", "III/286/catalog", "APOGEE-2 DR17 on VizieR"),
        Source(
            "APOGEE_DR17_sas",
            "file",
            "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
            "allStarLite-dr17-synspec_rev1.fits",
            "SDSS SAS allStarLite (large; streamed with column selection)",
        ),
    ),
    "LAMOST": (
        Source("LAMOST_MRS_vizier", "tap", "V/156/dr7mrs", "LAMOST MRS parameter catalogue"),
    ),
    "WIDEBINARY": (
        Source(
            "ELBADRY2021",
            "tap",
            "J/MNRAS/506/2269/table1",
            "El-Badry, Rix & Heintz 2021 Gaia eDR3 wide binaries (1.3M pairs)",
        ),
    ),
}

#: Elements to request. A superset; whatever a survey lacks is simply absent.
TARGET_ELEMENTS: tuple[str, ...] = (
    "Li", "C", "N", "O", "Na", "Mg", "Al", "Si", "K", "Ca", "Sc", "Ti", "TiII",
    "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn", "Rb", "Sr", "Y", "Zr", "Mo", "Ru",
    "Ba", "La", "Ce", "Nd", "Sm", "Eu",
)

_ELEMENT_CASE = {e.lower(): e for e in TARGET_ELEMENTS}
_ELEMENT_CASE.update({"fe": "Fe", "ti2": "TiII", "tiii": "TiII", "ti_ii": "TiII"})

_ABUND_RE = re.compile(r"^([a-z]{1,4})_fe$")


def _canon(name: str) -> str:
    """Lowercase, strip VizieR's decorative underscores, collapse doubles."""
    s = str(name).strip().lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def resolve_abundance_columns(columns) -> dict[str, dict[str, str]]:
    """Map catalogue columns onto ``{element: {value, err, flag}}``.

    Understands the GALAH, APOGEE and VizieR naming conventions at once, so a
    schema change in any of them does not need a code change.
    """
    out: dict[str, dict[str, str]] = {}
    for raw in columns:
        c = _canon(raw)
        kind = "value"
        rest = c
        if c.startswith("e_"):
            kind, rest = "err", c[2:]
        elif c.startswith("flag_"):
            kind, rest = "flag", c[5:]
        elif c.startswith("f_"):
            kind, rest = "flag", c[2:]
        elif c.endswith("_err"):
            kind, rest = "err", c[:-4]
        elif c.endswith("_flag"):
            kind, rest = "flag", c[:-5]
        elif c.endswith("_uncertainty"):
            kind, rest = "err", c[:-12]
        rest = _canon(rest)
        m = _ABUND_RE.match(rest)
        if not m:
            continue
        el = _ELEMENT_CASE.get(m.group(1))
        if el is None or el == "Fe":
            continue
        out.setdefault(el, {})[kind] = str(raw)
    return {el: d for el, d in out.items() if "value" in d}


_PARAM_PATTERNS: dict[str, tuple[str, ...]] = {
    "teff": (r"^teff$", r"^teff_spec$", r"^teff_gspspec$", r"^teff_?1?$"),
    "logg": (r"^logg$", r"^logg_spec$", r"^logg_?1?$"),
    "fe_h": (r"^fe_h$", r"^feh$", r"^m_h$", r"^__fe_h_$", r"^fe_h_atmo$"),
    "snr": (r"^snr$", r"^snr_c3_iraf$", r"^snrev$", r"^snr_?g?$", r"^s_n$"),
    "chi2": (r"^chi2_sp$", r"^chi2$", r"^aspcap_chi2$", r"^chi2_?fit$"),
    "ruwe": (r"^ruwe$",),
    "vbroad": (r"^vbroad$", r"^vsini$", r"^vmic$", r"^vmacro$"),
    "rv_scatter": (r"^vscatter$", r"^rv_scatter$", r"^e_rv$", r"^rv_?err$"),
    "field_id": (r"^field$", r"^field_id$", r"^plate$", r"^survey_field$"),
    "star_id": (r"^sobject_id$", r"^apogee_id$", r"^source_id$", r"^gaia.*source.*id$",
                r"^star_id$", r"^obsid$"),
    "ra": (r"^ra$", r"^raj2000$", r"^ra_?deg$", r"^_ra$"),
    "dec": (r"^de$", r"^dec$", r"^dej2000$", r"^de_?deg$", r"^_de$"),
}


def resolve_param_columns(columns) -> dict[str, str]:
    """Map catalogue columns onto the canonical stellar-parameter names."""
    canon = {_canon(c): str(c) for c in columns}
    out: dict[str, str] = {}
    for key, pats in _PARAM_PATTERNS.items():
        for pat in pats:
            rx = re.compile(pat)
            hit = next((orig for cc, orig in canon.items() if rx.match(cc)), None)
            if hit is not None:
                out[key] = hit
                break
    return out


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def _retry(fn, retries: int = 3, label: str = "fetch"):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one flaky call must not kill a run
            last = exc
            print(f"[tailings] {label} attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def tap_query(adql: str, *, url: str = VIZIER_TAP, retries: int = 3) -> pd.DataFrame:
    """Run an ADQL query against a TAP service and return a DataFrame.

    Asynchronous first: a synchronous VizieR query silently truncates or times
    out well below the row counts this channel needs, and a truncated
    catalogue would quietly shrink the sample without saying so. The sync path
    is kept as the last-attempt fallback, matching the pattern the other
    channels in this repository settled on.
    """
    import pyvo  # noqa: PLC0415 - runner-only import; keeps the module offline-importable

    def _go():
        svc = pyvo.dal.TAPService(url)
        try:
            return svc.run_async(adql).to_table().to_pandas()
        except Exception as exc:  # noqa: BLE001
            print(f"[tailings] async TAP failed ({exc!r}); retrying synchronously")
            return svc.search(adql).to_table().to_pandas()

    return _retry(_go, retries=retries, label="TAP query")


#: Keywords used to *discover* a survey's table when the encoded locators miss.
DISCOVERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "GALAH": ("GALAH",),
    "APOGEE": ("APOGEE",),
    "LAMOST": ("LAMOST",),
    "WIDEBINARY": ("wide binar", "El-Badry", "El-Badry", "binaries"),
}


def discover_tables(
    keywords: tuple[str, ...],
    *,
    url: str = VIZIER_TAP,
    query_fn=None,
    limit: int = 40,
) -> list[str]:
    """Find candidate tables by keyword in ``TAP_SCHEMA``.

    VizieR catalogue *numbers* drift between data releases -- III/283 is not
    III/298, and neither is guaranteed to be the current GALAH. Hard-coding one
    means the channel dies the day CDS renumbers, which is exactly what
    happened on the first dispatch. Asking the service what it actually holds
    removes the whole failure mode, and it is the same reasoning that made the
    *column* names dynamic.
    """
    query_fn = query_fn or (lambda q: tap_query(q, url=url))
    pats = []
    for k in keywords:
        for variant in {k, k.upper(), k.lower(), k.capitalize()}:
            pats.append(f"description LIKE '%{variant}%'")
            pats.append(f"table_name LIKE '%{variant}%'")
    adql = (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            "WHERE " + " OR ".join(dict.fromkeys(pats)))
    try:
        df = query_fn(adql)
    except Exception as exc:  # noqa: BLE001
        print(f"[tailings] table discovery failed: {exc!r}")
        return []
    if df is None or len(df) == 0:
        return []
    col = "table_name" if "table_name" in df.columns else df.columns[0]
    return [str(t) for t in df[col].tolist()]


def score_schema(columns) -> tuple[int, dict[str, str], dict[str, dict[str, str]]]:
    """Rank a candidate table by how much of what this channel needs it has.

    Returns ``(score, params, elements)``; a score of 0 means unusable. Ranking
    rather than first-match matters because discovery returns many tables per
    survey (per-field subsets, VACs, README stubs) and only one of them is the
    main abundance catalogue.
    """
    params = resolve_param_columns(columns)
    elements = resolve_abundance_columns(columns)
    if any(k not in params for k in ("teff", "logg", "fe_h")) or not elements:
        return 0, params, elements
    score = 10 * len(elements)
    for k, w in (("snr", 5), ("star_id", 5), ("chi2", 3), ("ruwe", 3),
                 ("vbroad", 2), ("rv_scatter", 2), ("field_id", 2),
                 ("ra", 1), ("dec", 1)):
        if k in params:
            score += w
    return score, params, elements


def teff_bands(teff_min: float, teff_max: float, n: int) -> list[tuple[float, float]]:
    """Split the temperature range into ``n`` query chunks.

    A single monolithic query at >10^5 rows times out; chunking bounds every
    request and lets a lost chunk be re-fetched on its own. Teff is the natural
    axis because it is indexed in every one of these catalogues and because the
    cool-dwarf selection is already a Teff cut.
    """
    edges = np.linspace(float(teff_min), float(teff_max), int(n) + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(int(n))]


def probe_table(table: str, *, url: str = VIZIER_TAP) -> pd.DataFrame | None:
    """Fetch one row to discover a table's real column names, or None if absent."""
    try:
        return tap_query(f"SELECT TOP 1 * FROM \"{table}\"", url=url, retries=1)
    except Exception as exc:  # noqa: BLE001
        print(f"[tailings] probe of {table} failed: {exc!r}")
        return None


@dataclass
class Acquisition:
    """What a survey pull actually produced, including how it degraded."""

    survey: str
    table: pd.DataFrame
    source_used: str | None
    locator: str | None
    n_rows: int
    elements: list[str]
    param_columns: dict[str, str] = field(default_factory=dict)
    degraded: bool = False
    degradation: str = ""
    sources_tried: list[str] = field(default_factory=list)
    scoreboard: list[dict] = field(default_factory=list)

    def provenance(self) -> dict:
        return {
            "survey": self.survey,
            "source_used": self.source_used,
            "locator": self.locator,
            "n_rows": int(self.n_rows),
            "n_elements": len(self.elements),
            "elements": self.elements,
            "degraded": bool(self.degraded),
            "degradation": self.degradation,
            "sources_tried": self.sources_tried,
            "scoreboard": self.scoreboard,
        }


COOL_DWARF_ADQL = (
    "{teff} < {teff_max} AND {teff} > {teff_min} AND {logg} > {logg_min} "
    "AND {snr} > {snr_min}"
)


def fetch_survey(
    survey: str,
    *,
    teff_max: float = 6000.0,
    teff_min: float = 3000.0,
    logg_min: float = 4.0,
    snr_min: float = 40.0,
    max_rows: int = 400_000,
    n_chunks: int = 8,
    tap_url: str = VIZIER_TAP,
    discover: bool = True,
    max_candidates: int = 12,
    probe_fn=None,
    query_fn=None,
) -> Acquisition:
    """Pull cool dwarfs with abundances from the BEST table this service holds.

    Not the first that answers: the *best*. Discovery returns many tables per
    survey -- per-field subsets, value-added catalogues, README stubs -- and
    only one of them is the main abundance catalogue. Each candidate is probed
    for one row, scored by how much of what this channel needs it actually has
    (stellar parameters, SNR, an identifier, and above all how many elements),
    and the highest scorer is used. The full scoreboard travels in the
    provenance, so the choice is auditable rather than incidental.

    ``probe_fn`` and ``query_fn`` exist so the offline suite can drive the full
    selection and degradation logic without a network.
    """
    probe_fn = probe_fn or (lambda t: probe_table(t, url=tap_url))
    query_fn = query_fn or (lambda q: tap_query(q, url=tap_url))

    encoded = [s.locator for s in SOURCES.get(survey, ()) if s.kind == "tap"]
    names = {s.locator: s.name for s in SOURCES.get(survey, ())}
    candidates = list(encoded)
    if discover:
        found = discover_tables(DISCOVERY_KEYWORDS.get(survey, (survey,)),
                                url=tap_url, query_fn=query_fn)
        candidates += [t for t in found if t not in candidates]
        print(f"[tailings] {survey}: {len(encoded)} encoded + "
              f"{len(candidates) - len(encoded)} discovered candidate tables")

    tried: list[str] = []
    scoreboard: list[dict] = []
    best = None
    for locator in candidates[:max_candidates]:
        tried.append(names.get(locator, locator))
        head = probe_fn(locator)
        if head is None or len(head.columns) == 0:
            scoreboard.append({"table": locator, "score": 0, "why": "no response"})
            continue
        score, params, elements = score_schema(head.columns)
        scoreboard.append({"table": locator, "score": int(score),
                           "n_elements": len(elements),
                           "why": "usable" if score else "missing parameters or elements"})
        if score and (best is None or score > best[0]):
            best = (score, locator, params, elements)

    if best is None:
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=None, locator=None,
            n_rows=0, elements=[], degraded=True,
            degradation="NO_DATA_REACHED: no candidate table for this survey had a usable schema",
            sources_tried=tried, scoreboard=scoreboard,
        )

    score, locator, params, elements = best
    label = names.get(locator, locator)
    print(f"[tailings] {survey}: using {label} ({locator}), score {score}, "
          f"{len(elements)} elements")

    cols: list[str] = []
    for k in ("star_id", "ra", "dec", "teff", "logg", "fe_h", "snr", "chi2",
              "ruwe", "vbroad", "rv_scatter", "field_id"):
        if k in params:
            cols.append(f'"{params[k]}"')
    for _el, d in elements.items():
        for kind in ("value", "err", "flag"):
            if kind in d:
                cols.append(f'"{d[kind]}"')

    select = ", ".join(dict.fromkeys(cols))
    bands = teff_bands(teff_min, teff_max, max(1, int(n_chunks)))
    per_chunk = max(1, int(max_rows) // len(bands))
    frames: list[pd.DataFrame] = []
    for lo, hi in bands:
        where = COOL_DWARF_ADQL.format(
            teff=f'"{params["teff"]}"',
            teff_max=hi,
            teff_min=lo,
            logg=f'"{params["logg"]}"',
            logg_min=logg_min,
            snr=f'"{params["snr"]}"' if "snr" in params else "1e9",
            snr_min=snr_min,
        )
        adql = (f"SELECT TOP {per_chunk} " + select
                + f' FROM "{locator}" WHERE ' + where)
        try:
            chunk = query_fn(adql)
        except Exception as exc:  # noqa: BLE001 - a lost chunk is not a lost run
            print(f"[tailings] {survey}/{label}: chunk {lo:.0f}-{hi:.0f} K failed: {exc!r}")
            continue
        if chunk is not None and len(chunk):
            frames.append(chunk)
            print(f"[tailings] {survey}/{label}: {len(chunk)} rows in {lo:.0f}-{hi:.0f} K")

    if not frames:
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=label, locator=locator,
            n_rows=0, elements=sorted(elements), param_columns=params, degraded=True,
            degradation=("NO_DATA_REACHED: the chosen table has a usable schema but "
                         "returned zero rows under the cool-dwarf selection"),
            sources_tried=tried, scoreboard=scoreboard,
        )

    df = pd.concat(frames, ignore_index=True)
    truncated = [1 for f in frames if len(f) >= per_chunk]
    norm = normalize(df, survey=survey)

    notes = []
    preferred = SOURCES.get(survey, ())
    if not preferred or locator != preferred[0].locator:
        notes.append(f"used {label} ({locator}) rather than the preferred "
                     f"{preferred[0].name if preferred else 'n/a'}")
    if truncated:
        # A chunk that returns exactly its cap was cut off; the sample is a
        # truncation of the catalogue, not the catalogue, and the report must
        # say so rather than quoting a row count as coverage.
        notes.append(f"{len(truncated)}/{len(bands)} Teff chunks hit the "
                     f"{per_chunk}-row cap: the sample is TRUNCATED, "
                     "raise --max-rows for full coverage")
    if len(frames) < len(bands):
        notes.append(f"only {len(frames)}/{len(bands)} Teff chunks returned; "
                     "the temperature coverage is incomplete")
    return Acquisition(
        survey=survey, table=norm, source_used=label, locator=locator,
        n_rows=len(norm), elements=sorted(resolve_abundance_columns(df.columns)),
        param_columns=params, degraded=bool(notes), degradation="; ".join(notes),
        sources_tried=tried, scoreboard=scoreboard,
    )


def normalize(df: pd.DataFrame, *, survey: str) -> pd.DataFrame:
    """Rename a raw survey table onto the canonical TAILINGS schema.

    Canonical: ``star_id, ra, dec, teff, logg, fe_h, snr, chi2, ruwe, vbroad,
    rv_scatter, field_id``; abundances as ``<El>``, errors ``e_<El>``, flags
    ``f_<El>``.
    """
    params = resolve_param_columns(df.columns)
    elements = resolve_abundance_columns(df.columns)
    out = pd.DataFrame(index=df.index)
    for canon, orig in params.items():
        out[canon] = df[orig]
    for el, d in elements.items():
        out[el] = pd.to_numeric(df[d["value"]], errors="coerce")
        if "err" in d:
            out[f"e_{el}"] = pd.to_numeric(df[d["err"]], errors="coerce")
        if "flag" in d:
            out[f"f_{el}"] = df[d["flag"]]
    for c in ("teff", "logg", "fe_h", "snr", "chi2", "ruwe", "vbroad", "rv_scatter"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["survey"] = survey
    return out


def apply_element_flags(df: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """Blank out any abundance whose own per-element quality flag is non-zero.

    Pipelines publish a per-element flag precisely because the value is not
    always usable. Honouring it before the manifold is fitted keeps known-bad
    measurements out of both the reference surface and the candidate list.
    """
    out = df.copy()
    for el in elements:
        fcol = f"f_{el}"
        if fcol not in out.columns or el not in out.columns:
            continue
        v = out[fcol]
        bad = (
            ~v.fillna("").astype(str).str.strip().isin(["", "0", "0.0", "nan", "False"])
            if v.dtype == object
            else (pd.to_numeric(v, errors="coerce").fillna(1) != 0)
        )
        out.loc[bad, el] = np.nan
    return out


def fetch_wide_binaries(
    *,
    max_rows: int = 200_000,
    max_r_chance_align: float = 0.1,
    tap_url: str = VIZIER_TAP,
    discover: bool = True,
    max_candidates: int = 12,
    probe_fn=None,
    query_fn=None,
) -> Acquisition:
    """Gaia wide binaries, cut on the catalogue's own chance-alignment estimate.

    ``R_chance_align`` is the probability that a pair is a projection rather
    than a bound system. Keeping it below 0.1 is the standard purity cut and it
    matters here more than usual: a chance alignment of two unrelated stars has
    no reason to share a composition, and would manufacture exactly the
    differential anomaly stage 4 looks for. Where the catalogue does not carry
    that column the pull still proceeds, but the omission is recorded as
    degradation rather than passed over.
    """
    probe_fn = probe_fn or (lambda t: probe_table(t, url=tap_url))
    query_fn = query_fn or (lambda q: tap_query(q, url=tap_url))

    encoded = [s.locator for s in SOURCES["WIDEBINARY"]]
    names = {s.locator: s.name for s in SOURCES["WIDEBINARY"]}
    candidates = list(encoded)
    if discover:
        found = discover_tables(DISCOVERY_KEYWORDS["WIDEBINARY"], url=tap_url,
                                query_fn=query_fn)
        candidates += [t for t in found if t not in candidates]

    tried: list[str] = []
    scoreboard: list[dict] = []
    for locator in candidates[:max_candidates]:
        label = names.get(locator, locator)
        tried.append(label)
        head = probe_fn(locator)
        if head is None or len(head.columns) == 0:
            scoreboard.append({"table": locator, "score": 0, "why": "no response"})
            continue
        cols = {_canon(c): str(c) for c in head.columns}
        id1 = next((v for k, v in cols.items()
                    if k in ("source_id1", "gaiaedr3_1", "source1", "sourceid1")), None)
        id2 = next((v for k, v in cols.items()
                    if k in ("source_id2", "gaiaedr3_2", "source2", "sourceid2")), None)
        rca = next((v for k, v in cols.items()
                    if k in ("r_chance_align", "rchancealign", "rchance")), None)
        if not (id1 and id2):
            scoreboard.append({"table": locator, "score": 0,
                               "why": "no pair of Gaia source identifiers"})
            continue
        scoreboard.append({"table": locator, "score": 1 + (1 if rca else 0),
                           "why": "usable"})
        sel = f'"{id1}", "{id2}"' + (f', "{rca}"' if rca else "")
        where = f' WHERE "{rca}" < {max_r_chance_align}' if rca else ""
        try:
            df = query_fn(f"SELECT TOP {int(max_rows)} {sel} FROM \"{locator}\"{where}")
        except Exception as exc:  # noqa: BLE001
            print(f"[tailings] wide binaries {locator}: query failed: {exc!r}")
            continue
        if df is None or len(df) == 0:
            continue
        df = df.rename(columns={id1: "source_id_a", id2: "source_id_b",
                                **({rca: "r_chance_align"} if rca else {})})
        notes = []
        if not rca:
            notes.append("catalogue has no R_chance_align column: chance alignments "
                         "are NOT removed and every pair verdict is provisional")
        if len(df) >= int(max_rows):
            notes.append(f"hit the {max_rows}-row cap: the pair list is TRUNCATED")
        print(f"[tailings] wide binaries: {len(df)} pairs from {label}")
        return Acquisition(survey="WIDEBINARY", table=df, source_used=label,
                           locator=locator, n_rows=len(df), elements=[],
                           degraded=bool(notes), degradation="; ".join(notes),
                           sources_tried=tried, scoreboard=scoreboard)
    return Acquisition(survey="WIDEBINARY", table=pd.DataFrame(), source_used=None,
                       locator=None, n_rows=0, elements=[], degraded=True,
                       degradation="NO_DATA_REACHED: wide-binary catalogue unreachable",
                       sources_tried=tried, scoreboard=scoreboard)


def join_pairs(
    pairs: pd.DataFrame,
    stars: pd.DataFrame,
    elements: list[str],
    *,
    id_col: str = "star_id",
) -> pd.DataFrame:
    """Attach both components' abundances to each wide-binary pair.

    Only pairs where *both* components have a spectrum survive; that is the
    whole point of the differential test.
    """
    if pairs.empty or stars.empty:
        return pd.DataFrame()
    keep = [id_col, "teff", "logg", "fe_h", "snr"] + [
        c for el in elements for c in (el, f"e_{el}") if c in stars.columns
    ]
    keep = [c for c in dict.fromkeys(keep) if c in stars.columns]
    s = stars[keep].copy()
    s[id_col] = s[id_col].astype(str)

    a = s.add_prefix("a_").rename(columns={f"a_{id_col}": "source_id_a"})
    b = s.add_prefix("b_").rename(columns={f"b_{id_col}": "source_id_b"})
    p = pairs.copy()
    p["source_id_a"] = p["source_id_a"].astype(str)
    p["source_id_b"] = p["source_id_b"].astype(str)
    out = p.merge(a, on="source_id_a", how="inner").merge(b, on="source_id_b", how="inner")
    out["pair_id"] = out["source_id_a"] + "_" + out["source_id_b"]
    return out


def write_checkpoint(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a parquet checkpoint, creating parents. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


__all__ = [
    "Acquisition",
    "SOURCES",
    "TARGET_ELEMENTS",
    "VIZIER_TAP",
    "DISCOVERY_KEYWORDS",
    "Source",
    "apply_element_flags",
    "discover_tables",
    "fetch_survey",
    "fetch_wide_binaries",
    "join_pairs",
    "normalize",
    "probe_table",
    "resolve_abundance_columns",
    "resolve_param_columns",
    "score_schema",
    "tap_query",
    "teff_bands",
    "write_checkpoint",
]
