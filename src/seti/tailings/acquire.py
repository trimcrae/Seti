"""Archive pulls for TAILINGS (runner-only), with provenance and honest fallback.

Sandbox egress to VizieR/Gaia/SDSS is 403-blocked, so every function here is
written to be exercised on a GitHub Actions runner and to be *importable*
offline: all network libraries are imported inside the functions, and every
fetch takes an injectable transport so the offline tests can drive the same
code paths without a socket.

Source strategy
---------------
**Bulk survey-native files are the primary route; VizieR TAP is the fallback.**
Three dispatches (runs 30203627605, 30204487245, 30204793446) reached the
service, ran clean, and returned *zero rows at every stage*, for two reasons
that both belong to VizieR rather than to the archive in general:

1. every encoded VizieR locator (``III/298/galahdr4``, ``III/297/galahdr4``,
   ``III/283/allstar``, ``III/286/catalog``, ``J/MNRAS/506/2269/table1``) came
   back "table not found" -- CDS catalogue numbers drift between releases; and
2. the ~34 tables auto-discovered as substitutes all scored zero elements: they
   are per-field subsets and value-added stubs, not the main abundance
   catalogue.

There is also a *scientific* reason not to depend on VizieR here. Correction #5
in ``docs/tailings.md`` requires de-trending the flag rate against **radial
velocity, fibre ID and detector position** -- Weinberg's two high-Ca APOGEE
stars were a bad-pixel/fibre combination and a whole population of low-K stars
was a heliocentric velocity sliding K onto a telluric band. VizieR's abbreviated
tables generally drop the fibre/plate columns entirely, so a VizieR-only pull
cannot run that veto at all. The survey-native FITS files carry them.

Because the sandbox cannot verify a URL, nothing here hard-codes one. Each
survey has an **ordered list of candidate URLs**; :func:`probe_download_routes`
issues a HEAD (falling back to a one-byte ranged GET) against every one of them
and records the status, the content length and whether it was selected, into
the provenance block. The next dispatch therefore *tells us which URL is live*
instead of failing silently.

Columns are resolved dynamically for the same reason. GALAH's native schema
(``mg_fe``, ``e_mg_fe``, ``flag_mg_fe``), APOGEE's (``MG_FE``, ``MG_FE_ERR``,
``MG_FE_FLAG``) and VizieR's mangling of both (``__Mg_Fe_``, ``e__Mg_Fe_``,
``f__Mg_Fe_``) all reduce to the same canonical element table by pattern, so a
schema change costs nothing.

Verdicts
--------
A channel whose null verdict is indistinguishable from its bug verdict will
mislead its own author, so the three failures are separated:

``NO_DATA_REACHED``          nothing answered -- no live URL, no usable table.
``QUERY_FAILED``             something answered and then errored: a download
                             that died, or every TAP chunk raising.
``QUERY_RETURNED_ZERO_ROWS`` the source answered correctly and the selection
                             matched nothing. This is a statement about the
                             *cuts*, not about the archive.

Every :class:`Acquisition` also carries ``stage_counts``: rows and candidates at
each step of the funnel, which travel into ``summary.json`` via the provenance.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Verdicts -- first-class, and deliberately not collapsed into one another
# ---------------------------------------------------------------------------
VERDICT_OK = "OK"
VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_QUERY_FAILED = "QUERY_FAILED"
VERDICT_ZERO_ROWS = "QUERY_RETURNED_ZERO_ROWS"

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


#: Ordered TAP candidates per survey -- the **fallback** route. All of these
#: were tried on the first three dispatches and none resolved; they are kept
#: because a locator that is dead today may be the current one tomorrow, and
#: because runtime discovery needs a seed list to compare against.
SOURCES: dict[str, tuple[Source, ...]] = {
    "GALAH": (
        Source("GALAH_DR4_vizier", "tap", "III/298/galahdr4", "GALAH DR4 main catalogue on VizieR"),
        Source("GALAH_DR4_vizier_alt", "tap", "III/297/galahdr4", "alternate DR4 VizieR number"),
        Source("GALAH_DR3_vizier", "tap", "III/283/allstar", "GALAH DR3 allstar (588k)"),
    ),
    "APOGEE": (
        Source("APOGEE_DR17_vizier", "tap", "III/286/catalog", "APOGEE-2 DR17 on VizieR"),
        Source("APOGEE_DR16_vizier", "tap", "III/284/allstars", "APOGEE-2 DR16 on VizieR"),
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


@dataclass(frozen=True)
class DownloadRoute:
    """One candidate bulk-file URL for a survey catalogue.

    ``abundances=False`` marks a file that is known *not* to carry an element
    panel -- a value-added catalogue of ages or kinematics, say. Those are still
    probed, because a 200 from one of them proves the host and the path prefix
    are live and tells the next dispatch where to look, but they are never
    selected for the main pull: downloading 300 MB to discover it has no
    abundances is a waste, and silently accepting it would shrink the element
    vector without saying so.
    """

    name: str
    url: str
    note: str = ""
    abundances: bool = True


#: Ordered bulk-download candidates. **None of these URLs can be verified from
#: the sandbox** (egress is 403-blocked), which is exactly why there is a list
#: and a prober rather than a single hard-coded string. The probe report in
#: ``provenance.json`` records the status and content-length of every one.
FILE_ROUTES: dict[str, tuple[DownloadRoute, ...]] = {
    "GALAH": (
        DownloadRoute(
            "GALAH_DR4_allstar_dc",
            "https://datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/"
            "galah_dr4_allstar_240705.fits",
            "GALAH DR4 main allstar catalogue (Data Central)",
        ),
        DownloadRoute(
            "GALAH_DR4_allstar_cloud",
            "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/"
            "galah_dr4_allstar_240705.fits",
            "same file on the cloud.datacentral hostname",
        ),
        DownloadRoute(
            "GALAH_DR4_allstar_flat",
            "https://datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/"
            "galah_dr4_allstar_240705.fits",
            "DR4 without the catalogs/ path element",
        ),
        DownloadRoute(
            "GALAH_DR4_allspec_dc",
            "https://datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/"
            "galah_dr4_allspec_240705.fits",
            "DR4 allspec (per-spectrum; dedupe on star_id handles the repeats)",
        ),
        DownloadRoute(
            "GALAH_DR3_allstar_v2_cloud",
            "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
            "GALAH_DR3_main_allstar_v2.fits",
            "GALAH DR3 main allstar v2 (588k stars, 30 elements) -- the safest bet",
        ),
        DownloadRoute(
            "GALAH_DR3_allstar_v2_dc",
            "https://datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
            "GALAH_DR3_main_allstar_v2.fits",
            "same file on the bare datacentral hostname",
        ),
        DownloadRoute(
            "GALAH_DR3_allstar_v1_cloud",
            "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
            "GALAH_DR3_main_allstar_v1.fits",
            "DR3 v1 of the same catalogue",
        ),
        DownloadRoute(
            "GALAH_DR3_allspec_v2_cloud",
            "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
            "GALAH_DR3_main_allspec_v2.fits",
            "DR3 per-spectrum table",
        ),
        DownloadRoute(
            "GALAH_DR3_VAC_dynamics",
            "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
            "GALAH_DR3_VAC_dynamics_v2.fits",
            "VAC: kinematics only, no element panel -- probed to prove the host, never used",
            abundances=False,
        ),
        DownloadRoute(
            "GALAH_DR3_VAC_ages",
            "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
            "GALAH_DR3_VAC_ages_v2.fits",
            "VAC: ages/masses only -- probed to prove the host, never used",
            abundances=False,
        ),
    ),
    "APOGEE": (
        DownloadRoute(
            "APOGEE_DR17_allStarLite_rev1",
            "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
            "allStarLite-dr17-synspec_rev1.fits",
            "DR17 ASPCAP allStarLite (rev1) -- the recommended DR17 abundances, ~0.5 GB",
        ),
        DownloadRoute(
            "APOGEE_DR17_allStar_rev1",
            "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
            "allStar-dr17-synspec_rev1.fits",
            "the full allStar (~2.5 GB); read column-by-column, never wholly in RAM",
        ),
        DownloadRoute(
            "APOGEE_DR17_allStarLite_synspec",
            "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/"
            "allStarLite-dr17-synspec.fits",
            "the non-rev1 synspec variant",
        ),
        DownloadRoute(
            "APOGEE_DR17_allStar_synspec",
            "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/"
            "allStar-dr17-synspec.fits",
            "the non-rev1 synspec full table",
        ),
        DownloadRoute(
            "APOGEE_DR17_allStar_rev1_dr17host",
            "https://dr17.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
            "allStar-dr17-synspec_rev1.fits",
            "the dr17.sdss.org hostname for the same SAS path",
        ),
        DownloadRoute(
            "APOGEE_DR16_allStar",
            "https://data.sdss.org/sas/dr16/apogee/spectro/aspcap/r12/l33/"
            "allStar-r12-l33.fits",
            "DR16 ASPCAP -- an older release, used only if every DR17 route is dead",
        ),
    ),
    "LAMOST": (
        DownloadRoute(
            "LAMOST_DR8_MRS_stellar",
            "https://www.lamost.org/dr8/v2.0/catdl?name=dr8_v2.0_MRS_stellar.csv.gz",
            "LAMOST DR8 MRS stellar-parameter catalogue (low confidence in this path)",
        ),
        DownloadRoute(
            "LAMOST_DR7_MRS_stellar",
            "http://dr7.lamost.org/v2.0/catdl?name=dr7_v2.0_MRS_stellar.csv.gz",
            "LAMOST DR7 MRS (low confidence)",
        ),
    ),
    "WIDEBINARY": (
        DownloadRoute(
            "ELBADRY2021_zenodo_records",
            "https://zenodo.org/records/4435257/files/all_columns_catalog.fits.gz",
            "El-Badry, Rix & Heintz 2021 wide binaries, Zenodo 4435257",
            abundances=False,
        ),
        DownloadRoute(
            "ELBADRY2021_zenodo_record",
            "https://zenodo.org/record/4435257/files/all_columns_catalog.fits.gz?download=1",
            "the older /record/ URL form with the download query string",
            abundances=False,
        ),
        DownloadRoute(
            "ELBADRY2021_vizier_fits",
            "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/506/2269/table1.dat.gz",
            "the CDS FTP-over-HTTP copy of the same table",
            abundances=False,
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
_ABUND_XH_RE = re.compile(r"^([a-z]{1,4})_h$")


def resolve_xh_columns(columns) -> dict[str, str]:
    """Find ``[X/H]``-style abundance columns. **Diagnostic only.**

    The channel works in ``[X/Fe]`` and converts to ``[X/H]`` itself
    (``manifold.to_xh``), so accepting an ``[X/H]`` column as if it were
    ``[X/Fe]`` would silently double-subtract the metallicity and corrupt every
    residual. This function therefore never feeds the pull -- it exists so that
    a catalogue rejected for "no [X/Fe] columns" can say *"but it does carry 30
    [X/H] columns"*, which is a schema-convention problem with a one-line fix
    rather than a dead archive.
    """
    out: dict[str, str] = {}
    for raw in columns:
        c = _canon(raw)
        if c.startswith(("e_", "flag_", "f_")) or c.endswith(("_err", "_flag")):
            continue
        m = _ABUND_XH_RE.match(c)
        if not m:
            continue
        el = _ELEMENT_CASE.get(m.group(1))
        if el is None or el == "Fe":
            continue
        out[el] = str(raw)
    return out


def unquote_table(name: str) -> str:
    """Strip the double quotes VizieR's ``TAP_SCHEMA`` wraps table names in.

    ``TAP_SCHEMA.tables.table_name`` comes back as ``"III/283/allstar"`` --
    *including* the quote characters. Interpolating that into a quoted FROM
    clause yields ``FROM ""III/283/allstar""``, which every candidate table
    rejects, and the run then reports a clean NO_DATA_REACHED for what is
    really a quoting bug. It cost a dispatch; hence a named function rather
    than an inline strip.
    """
    return str(name).strip().strip('"').strip()


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
    # The instrumental covariates are not decoration: correction #5 requires the
    # flag rate to be de-trended against radial velocity, fibre and detector
    # position, because that is how Weinberg traced a whole population of low-K
    # APOGEE stars to a telluric band at v_helio ~ -70 km/s.
    "rv": (r"^rv$", r"^vhelio$", r"^vhelio_avg$", r"^rv_com$", r"^hrv$", r"^vrad$",
           r"^rv_galah$", r"^rv_gaia$", r"^rv_avg$", r"^radial_velocity$"),
    "fiber": (r"^fiber$", r"^fiberid$", r"^fibre$", r"^pivot$", r"^fibre_?id$",
              r"^meanfib$", r"^fiber_?num$", r"^minfib$"),
    "logg": (r"^logg$", r"^logg_spec$", r"^logg_?1?$"),
    "fe_h": (r"^fe_h$", r"^feh$", r"^m_h$", r"^__fe_h_$", r"^fe_h_atmo$"),
    "snr": (r"^snr$", r"^snr_c3_iraf$", r"^snrev$", r"^snr_?g?$", r"^s_n$",
            r"^snr_c2_iraf$", r"^snr_px$", r"^snr_.*$"),
    "chi2": (r"^chi2_sp$", r"^chi2$", r"^aspcap_chi2$", r"^chi2_?fit$"),
    "ruwe": (r"^ruwe$",),
    "vbroad": (r"^vbroad$", r"^vsini$", r"^vmic$", r"^vmacro$"),
    "rv_scatter": (r"^vscatter$", r"^rv_scatter$", r"^e_rv$", r"^rv_?err$",
                   r"^e_rv_galah$", r"^e_rv_.*$"),
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


# ---------------------------------------------------------------------------
# Bulk-file route: probe, download, read
# ---------------------------------------------------------------------------
def _http_probe(url: str, *, timeout: float = 30.0) -> dict:
    """HEAD a URL, falling back to a one-byte ranged GET.

    Two requests because the two failure modes are different: some servers
    (S3-style redirectors, some CDS mirrors) refuse HEAD with 403/405 while
    serving GET perfectly, and a channel that concluded "dead" from a refused
    HEAD would discard a live catalogue. The ranged GET costs one byte and
    returns the true total through ``Content-Range``.
    """
    import urllib.error  # noqa: PLC0415 - runner-only
    import urllib.request  # noqa: PLC0415

    def _attempt(method: str, headers: dict) -> dict:
        req = urllib.request.Request(url, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            hdrs = {k.lower(): v for k, v in dict(r.headers).items()}
            total = None
            crange = hdrs.get("content-range")
            if crange:
                m = re.search(r"/(\d+)\s*$", str(crange))
                total = int(m.group(1)) if m else None
            elif hdrs.get("content-length") is not None and method == "HEAD":
                try:
                    total = int(hdrs["content-length"])
                except (TypeError, ValueError):
                    total = None
            return {
                "status": int(getattr(r, "status", 0) or r.getcode()),
                "probe_method": method,
                "content_length": total,
                "content_type": hdrs.get("content-type"),
                "accept_ranges": hdrs.get("accept-ranges"),
                "final_url": r.geturl(),
                "error": None,
            }

    errors: list[str] = []
    last_status = 0
    for method, headers in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            out = _attempt(method, {"User-Agent": "seti-tailings/1.0", **headers})
            if out["status"] in (200, 206):
                return out
            last_status = int(out["status"])
            errors.append(f"{method} -> HTTP {out['status']}")
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            last_status = int(exc.code)
            errors.append(f"{method} -> HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{method} -> {exc!r}")
    return {"status": int(last_status), "probe_method": "HEAD+GET", "content_length": None,
            "content_type": None, "accept_ranges": None, "final_url": url,
            "error": "; ".join(errors)}


def probe_download_routes(
    routes,
    *,
    probe_fn=None,
    timeout: float = 30.0,
    min_bytes: int = 1_000_000,
) -> list[dict]:
    """Probe every candidate URL and report status, size and selection.

    The whole point is that the sandbox cannot check a URL, so the *runner*
    checks all of them and writes down what it saw. A route is **eligible**
    when it answers 200/206, is declared to carry an element panel, and is
    either of unknown length or larger than ``min_bytes`` -- a 4 kB reply to a
    catalogue request is an error page, not a catalogue, and accepting it would
    be the file-route version of the quoting bug that cost a dispatch.

    ``selected`` marks the first eligible route, i.e. the one the download will
    be attempted from. ``used`` is written later by the caller, once a route has
    actually yielded rows, so the two are never confused in the report.
    """
    probe_fn = probe_fn or (lambda u: _http_probe(u, timeout=timeout))
    report: list[dict] = []
    for route in routes:
        rec = {
            "name": route.name,
            "url": route.url,
            "note": route.note,
            "expects_abundances": bool(route.abundances),
            "status": 0,
            "content_length": None,
            "content_type": None,
            "error": None,
            "eligible": False,
            "selected": False,
            "used": False,
            "why": "",
        }
        try:
            rec.update({k: v for k, v in (probe_fn(route.url) or {}).items()
                        if k in ("status", "content_length", "content_type",
                                 "probe_method", "accept_ranges", "final_url", "error")})
        except Exception as exc:  # noqa: BLE001 - a dead URL is data, not a crash
            rec["error"] = repr(exc)
        live = int(rec.get("status") or 0) in (200, 206)
        size = rec.get("content_length")
        if not live:
            rec["why"] = f"no response (HTTP {rec.get('status')})"
        elif not route.abundances:
            rec["why"] = "live, but this file carries no element panel -- never used for the pull"
        elif size is not None and size < int(min_bytes):
            rec["why"] = (f"live but only {size} bytes: too small to be the catalogue "
                          "(almost certainly an error page)")
        else:
            rec["eligible"] = True
            rec["why"] = "live and eligible"
        report.append(rec)
        print(f"[tailings] route {route.name}: HTTP {rec.get('status')} "
              f"len={rec.get('content_length')} -> {rec['why']}")
    for rec in report:
        if rec["eligible"]:
            rec["selected"] = True
            break
    return report


def stream_download(
    url: str,
    dest: str | Path,
    *,
    timeout: float = 600.0,
    chunk_bytes: int = 1 << 22,
    retries: int = 3,
    opener=None,
) -> Path:
    """Stream a large file to disk in chunks, never holding it in RAM.

    These catalogues run to hundreds of megabytes (the full DR17 ``allStar`` is
    ~2.5 GB); reading one into a bytes object would kill the runner. The file
    lands as ``<name>.part`` and is renamed on completion, so a killed download
    can never be mistaken for a complete one by a later stage.
    """
    import urllib.request  # noqa: PLC0415 - runner-only

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[tailings] using cached {dest} ({dest.stat().st_size} bytes)")
        return dest
    tmp = dest.with_name(dest.name + ".part")

    def _go() -> Path:
        req = urllib.request.Request(url, headers={"User-Agent": "seti-tailings/1.0"})
        open_url = opener or (lambda r: urllib.request.urlopen(r, timeout=timeout))  # noqa: S310
        n = 0
        with open_url(req) as r, open(tmp, "wb") as fh:
            while True:
                buf = r.read(chunk_bytes)
                if not buf:
                    break
                fh.write(buf)
                n += len(buf)
        if n == 0:
            raise RuntimeError(f"empty body from {url}")
        tmp.replace(dest)
        print(f"[tailings] downloaded {url} -> {dest} ({n} bytes)")
        return dest

    return _retry(_go, retries=retries, label=f"download {url}")


@dataclass(frozen=True)
class Selection:
    """The cool-dwarf sample cut, carried as one object.

    Applied inside the FITS reader (so only the surviving rows are ever
    materialised) and again on whatever the reader returns, which makes it
    idempotent and lets an injected test reader ignore it entirely.
    """

    teff_min: float = 3000.0
    teff_max: float = 6000.0
    logg_min: float = 4.0
    snr_min: float = 40.0
    feh_min: float = -1.0
    max_rows: int | None = None

    def describe(self) -> dict:
        return {"teff_min": self.teff_min, "teff_max": self.teff_max,
                "logg_min": self.logg_min, "snr_min": self.snr_min,
                "feh_min": self.feh_min, "max_rows": self.max_rows}


def _numeric(a) -> np.ndarray:
    return pd.to_numeric(pd.Series(np.asarray(a).ravel()), errors="coerce").to_numpy(dtype=float)


def selection_mask(
    frame_or_cols,
    selection: Selection,
    *,
    getter=None,
    n_rows: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Boolean mask for the cool-dwarf cut, plus the cuts that could not be applied.

    Works either on a DataFrame or on a bare column-name list plus a ``getter``,
    so the FITS reader can build the mask from four memmapped columns without
    materialising the table.

    A cut that cannot be applied is *reported*, never silently dropped: a
    catalogue with no SNR column yields a sample with no SNR floor, and the
    contamination model depends on knowing that.
    """
    is_frame = hasattr(frame_or_cols, "columns")
    cols = list(frame_or_cols.columns) if is_frame else list(frame_or_cols)
    if getter is None:
        def getter(name):
            return frame_or_cols[name].to_numpy()
    if n_rows is None:
        n_rows = int(len(frame_or_cols)) if is_frame else 0
    params = resolve_param_columns(cols)
    mask: np.ndarray | None = None
    missing: list[str] = []
    tests = (
        ("teff", lambda v: (v > selection.teff_min) & (v < selection.teff_max)),
        ("logg", lambda v: v > selection.logg_min),
        ("snr", lambda v: v > selection.snr_min),
        ("fe_h", lambda v: v > selection.feh_min),
    )
    for key, test in tests:
        if key not in params:
            missing.append(key)
            continue
        v = _numeric(getter(params[key]))
        m = test(v) & np.isfinite(v)
        mask = m if mask is None else (mask & m)
    if mask is None:
        mask = np.ones(int(n_rows), dtype=bool)
    return mask, missing


def apply_selection(df: pd.DataFrame, selection: Selection | None) -> tuple[pd.DataFrame, list[str]]:
    """Apply the cool-dwarf cut to an in-memory table (idempotent)."""
    if selection is None or df is None or df.empty:
        return df, []
    mask, missing = selection_mask(df, selection)
    out = df.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)
    if selection.max_rows and len(out) > int(selection.max_rows):
        out = out.iloc[: int(selection.max_rows)].reset_index(drop=True)
    return out, missing


def _default_column_picker(colnames) -> list[str]:
    params = resolve_param_columns(colnames)
    elements = resolve_abundance_columns(colnames)
    want = list(params.values())
    for d in elements.values():
        want += [d[k] for k in ("value", "err", "flag") if k in d]
    return list(dict.fromkeys(want))


def read_fits_table(
    path: str | Path,
    *,
    selection: Selection | None = None,
    column_picker=None,
    hdu: int | None = None,
) -> pd.DataFrame:
    """Read only the needed columns of a large FITS table, lazily.

    ``memmap=True`` plus per-column access means the peak footprint is one
    column, not the file: a 733k-row float column is ~6 MB where the file is
    gigabytes. The row mask is built from the four parameter columns first and
    every other column is indexed with it, so the abundance panel is never
    materialised at full length.
    """
    from astropy.io import fits  # noqa: PLC0415 - runner-only

    with fits.open(str(path), memmap=True, lazy_load_hdus=True) as hdul:
        idx_hdu = hdu
        if idx_hdu is None:
            best = (-1, None)
            for i, h in enumerate(hdul):
                cols = getattr(getattr(h, "columns", None), "names", None)
                if not cols:
                    continue
                score, _, _ = score_schema(cols)
                if score > best[0]:
                    best = (score, i)
            idx_hdu = best[1] if best[1] is not None else 1
        h = hdul[idx_hdu]
        colnames = list(h.columns.names)
        n_total = int(h.header.get("NAXIS2", 0) or 0)

        def _get(name):
            return h.data[name]

        if selection is None:
            mask: np.ndarray = np.ones(n_total, dtype=bool)
            missing: list[str] = []
        else:
            mask, missing = selection_mask(colnames, selection, getter=_get, n_rows=n_total)
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if selection is not None and selection.max_rows and idx.size > int(selection.max_rows):
            idx = idx[: int(selection.max_rows)]

        picker = column_picker or _default_column_picker
        want = [c for c in picker(colnames) if c in colnames]
        out: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        for name in dict.fromkeys(want):
            arr = h.data[name]
            if getattr(arr, "ndim", 1) > 1:
                # Multi-dimensional columns (APOGEE's X_H panel, PARAM_COV) are
                # not scalars per star; the named scalar columns carry the same
                # numbers and resolve cleanly.
                skipped.append(name)
                continue
            v = np.asarray(arr)[idx]
            if v.dtype.kind in "SU":
                v = np.char.strip(v.astype(str))
            elif v.dtype.kind == "O":
                v = np.asarray([str(x).strip() for x in v])
            out[name] = v
        df = pd.DataFrame(out)
        # FITS is big-endian; pyarrow will not accept byte-swapped arrays, and
        # the failure surfaces much later at the first to_parquet.  Normalise
        # here so nothing downstream ever handles a '>i8'.
        df = to_native_byteorder(df)
        df.attrs["n_rows_file"] = n_total
        df.attrs["n_rows_selected"] = int(idx.size)
        df.attrs["hdu"] = int(idx_hdu)
        df.attrs["cuts_not_applied"] = missing
        df.attrs["columns_skipped_multidim"] = skipped
        return df


def download_and_read(
    url: str,
    *,
    selection: Selection | None = None,
    cache_dir: str | Path | None = None,
    column_picker=None,
    timeout: float = 600.0,
) -> pd.DataFrame:
    """Default bulk reader: stream the file to disk, then read what is needed."""
    cache = Path(cache_dir or os.environ.get("TAILINGS_CACHE", ".tailings_cache"))
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "catalogue.fits"
    path = stream_download(url, cache / name, timeout=timeout)
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        sep = "\t" if ".tsv" in name else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
        df.attrs["n_rows_file"] = int(len(df))
        df, missing = apply_selection(df, selection)
        df.attrs["n_rows_selected"] = int(len(df))
        df.attrs["cuts_not_applied"] = missing
        return df
    return read_fits_table(path, selection=selection, column_picker=column_picker)


#: Keywords used to *discover* a survey's table when the encoded locators miss.
DISCOVERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "GALAH": ("GALAH",),
    "APOGEE": ("APOGEE",),
    "LAMOST": ("LAMOST",),
    "WIDEBINARY": ("wide binar", "El-Badry", "El-Badry", "binaries"),
}

#: Elements probed against ``TAP_SCHEMA.columns`` when hunting for the real
#: abundance catalogue. Four is enough to identify it and keeps the ADQL short:
#: Mg (alpha), Ni (Fe-peak), Ba (s-process) and Ca span the families, so a table
#: carrying all four is an abundance catalogue and not a parameter stub.
DISCOVERY_ELEMENTS: tuple[str, ...] = ("Mg", "Ni", "Ba", "Ca")


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
    return [unquote_table(t) for t in df[col].tolist() if unquote_table(t)]


def discover_tables_by_columns(
    *,
    url: str = VIZIER_TAP,
    query_fn=None,
    elements: tuple[str, ...] = DISCOVERY_ELEMENTS,
    limit: int = 400,
) -> dict[str, list[str]]:
    """Find abundance catalogues by asking which tables carry abundance *columns*.

    Scoring a table by its *description* is what failed on the third dispatch:
    ~34 tables answered with a GALAH/APOGEE description and every one of them
    had zero elements. The description says what a catalogue is about; the
    column list says what it contains, and only the second is the thing this
    channel needs. VizieR mangles ``[Mg/Fe]`` into ``__Mg_Fe_``, so the LIKE
    pattern is deliberately loose on both sides and ``_canon`` does the rest.
    """
    query_fn = query_fn or (lambda q: tap_query(q, url=url))
    pats: list[str] = []
    for el in elements:
        for variant in {f"{el}_Fe", f"{el.lower()}_fe", f"{el.upper()}_FE"}:
            pats.append(f"column_name LIKE '%{variant}%'")
    adql = (f"SELECT TOP {int(limit)} table_name, column_name FROM TAP_SCHEMA.columns "
            "WHERE " + " OR ".join(dict.fromkeys(pats)))
    try:
        df = query_fn(adql)
    except Exception as exc:  # noqa: BLE001
        print(f"[tailings] column-based discovery failed: {exc!r}")
        return {}
    if df is None or len(df) == 0 or "column_name" not in getattr(df, "columns", []):
        return {}
    tcol = "table_name" if "table_name" in df.columns else df.columns[0]
    out: dict[str, list[str]] = {}
    for t, c in zip(df[tcol].tolist(), df["column_name"].tolist(), strict=False):
        out.setdefault(unquote_table(t), []).append(str(c))
    return out


def fetch_table_columns(
    tables,
    *,
    url: str = VIZIER_TAP,
    query_fn=None,
    chunk: int = 12,
) -> dict[str, list[str]]:
    """Full column lists for candidate tables, straight from ``TAP_SCHEMA.columns``.

    One query answers what a dozen ``SELECT TOP 1 *`` probes would, and it
    answers for tables that refuse a data query as well. Matched with LIKE
    rather than ``IN`` because VizieR stores the name double-quoted in some
    places and bare in others, and an equality test silently misses half of them
    -- the same class of bug as the FROM-clause quoting failure.
    """
    query_fn = query_fn or (lambda q: tap_query(q, url=url))
    tables = [unquote_table(t) for t in tables if unquote_table(t)]
    out: dict[str, list[str]] = {}
    for i in range(0, len(tables), max(1, int(chunk))):
        block = tables[i: i + max(1, int(chunk))]
        where = " OR ".join(f"table_name LIKE '%{t}%'" for t in block)
        adql = ("SELECT TOP 4000 table_name, column_name FROM TAP_SCHEMA.columns "
                f"WHERE {where}")
        try:
            df = query_fn(adql)
        except Exception as exc:  # noqa: BLE001
            print(f"[tailings] column index for {block}: {exc!r}")
            continue
        if df is None or len(df) == 0 or "column_name" not in getattr(df, "columns", []):
            continue
        tcol = "table_name" if "table_name" in df.columns else df.columns[0]
        for t, c in zip(df[tcol].tolist(), df["column_name"].tolist(), strict=False):
            out.setdefault(unquote_table(t), []).append(str(c))
    return out


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
                 ("rv", 3), ("fiber", 3), ("ra", 1), ("dec", 1)):
        if k in params:
            score += w
    return score, params, elements


def schema_reason(params: dict, elements: dict) -> str:
    """Say *why* a candidate table was rejected, in the terms the channel needs.

    "missing parameters or elements" was the message the third dispatch printed
    34 times, and it did not distinguish "this is a kinematics VAC" from "this
    is the abundance table under a different column convention". Naming what is
    absent is what makes the next dispatch cheaper.
    """
    missing = [k for k in ("teff", "logg", "fe_h") if k not in params]
    bits = []
    if missing:
        bits.append("no " + ", ".join(missing))
    if not elements:
        bits.append("no [X/Fe] columns")
    if not bits:
        extras = [k for k in ("snr", "star_id", "rv", "fiber") if k not in params]
        return ("usable" if not extras
                else "usable, but without " + ", ".join(extras))
    return "rejected: " + "; ".join(bits) + f" ({len(elements)} elements resolved)"


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
    table = unquote_table(table)
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
    verdict: str = VERDICT_OK
    route: str = "tap"
    stage_counts: dict = field(default_factory=dict)
    download_routes: list[dict] = field(default_factory=list)

    def provenance(self) -> dict:
        return {
            "survey": self.survey,
            "verdict": self.verdict,
            "route": self.route,
            "source_used": self.source_used,
            "locator": self.locator,
            "n_rows": int(self.n_rows),
            "n_elements": len(self.elements),
            "elements": self.elements,
            "degraded": bool(self.degraded),
            "degradation": self.degradation,
            "sources_tried": self.sources_tried,
            "scoreboard": self.scoreboard,
            "stage_counts": {k: v for k, v in (self.stage_counts or {}).items()},
            "download_routes": self.download_routes,
        }


COOL_DWARF_ADQL = (
    "{teff} < {teff_max} AND {teff} > {teff_min} AND {logg} > {logg_min} "
    "AND {snr} > {snr_min} AND {feh} > {feh_min}"
)


# ---------------------------------------------------------------------------
# Primary route: survey-native bulk files
# ---------------------------------------------------------------------------
def fetch_survey_file(
    survey: str,
    *,
    selection: Selection | None = None,
    routes=None,
    route_probe_fn=None,
    read_fn=None,
    cache_dir: str | Path | None = None,
    min_bytes: int = 1_000_000,
) -> Acquisition:
    """Pull a survey catalogue from its own bulk file, not from VizieR.

    Ordered candidate URLs are all probed *first* so the report says which host
    and path are live even when the pull succeeds on the third one; then the
    eligible routes are tried in order until one yields rows. Every outcome is a
    distinct verdict: nothing live is ``NO_DATA_REACHED``, a download or read
    that raised is ``QUERY_FAILED``, and a file that read cleanly but had no row
    inside the cool-dwarf box is ``QUERY_RETURNED_ZERO_ROWS`` -- a statement
    about the cuts, not about the archive.
    """
    selection = selection or Selection()
    routes = tuple(routes if routes is not None else FILE_ROUTES.get(survey, ()))
    counts: dict = {"routes_registered": len(routes)}
    if not routes:
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=None, locator=None,
            n_rows=0, elements=[], degraded=True, verdict=VERDICT_NO_DATA, route="file",
            degradation=(f"{VERDICT_NO_DATA}: no bulk-download route is registered for "
                         f"{survey}"),
            stage_counts=counts,
        )

    report = probe_download_routes(routes, probe_fn=route_probe_fn, min_bytes=min_bytes)
    counts["routes_probed"] = len(report)
    counts["routes_live"] = int(sum(1 for r in report if int(r.get("status") or 0) in (200, 206)))
    counts["routes_eligible"] = int(sum(1 for r in report if r.get("eligible")))
    eligible = [r for r in report if r.get("eligible")]
    tried = [r["name"] for r in report]

    if not eligible:
        detail = "; ".join(f"{r['name']} HTTP {r.get('status')}" for r in report)
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=None, locator=None,
            n_rows=0, elements=[], degraded=True, verdict=VERDICT_NO_DATA, route="file",
            degradation=(f"{VERDICT_NO_DATA}: no candidate bulk-download URL for {survey} "
                         f"answered ({detail}). The probe report in provenance.json lists "
                         "every URL and its status, which is the input to fixing this."),
            sources_tried=tried, download_routes=report, stage_counts=counts,
        )

    read_fn = read_fn or (
        lambda url, sel: download_and_read(url, selection=sel, cache_dir=cache_dir))

    errors: list[str] = []
    zero_row_routes: list[str] = []
    counts["rows_in_file"] = 0
    counts["rows_after_selection"] = 0
    for rec in eligible:
        try:
            raw = read_fn(rec["url"], selection)
        except Exception as exc:  # noqa: BLE001 - a dead mirror is not a dead run
            msg = f"{rec['name']}: {exc!r}"
            print(f"[tailings] {survey}: download/read failed for {msg}")
            rec["why"] = f"download or read failed: {exc!r}"
            errors.append(msg)
            continue
        if raw is None:
            rec["why"] = "reader returned nothing"
            errors.append(f"{rec['name']}: reader returned None")
            continue
        attrs = dict(getattr(raw, "attrs", {}) or {})
        n_file = int(attrs.get("n_rows_file", len(raw)))
        df, missing_cuts = apply_selection(raw, selection)
        counts["rows_in_file"] = max(int(counts.get("rows_in_file", 0)), n_file)
        counts["rows_after_selection"] = max(int(counts.get("rows_after_selection", 0)), len(df))
        elements = resolve_abundance_columns(df.columns)
        params = resolve_param_columns(df.columns)
        if len(df) == 0:
            rec["why"] = (f"read cleanly ({n_file} rows in file) but zero rows inside the "
                          "cool-dwarf selection")
            zero_row_routes.append(rec["name"])
            continue
        if not elements:
            xh = resolve_xh_columns(df.columns)
            extra = (f", but it does carry {len(xh)} [X/H] columns ({sorted(xh)[:8]}): "
                     "that is a schema-convention mismatch, not a dead archive"
                     if xh else "")
            rec["why"] = f"read cleanly but carries no [X/Fe] columns{extra}"
            errors.append(f"{rec['name']}: no [X/Fe] columns{extra}")
            continue

        rec["used"] = True
        norm = normalize(df, survey=survey)
        counts["rows_normalised"] = int(len(norm))
        counts["n_elements"] = int(len(elements))
        notes: list[str] = []
        if rec is not report[0]:
            notes.append(f"used the fallback route {rec['name']} ({rec['url']}) rather than "
                         f"the preferred {report[0]['name']}")
        cuts_missing = list(dict.fromkeys(list(missing_cuts) + list(attrs.get("cuts_not_applied", []))))
        if cuts_missing:
            notes.append("the file has no " + ", ".join(cuts_missing)
                         + " column, so that cut was NOT applied and the sample is broader "
                           "than the stated selection")
        for key in ("rv", "fiber"):
            if key not in params:
                notes.append(f"no {key} column: the instrumental-covariate veto on {key} "
                             "cannot run on this table")
        if selection.max_rows and len(norm) >= int(selection.max_rows):
            notes.append(f"hit the {selection.max_rows}-row cap: the sample is TRUNCATED")
        print(f"[tailings] {survey}: {len(norm)} rows from {rec['name']} "
              f"({len(elements)} elements)")
        return Acquisition(
            survey=survey, table=norm, source_used=rec["name"], locator=rec["url"],
            n_rows=len(norm), elements=sorted(elements), param_columns=params,
            degraded=bool(notes), degradation="; ".join(notes), verdict=VERDICT_OK,
            route="file", sources_tried=tried, download_routes=report, stage_counts=counts,
        )

    if zero_row_routes:
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=zero_row_routes[0], locator=None,
            n_rows=0, elements=[], degraded=True, verdict=VERDICT_ZERO_ROWS, route="file",
            degradation=(f"{VERDICT_ZERO_ROWS}: {', '.join(zero_row_routes)} downloaded and "
                         "parsed correctly but no row survived the cool-dwarf selection "
                         f"{selection.describe()}. This is a statement about the cuts, not "
                         "about the archive."),
            sources_tried=tried, download_routes=report, stage_counts=counts,
        )
    return Acquisition(
        survey=survey, table=pd.DataFrame(), source_used=None, locator=None,
        n_rows=0, elements=[], degraded=True, verdict=VERDICT_QUERY_FAILED, route="file",
        degradation=(f"{VERDICT_QUERY_FAILED}: every live bulk-download route for {survey} "
                     f"errored during download or read ({'; '.join(errors)})"),
        sources_tried=tried, download_routes=report, stage_counts=counts,
    )


# ---------------------------------------------------------------------------
# Fallback route: VizieR TAP
# ---------------------------------------------------------------------------
def fetch_survey_tap(
    survey: str,
    *,
    teff_max: float = 6000.0,
    teff_min: float = 3000.0,
    logg_min: float = 4.0,
    snr_min: float = 40.0,
    feh_min: float = -1.0,
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
    only one of them is the main abundance catalogue. Candidates come from three
    places now, because keyword discovery alone was what failed on the third
    dispatch: the encoded locators, a keyword search of ``TAP_SCHEMA.tables``,
    and a search of ``TAP_SCHEMA.columns`` for tables that actually carry
    ``[X/Fe]`` columns. Each candidate's full column list is pulled from
    ``TAP_SCHEMA.columns`` where possible (one query for a dozen tables) and
    only then probed, and every rejection is recorded with its reason.

    ``probe_fn`` and ``query_fn`` exist so the offline suite can drive the full
    selection and degradation logic without a network.
    """
    probe_fn = probe_fn or (lambda t: probe_table(t, url=tap_url))
    query_fn = query_fn or (lambda q: tap_query(q, url=tap_url))

    encoded = [unquote_table(s.locator) for s in SOURCES.get(survey, ()) if s.kind == "tap"]
    names = {unquote_table(s.locator): s.name for s in SOURCES.get(survey, ())}
    candidates = list(encoded)
    by_column: dict[str, list[str]] = {}
    if discover:
        found = discover_tables(DISCOVERY_KEYWORDS.get(survey, (survey,)),
                                url=tap_url, query_fn=query_fn)
        by_column = discover_tables_by_columns(url=tap_url, query_fn=query_fn)
        kw = tuple(k.lower() for k in DISCOVERY_KEYWORDS.get(survey, (survey,)))
        # A table that carries [X/Fe] columns *and* names the survey is the
        # thing being looked for; put those ahead of everything discovered by
        # description alone, which is what returned 34 empty stubs last time.
        strong = [t for t in by_column if any(k in t.lower() for k in kw)]
        for t in strong + found + list(by_column):
            if t not in candidates:
                candidates.append(t)
        print(f"[tailings] {survey}: {len(encoded)} encoded + {len(found)} by keyword + "
              f"{len(by_column)} by abundance column -> {len(candidates)} candidates "
              f"({len(strong)} carry both the survey name and [X/Fe] columns)")

    short = candidates[:max_candidates]
    column_index: dict[str, list[str]] = {}
    if discover and short:
        column_index = fetch_table_columns(short, url=tap_url, query_fn=query_fn)

    counts: dict = {
        "candidate_tables": len(candidates),
        "candidates_examined": len(short),
        "tables_in_column_index": len(column_index),
    }

    tried: list[str] = []
    scoreboard: list[dict] = []
    best = None
    n_responded = 0
    for locator in short:
        tried.append(names.get(locator, locator))
        cols = column_index.get(locator)
        origin = "TAP_SCHEMA.columns"
        if not cols:
            head = probe_fn(locator)
            origin = "SELECT TOP 1 *"
            if head is None or len(head.columns) == 0:
                scoreboard.append({"table": locator, "score": 0, "n_elements": 0,
                                   "why": "no response", "schema_from": origin})
                continue
            cols = list(head.columns)
        n_responded += 1
        score, params, elements = score_schema(cols)
        scoreboard.append({
            "table": locator,
            "score": int(score),
            "n_elements": len(elements),
            "n_columns": len(cols),
            "schema_from": origin,
            "elements": sorted(elements)[:40],
            "why": schema_reason(params, elements),
        })
        if score and (best is None or score > best[0]):
            best = (score, locator, params, elements)
    counts["tables_with_schema"] = n_responded
    counts["tables_usable"] = int(sum(1 for r in scoreboard if r["score"] > 0))

    if best is None:
        top = sorted(scoreboard, key=lambda r: -int(r.get("n_elements", 0)))[:5]
        detail = "; ".join(f"{r['table']} -> {r['why']}" for r in top)
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=None, locator=None,
            n_rows=0, elements=[], degraded=True, verdict=VERDICT_NO_DATA, route="tap",
            degradation=(f"{VERDICT_NO_DATA}: no candidate table for this survey had a "
                         f"usable schema. Nearest misses: {detail}" if detail else
                         f"{VERDICT_NO_DATA}: no candidate table for this survey had a "
                         "usable schema"),
            sources_tried=tried, scoreboard=scoreboard, stage_counts=counts,
        )

    score, locator, params, elements = best
    label = names.get(locator, locator)
    print(f"[tailings] {survey}: using {label} ({locator}), score {score}, "
          f"{len(elements)} elements")

    cols: list[str] = []
    for k in ("star_id", "ra", "dec", "teff", "logg", "fe_h", "snr", "chi2",
              "ruwe", "vbroad", "rv_scatter", "rv", "fiber", "field_id"):
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
    n_failed = 0
    for lo, hi in bands:
        where = COOL_DWARF_ADQL.format(
            teff=f'"{params["teff"]}"',
            teff_max=hi,
            teff_min=lo,
            logg=f'"{params["logg"]}"',
            logg_min=logg_min,
            snr=f'"{params["snr"]}"' if "snr" in params else "1e9",
            snr_min=snr_min,
            feh=f'"{params["fe_h"]}"',
            feh_min=feh_min,
        )
        adql = (f"SELECT TOP {per_chunk} " + select
                + f' FROM "{locator}" WHERE ' + where)
        try:
            chunk = query_fn(adql)
        except Exception as exc:  # noqa: BLE001 - a lost chunk is not a lost run
            print(f"[tailings] {survey}/{label}: chunk {lo:.0f}-{hi:.0f} K failed: {exc!r}")
            n_failed += 1
            continue
        if chunk is not None and len(chunk):
            frames.append(chunk)
            print(f"[tailings] {survey}/{label}: {len(chunk)} rows in {lo:.0f}-{hi:.0f} K")

    counts.update({"chunks_requested": len(bands), "chunks_failed": int(n_failed),
                   "chunks_with_rows": len(frames),
                   "rows_returned": int(sum(len(f) for f in frames))})

    if not frames:
        # The two failures are not the same statement and must not share a
        # verdict: every chunk raising is a broken query, every chunk answering
        # with nothing is an empty selection.
        if n_failed >= len(bands):
            return Acquisition(
                survey=survey, table=pd.DataFrame(), source_used=label, locator=locator,
                n_rows=0, elements=sorted(elements), param_columns=params, degraded=True,
                verdict=VERDICT_QUERY_FAILED, route="tap",
                degradation=(f"{VERDICT_QUERY_FAILED}: all {len(bands)} Teff chunks raised "
                             f"against {label} ({locator}); the table exists and has a usable "
                             "schema, so this is a query or service failure, not an empty sky"),
                sources_tried=tried, scoreboard=scoreboard, stage_counts=counts,
            )
        return Acquisition(
            survey=survey, table=pd.DataFrame(), source_used=label, locator=locator,
            n_rows=0, elements=sorted(elements), param_columns=params, degraded=True,
            verdict=VERDICT_ZERO_ROWS, route="tap",
            degradation=(f"{VERDICT_ZERO_ROWS}: {label} ({locator}) has a usable schema and "
                         f"answered {len(bands) - n_failed}/{len(bands)} chunks, but returned "
                         "zero rows under the cool-dwarf selection. This is a statement about "
                         "the cuts, not about the archive."),
            sources_tried=tried, scoreboard=scoreboard, stage_counts=counts,
        )

    df = pd.concat(frames, ignore_index=True)
    truncated = [1 for f in frames if len(f) >= per_chunk]
    norm = normalize(df, survey=survey)
    counts["rows_normalised"] = int(len(norm))
    counts["n_elements"] = int(len(elements))

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
    for key in ("rv", "fiber"):
        if key not in params:
            notes.append(f"no {key} column on this table: the instrumental-covariate veto "
                         f"on {key} cannot run, which is a known weakness of the abbreviated "
                         "VizieR copies")
    return Acquisition(
        survey=survey, table=norm, source_used=label, locator=locator,
        n_rows=len(norm), elements=sorted(resolve_abundance_columns(df.columns)),
        param_columns=params, degraded=bool(notes), degradation="; ".join(notes),
        verdict=VERDICT_OK, route="tap",
        sources_tried=tried, scoreboard=scoreboard, stage_counts=counts,
    )


def fetch_survey(
    survey: str,
    *,
    teff_max: float = 6000.0,
    teff_min: float = 3000.0,
    logg_min: float = 4.0,
    snr_min: float = 40.0,
    feh_min: float = -1.0,
    max_rows: int = 400_000,
    n_chunks: int = 8,
    tap_url: str = VIZIER_TAP,
    discover: bool = True,
    max_candidates: int = 12,
    probe_fn=None,
    query_fn=None,
    use_files: bool | None = None,
    routes=None,
    route_probe_fn=None,
    read_fn=None,
    cache_dir: str | Path | None = None,
) -> Acquisition:
    """Acquire one survey: bulk file first, VizieR TAP as the fallback.

    The order is the whole change. VizieR's abbreviated tables were unreachable
    on three dispatches *and* would not have been sufficient: correction #5
    needs radial velocity, fibre ID and detector position to de-trend the flag
    rate, and those columns exist in the survey's own FITS release and not in
    the trimmed CDS copy. So the file route is primary and TAP is what runs when
    every URL is dead.

    ``use_files`` defaults to "on, unless the caller injected a TAP transport
    and no file reader" -- an injected ``probe_fn``/``query_fn`` means the
    caller is deliberately driving the TAP path (the offline suite does exactly
    that), and reaching for a socket underneath it would be wrong.
    """
    if use_files is None:
        use_files = read_fn is not None or (probe_fn is None and query_fn is None)

    file_acq: Acquisition | None = None
    if use_files:
        selection = Selection(teff_min=teff_min, teff_max=teff_max, logg_min=logg_min,
                              snr_min=snr_min, feh_min=feh_min, max_rows=max_rows)
        file_acq = fetch_survey_file(
            survey, selection=selection, routes=routes, route_probe_fn=route_probe_fn,
            read_fn=read_fn, cache_dir=cache_dir,
        )
        if file_acq.n_rows:
            return file_acq
        print(f"[tailings] {survey}: bulk-file route gave {file_acq.verdict}; "
              "falling back to VizieR TAP")

    tap_acq = fetch_survey_tap(
        survey, teff_max=teff_max, teff_min=teff_min, logg_min=logg_min,
        snr_min=snr_min, feh_min=feh_min, max_rows=max_rows, n_chunks=n_chunks,
        tap_url=tap_url, discover=discover, max_candidates=max_candidates,
        probe_fn=probe_fn, query_fn=query_fn,
    )
    if file_acq is None:
        return tap_acq

    # Both routes were attempted: carry the file evidence into the TAP result so
    # the report says what happened on *both*, and keep the more specific
    # verdict when the fallback also came back empty.
    tap_acq.download_routes = file_acq.download_routes
    tap_acq.sources_tried = list(file_acq.sources_tried) + list(tap_acq.sources_tried)
    tap_acq.stage_counts = {**file_acq.stage_counts, **tap_acq.stage_counts}
    tap_acq.degraded = True
    prefix = f"bulk-file route: {file_acq.verdict} ({file_acq.degradation})"
    if tap_acq.n_rows:
        tap_acq.degradation = "; ".join(x for x in (prefix, tap_acq.degradation) if x)
    else:
        if tap_acq.verdict == VERDICT_NO_DATA and file_acq.verdict != VERDICT_NO_DATA:
            tap_acq.verdict = file_acq.verdict
            tap_acq.degradation = (f"{file_acq.verdict}: {file_acq.degradation} | VizieR TAP "
                                   f"fallback: {tap_acq.degradation}")
        else:
            tap_acq.degradation = f"{tap_acq.degradation} | {prefix}"
    return tap_acq


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
    for c in ("teff", "logg", "fe_h", "snr", "chi2", "ruwe", "vbroad",
              "rv_scatter", "rv", "fiber"):
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


# ---------------------------------------------------------------------------
# Wide binaries
# ---------------------------------------------------------------------------
_PAIR_ID1 = ("source_id1", "gaiaedr3_1", "source1", "sourceid1", "source_id_a", "id1")
_PAIR_ID2 = ("source_id2", "gaiaedr3_2", "source2", "sourceid2", "source_id_b", "id2")
_PAIR_RCA = ("r_chance_align", "rchancealign", "rchance", "rchancealign1")


def _resolve_pair_columns(columns) -> tuple[str | None, str | None, str | None]:
    cols = {_canon(c): str(c) for c in columns}
    id1 = next((v for k, v in cols.items() if k in _PAIR_ID1), None)
    id2 = next((v for k, v in cols.items() if k in _PAIR_ID2), None)
    rca = next((v for k, v in cols.items() if k in _PAIR_RCA), None)
    return id1, id2, rca


def _pair_column_picker(colnames) -> list[str]:
    return [c for c in _resolve_pair_columns(colnames) if c]


def fetch_wide_binaries_file(
    *,
    max_rows: int = 200_000,
    max_r_chance_align: float = 0.1,
    routes=None,
    route_probe_fn=None,
    read_fn=None,
    cache_dir: str | Path | None = None,
    min_bytes: int = 100_000,
) -> Acquisition:
    """El-Badry+2021 pairs from the published file rather than the VizieR copy."""
    routes = tuple(routes if routes is not None else FILE_ROUTES.get("WIDEBINARY", ()))
    counts: dict = {"routes_registered": len(routes)}
    if not routes:
        return Acquisition(survey="WIDEBINARY", table=pd.DataFrame(), source_used=None,
                           locator=None, n_rows=0, elements=[], degraded=True,
                           verdict=VERDICT_NO_DATA, route="file", stage_counts=counts,
                           degradation=f"{VERDICT_NO_DATA}: no wide-binary file route registered")
    # These files carry no element panel by construction, so the "expects
    # abundances" gate would reject every one of them; eligibility here is
    # simply "it answered".
    report = probe_download_routes(
        [DownloadRoute(r.name, r.url, r.note, abundances=True) for r in routes],
        probe_fn=route_probe_fn, min_bytes=min_bytes)
    counts["routes_probed"] = len(report)
    counts["routes_eligible"] = int(sum(1 for r in report if r.get("eligible")))
    tried = [r["name"] for r in report]
    eligible = [r for r in report if r.get("eligible")]
    if not eligible:
        return Acquisition(survey="WIDEBINARY", table=pd.DataFrame(), source_used=None,
                           locator=None, n_rows=0, elements=[], degraded=True,
                           verdict=VERDICT_NO_DATA, route="file", sources_tried=tried,
                           download_routes=report, stage_counts=counts,
                           degradation=(f"{VERDICT_NO_DATA}: no wide-binary download URL "
                                        "answered"))

    read_fn = read_fn or (
        lambda url, sel: download_and_read(url, selection=None, cache_dir=cache_dir,
                                           column_picker=_pair_column_picker))
    errors: list[str] = []
    zero: list[str] = []
    for rec in eligible:
        try:
            df = read_fn(rec["url"], None)
        except Exception as exc:  # noqa: BLE001
            rec["why"] = f"download or read failed: {exc!r}"
            errors.append(f"{rec['name']}: {exc!r}")
            continue
        if df is None or len(df) == 0:
            rec["why"] = "read cleanly but empty"
            zero.append(rec["name"])
            continue
        id1, id2, rca = _resolve_pair_columns(df.columns)
        if not (id1 and id2):
            rec["why"] = "no pair of Gaia source identifiers"
            errors.append(f"{rec['name']}: no identifier pair")
            continue
        counts["pairs_in_file"] = int(len(df))
        notes: list[str] = []
        if rca:
            df = df[pd.to_numeric(df[rca], errors="coerce") < float(max_r_chance_align)]
        else:
            notes.append("catalogue has no R_chance_align column: chance alignments "
                         "are NOT removed and every pair verdict is provisional")
        counts["pairs_after_purity_cut"] = int(len(df))
        if len(df) == 0:
            rec["why"] = "no pair passed the R_chance_align purity cut"
            zero.append(rec["name"])
            continue
        if len(df) > int(max_rows):
            df = df.iloc[: int(max_rows)]
            notes.append(f"hit the {max_rows}-row cap: the pair list is TRUNCATED")
        rec["used"] = True
        out = df.rename(columns={id1: "source_id_a", id2: "source_id_b",
                                 **({rca: "r_chance_align"} if rca else {})})
        counts["pairs_kept"] = int(len(out))
        return Acquisition(survey="WIDEBINARY", table=out.reset_index(drop=True),
                           source_used=rec["name"], locator=rec["url"], n_rows=len(out),
                           elements=[], degraded=bool(notes), degradation="; ".join(notes),
                           verdict=VERDICT_OK, route="file", sources_tried=tried,
                           download_routes=report, stage_counts=counts)
    if zero:
        return Acquisition(survey="WIDEBINARY", table=pd.DataFrame(), source_used=zero[0],
                           locator=None, n_rows=0, elements=[], degraded=True,
                           verdict=VERDICT_ZERO_ROWS, route="file", sources_tried=tried,
                           download_routes=report, stage_counts=counts,
                           degradation=(f"{VERDICT_ZERO_ROWS}: {', '.join(zero)} parsed "
                                        "correctly but yielded no pair after the purity cut"))
    return Acquisition(survey="WIDEBINARY", table=pd.DataFrame(), source_used=None,
                       locator=None, n_rows=0, elements=[], degraded=True,
                       verdict=VERDICT_QUERY_FAILED, route="file", sources_tried=tried,
                       download_routes=report, stage_counts=counts,
                       degradation=(f"{VERDICT_QUERY_FAILED}: every live wide-binary route "
                                    f"errored ({'; '.join(errors)})"))


def fetch_wide_binaries(
    *,
    max_rows: int = 200_000,
    max_r_chance_align: float = 0.1,
    tap_url: str = VIZIER_TAP,
    discover: bool = True,
    max_candidates: int = 12,
    probe_fn=None,
    query_fn=None,
    use_files: bool | None = None,
    routes=None,
    route_probe_fn=None,
    read_fn=None,
    cache_dir: str | Path | None = None,
) -> Acquisition:
    """Gaia wide binaries, cut on the catalogue's own chance-alignment estimate.

    ``R_chance_align`` is the probability that a pair is a projection rather
    than a bound system. Keeping it below 0.1 is the standard purity cut and it
    matters here more than usual: a chance alignment of two unrelated stars has
    no reason to share a composition, and would manufacture exactly the
    differential anomaly stage 4 looks for. Where the catalogue does not carry
    that column the pull still proceeds, but the omission is recorded as
    degradation rather than passed over.

    Same route order as :func:`fetch_survey`: the published file first, the
    VizieR copy (``J/MNRAS/506/2269/table1``, which did not resolve on any of
    the three dispatches) as the fallback.
    """
    if use_files is None:
        use_files = read_fn is not None or (probe_fn is None and query_fn is None)

    file_acq: Acquisition | None = None
    if use_files:
        file_acq = fetch_wide_binaries_file(
            max_rows=max_rows, max_r_chance_align=max_r_chance_align, routes=routes,
            route_probe_fn=route_probe_fn, read_fn=read_fn, cache_dir=cache_dir)
        if file_acq.n_rows:
            return file_acq

    probe_fn = probe_fn or (lambda t: probe_table(t, url=tap_url))
    query_fn = query_fn or (lambda q: tap_query(q, url=tap_url))

    encoded = [unquote_table(s.locator) for s in SOURCES["WIDEBINARY"]]
    names = {unquote_table(s.locator): s.name for s in SOURCES["WIDEBINARY"]}
    candidates = list(encoded)
    if discover:
        found = discover_tables(DISCOVERY_KEYWORDS["WIDEBINARY"], url=tap_url,
                                query_fn=query_fn)
        candidates += [t for t in found if t not in candidates]

    tried: list[str] = []
    scoreboard: list[dict] = []
    counts: dict = {"candidate_tables": len(candidates)}
    n_failed = 0
    n_zero = 0
    for locator in candidates[:max_candidates]:
        label = names.get(locator, locator)
        tried.append(label)
        head = probe_fn(locator)
        if head is None or len(head.columns) == 0:
            scoreboard.append({"table": locator, "score": 0, "why": "no response"})
            continue
        id1, id2, rca = _resolve_pair_columns(head.columns)
        if not (id1 and id2):
            scoreboard.append({"table": locator, "score": 0,
                               "why": "no pair of Gaia source identifiers"})
            continue
        scoreboard.append({"table": locator, "score": 1 + (1 if rca else 0),
                           "why": "usable" if rca else
                                  "usable, but with no R_chance_align purity column"})
        sel = f'"{id1}", "{id2}"' + (f', "{rca}"' if rca else "")
        where = f' WHERE "{rca}" < {max_r_chance_align}' if rca else ""
        try:
            df = query_fn(f"SELECT TOP {int(max_rows)} {sel} FROM \"{locator}\"{where}")
        except Exception as exc:  # noqa: BLE001
            print(f"[tailings] wide binaries {locator}: query failed: {exc!r}")
            n_failed += 1
            continue
        if df is None or len(df) == 0:
            n_zero += 1
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
        counts["pairs_kept"] = int(len(df))
        out = Acquisition(survey="WIDEBINARY", table=df, source_used=label,
                          locator=locator, n_rows=len(df), elements=[],
                          degraded=bool(notes), degradation="; ".join(notes),
                          verdict=VERDICT_OK, route="tap",
                          sources_tried=tried, scoreboard=scoreboard, stage_counts=counts)
        if file_acq is not None:
            out.download_routes = file_acq.download_routes
            out.degraded = True
            out.degradation = "; ".join(x for x in (
                f"bulk-file route: {file_acq.verdict} ({file_acq.degradation})",
                out.degradation) if x)
        return out

    counts.update({"tap_queries_failed": n_failed, "tap_queries_zero_rows": n_zero})
    if n_failed:
        verdict, why = VERDICT_QUERY_FAILED, (
            f"{VERDICT_QUERY_FAILED}: {n_failed} wide-binary table(s) had a usable schema "
            "and then errored on the pair query")
    elif n_zero:
        verdict, why = VERDICT_ZERO_ROWS, (
            f"{VERDICT_ZERO_ROWS}: a wide-binary table answered but no pair survived the "
            f"R_chance_align < {max_r_chance_align} purity cut")
    else:
        verdict, why = VERDICT_NO_DATA, (
            f"{VERDICT_NO_DATA}: wide-binary catalogue unreachable")
    out = Acquisition(survey="WIDEBINARY", table=pd.DataFrame(), source_used=None,
                      locator=None, n_rows=0, elements=[], degraded=True,
                      verdict=verdict, route="tap", degradation=why,
                      sources_tried=tried, scoreboard=scoreboard, stage_counts=counts)
    if file_acq is not None:
        out.download_routes = file_acq.download_routes
        out.stage_counts = {**file_acq.stage_counts, **out.stage_counts}
        out.sources_tried = list(file_acq.sources_tried) + list(out.sources_tried)
        if verdict == VERDICT_NO_DATA and file_acq.verdict != VERDICT_NO_DATA:
            out.verdict = file_acq.verdict
        out.degradation = f"{why} | bulk-file route: {file_acq.verdict} " \
                          f"({file_acq.degradation})"
    return out


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


def to_native_byteorder(df: pd.DataFrame) -> pd.DataFrame:
    """Force every column to the machine's native byte order.

    FITS stores numbers big-endian, so ``astropy`` hands back dtypes like
    ``>i8`` / ``>f8`` on a little-endian runner.  pyarrow refuses those outright
    -- ``ArrowNotImplementedError: ('Byte-swapped arrays not supported',
    'Conversion failed for column star_id with type >i8')`` -- so the first
    ``to_parquet`` after a successful bulk-file download dies, *after* the
    hundreds of megabytes have been fetched and parsed.  That is exactly the
    expensive place to fail, so the conversion is done centrally rather than
    trusted to each reader.

    ``newbyteorder`` only relabels the dtype; ``astype`` does the actual swap.
    """
    if df is None or df.empty:
        return df
    fixed = {}
    for col in df.columns:
        s = df[col]
        dt = getattr(s, "dtype", None)
        # '>' is big-endian; '=' and '|' are already native/not-applicable.
        if getattr(dt, "byteorder", "=") == ">":
            fixed[col] = s.astype(dt.newbyteorder("="))
    if fixed:
        df = df.copy()
        for col, s in fixed.items():
            df[col] = s
    return df


def write_checkpoint(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a parquet checkpoint, creating parents. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    to_native_byteorder(df).to_parquet(p, index=False)
    return p


__all__ = [
    "Acquisition",
    "DISCOVERY_ELEMENTS",
    "DISCOVERY_KEYWORDS",
    "DownloadRoute",
    "FILE_ROUTES",
    "SOURCES",
    "to_native_byteorder",
    "Selection",
    "Source",
    "TARGET_ELEMENTS",
    "VERDICT_NO_DATA",
    "VERDICT_OK",
    "VERDICT_QUERY_FAILED",
    "VERDICT_ZERO_ROWS",
    "VIZIER_TAP",
    "apply_element_flags",
    "apply_selection",
    "discover_tables",
    "discover_tables_by_columns",
    "download_and_read",
    "fetch_survey",
    "fetch_survey_file",
    "fetch_survey_tap",
    "fetch_table_columns",
    "fetch_wide_binaries",
    "fetch_wide_binaries_file",
    "join_pairs",
    "normalize",
    "probe_download_routes",
    "probe_table",
    "read_fits_table",
    "resolve_abundance_columns",
    "resolve_param_columns",
    "resolve_xh_columns",
    "schema_reason",
    "score_schema",
    "selection_mask",
    "stream_download",
    "tap_query",
    "teff_bands",
    "unquote_table",
    "write_checkpoint",
]
