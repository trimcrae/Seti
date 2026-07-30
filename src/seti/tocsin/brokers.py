"""Broker adapters --- runner-only network code, one normalised output.

Every function here touches the network and therefore **cannot run in the
sandbox** (``docs/channel-brief.md`` §0: egress to the brokers is 403-blocked);
they run on the GitHub Actions runner.  Everything they return is a
:class:`~seti.tocsin.schema.NormalizedAlert`, so the funnel never sees a broker's
column names.

Which broker, and why
---------------------
Seven community brokers carry the stream; only one of them can support an
*unattended nightly cron with no human in the loop*, and it is not the obvious
one:

* **ALeRCE TAP (primary).**  ``https://tap.alerce.online/tap`` is a public IVOA
  TAP service with full ADQL over the LSST per-epoch tables, **no account and no
  token**, indexed on ``mjd``/``ra``/``dec``.  One ADQL statement answers "every
  detection from night X", which is exactly the shape a nightly screen needs, and
  ``pyvo`` is already a dependency of this repository.  Crucially it also serves
  the **forced-photometry** table, which is the ledger's denominator, in bulk ---
  so the visit history costs one query rather than one REST call per object.
* **Lasair-LSST (secondary, token).**  Free self-service registration, but a
  token is required for every endpoint, per-epoch fields are reachable only one
  object at a time, and the registered tier allows 100 calls/hour.  Good for
  deep per-object vetting of a short list; unusable as the bulk path.
* **Fink (enrichment).**  No auth, but no whole-night endpoint at all, a hard
  10,000-row cap with *no pagination*, and its bulk path (Data Transfer) requires
  a human to submit a web form --- a hard blocker for unattended operation.  What
  Fink uniquely has is a far richer cross-match layer (SIMBAD, **VSX**, **GCVS**,
  Gaia DR3 variability flags, TNS, Legacy photo-z), which is high-value
  contamination rejection for a stellar sample: it is the cheapest way to ask
  "is this star already a catalogued variable?".

So the channel reads bulk from ALeRCE, enriches a shortlist from Fink, and uses
Lasair only when a token happens to be configured.

Lasair-LSST: the two-stage architecture that API forces
-------------------------------------------------------
Lasair stores per-object *aggregates* in MySQL and per-*epoch* rows in Cassandra.
Only the MySQL side is reachable from the free-form ``/api/query/`` endpoint, and
every field this channel needs for vetting --- ``snr``, ``reliability``,
``isDipole``, ``isNegative``, ``glint_trail``, ``pixelFlags_*`` --- lives on the
Cassandra side, reachable only one object at a time via ``/api/object/`` with
``lite=False``.  That is not a limitation to work around, it is the natural shape
of the funnel:

* **Stage A (cheap, bulk).**  One paged SQL query over ``objects`` selects
  everything that changed in the requested night window and sits within an
  arcsecond of a catalogued *star*.  Local cross-match against the Gaia
  nearby-star list reduces that to a shortlist.
* **Stage B (expensive, per-object).**  ``/api/object/?lite=False`` pulls the
  full per-epoch ``diaSources`` for each shortlisted object --- the vetting
  fields --- **and** ``diaForcedSources``, which is the forced photometry at the
  position on every overlapping visit.  That second list is the ledger's real
  denominator: it says how many times the star was looked at, including the
  nights it showed nothing.

Rate limits are per account and hourly (registered tier: 100 calls/h, 10k rows
per query; power-user on request: 10,000 calls/h, 10^6 rows).  Stage A costs a
handful of calls; Stage B costs one per shortlisted object, which is why the
shortlist is cut hard before it.

Known-bad behaviour compensated for here
----------------------------------------
* the client's ``object(lite=True)`` default returns only 6 columns --- this code
  always passes ``lite=False``;
* Lasair's server-side ``reliabilityThreshold`` is a no-op (a misplaced
  parenthesis in ``lightcurves.py`` makes its ``isnan`` guard always true), so
  reliability is filtered **client-side** here;
* ``limit`` is silently clamped to the account tier's cap, so paging checks the
  returned row count against the requested one instead of trusting it;
* the DB column is ``decl``, not ``dec``; times are MJD **TAI**; fluxes are nJy
  and signed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .schema import NormalizedAlert

LASAIR_ENDPOINT = "https://api.lasair.lsst.ac.uk/api"
LASAIR_KAFKA = "lasair-lsst-kafka.lsst.ac.uk:9092"
ALERCE_TAP = "https://tap.alerce.online/tap"
FINK_LSST_API = "https://api.lsst.fink-portal.org"

# ALeRCE encodes the LSST filter as an integer.  Note u = 6, NOT 0 --- getting
# this wrong silently relabels every u-band detection as g and would corrupt
# every colour measurement in the channel.
ALERCE_BAND = {6: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}

# ALeRCE `sid`.  Documented as the source table (1 = diaObject, 2 = ssObject),
# which if true means selecting sid = 1 excludes alerts already associated with a
# known minor planet (~400 per visit, up to ~5,000 near the ecliptic).
#
# TREAT THIS AS UNVERIFIED.  `alerce_tap.object` carries BOTH `sid` and `tid`,
# which is exactly what a survey-id/telescope-id pair looks like; if `sid` is
# really a survey id then filtering `sid = 1` selects ZTF and the inner join to
# `lsst_detection` returns precisely nothing --- a clean, plausible, wrong null.
# `AlerceTAP.diagnostics()` measures the true values on the runner, and
# `sid_diaobject` is a parameter everywhere rather than a constant, so the answer
# can be applied without touching the query code.
ALERCE_SID_DIAOBJECT = 1
ALERCE_SID_SSOBJECT = 2

# Columns pulled in the bulk `objects` pass.  `decl` is the Lasair spelling.
# `latestPairColour*` are Lasair's own 33-minute in-night pair colour and its
# extinction-corrected blackbody temperature (kK) --- a free first cut on
# achromaticity before any per-epoch call is spent.
_STAGE_A_SELECT = ", ".join([
    "objects.diaObjectId", "objects.ra", "objects.decl",
    "objects.firstDiaSourceMjdTai", "objects.lastDiaSourceMjdTai",
    "objects.nDiaSources", "objects.nPosDiaSources", "objects.nPosDiaSourcesNights",
    "objects.nSourcesGood", "objects.medianR", "objects.latestR",
    "objects.latest_psfFlux", "objects.jump1", "objects.jump2",
    "objects.latestPairMJD", "objects.latestPairColourMag",
    "objects.latestPairColourBands", "objects.latestPairColourTemp",
    "objects.glat", "objects.ebv",
    "sherlock_classifications.classification",
    "sherlock_classifications.catalogue_object_type",
    "sherlock_classifications.separationArcsec",
] + [f"objects.{b}_psfFlux" for b in "ugrizy"]
  + [f"objects.{b}_latestMJD" for b in "ugrizy"])


class BrokerError(RuntimeError):
    """Raised when a broker is unreachable or answers unusably."""


@dataclass
class BrokerResult:
    """What one broker pass returned, with its degradation stated explicitly."""

    rows: list[dict] = field(default_factory=list)
    alerts: list[NormalizedAlert] = field(default_factory=list)
    forced_mjds: dict[str, list[float]] = field(default_factory=dict)
    calls: int = 0
    reached: bool = False
    verdict: str = "NOT_RUN"
    notes: list[str] = field(default_factory=list)


class AlerceTAP:
    """Public IVOA TAP access to ALeRCE's LSST tables.  No credentials.

    Column names are **lower-cased** by ADQL, so ``psfFlux`` comes back as
    ``psfflux`` and ``midpointMjdTai`` is renamed ``mjd``.  Rather than hard-code
    a schema this class can be asked to *report* the live schema
    (:meth:`describe`), which is what the probe workflow does first: the exact
    column set of a live broker in July 2026 cannot be verified from inside the
    sandbox, so the channel discovers it on the runner and records it, instead of
    assuming and failing silently months later.
    """

    def __init__(self, url: str = ALERCE_TAP, timeout: float = 900.0,
                 maxrec: int = 2_000_000):
        self.url = url
        self.timeout = float(timeout)
        self.maxrec = int(maxrec)
        self.calls = 0
        self._svc = None

    def _service(self):
        if self._svc is None:
            try:
                import pyvo
            except ImportError as exc:  # pragma: no cover - pyvo is a dependency
                raise BrokerError(f"pyvo unavailable: {exc}") from exc
            self._svc = pyvo.dal.TAPService(self.url)
            try:
                self._svc._session.timeout = self.timeout
            except Exception:
                pass
        return self._svc

    def query(self, adql: str, maxrec: int | None = None, retries: int = 4):
        """Run one ADQL query, returning a list of row dicts.

        Retries with exponential backoff on transport errors, because a nightly
        cron that dies on one flaky TAP call loses a night of coverage that can
        never be recovered --- the sky has moved on.
        """
        svc = self._service()
        last = None
        for attempt in range(retries):
            self.calls += 1
            try:
                res = svc.search(adql, maxrec=maxrec or self.maxrec)
            except Exception as exc:
                last = str(exc)
                # A malformed query will fail identically every time; retrying
                # it wastes the run's budget, so surface ADQL errors at once.
                if any(k in last.lower() for k in ("syntax", "unknown column",
                                                   "unknown table", "not found")):
                    raise BrokerError(f"ADQL rejected: {last[:600]}\nquery: {adql}") from exc
                time.sleep(2.0 ** attempt)
                continue
            try:
                tab = res.to_table()
            except Exception as exc:
                raise BrokerError(f"TAP result unreadable: {exc}") from exc
            return [{c: row[c] for c in tab.colnames} for row in tab]
        raise BrokerError(f"TAP query failed after {retries} attempts: {last}")

    # -- schema discovery --------------------------------------------------
    def describe(self, tables=("detection", "lsst_detection", "forced_photometry",
                              "lsst_forced_photometry", "object", "lsst_dia_object",
                              "xmatch", "gaiadr3_source")) -> dict:
        """Live column list per table, from ``TAP_SCHEMA``.

        The probe commits this verbatim so that a broker schema change shows up
        as a diff in version control rather than as an unexplained null.
        """
        names = ", ".join(f"'alerce_tap.{t}'" for t in tables)
        rows = self.query(
            "SELECT table_name, column_name, datatype FROM TAP_SCHEMA.columns "
            f"WHERE table_name IN ({names}) ORDER BY table_name, column_name",
            maxrec=20000)
        out: dict[str, list[str]] = {}
        for r in rows:
            t = str(r.get("table_name", ""))
            out.setdefault(t, []).append(str(r.get("column_name", "")))
        return out

    # -- diagnostics -------------------------------------------------------
    def diagnostics(self, now_mjd: float, gaia_catid: int = 1,
                    on_result=None) -> dict:
        """A battery of small, independent queries that characterise the service.

        Written because the first probe returned an empty night window and the
        cause was not decidable from the schema dump alone.  Each query runs on
        its own and its error is captured rather than raised, so one failure
        cannot mask the others --- the point is to come back with a *complete*
        picture in a single runner pass rather than to iterate blind.

        What each group is actually asking:

        * ``sample_*`` --- what do real rows look like?  This is what settles the
          meaning of ``sid``, which is the leading suspect for the empty window.
        * ``mjd_*`` --- how far does the data actually extend?  If the newest
          ``mjd`` is weeks old, the window was empty because ALeRCE's LSST
          ingestion lags, not because the query is wrong.
        * ``count_*`` --- how much survives each cut, added one at a time, so the
          exact clause that empties the result is identifiable.
        * ``join_*`` --- do the joins produce rows at all, independent of time?
        """
        out: dict = {}

        def run(name: str, adql: str, maxrec: int = 5) -> None:
            try:
                rows = self.query(adql, maxrec=maxrec, retries=2)
                out[name] = {"rows": len(rows), "data": rows[:maxrec], "adql": adql}
            except Exception as exc:                      # noqa: BLE001
                out[name] = {"error": f"{type(exc).__name__}: {exc}"[:500],
                             "adql": adql}
            # Report after every query so a job timeout part-way through still
            # leaves every answer obtained so far on disk.
            if on_result is not None:
                on_result(name, out[name])

        # What tables exist at all.
        run("tables", "SELECT table_name FROM TAP_SCHEMA.tables", maxrec=200)

        # What a real row looks like --- settles the sid question.
        run("sample_lsst_detection",
            "SELECT oid, sid, measurement_id, psfflux, psffluxerr, templateflux, "
            "snr, reliability, raerr, decerr, extendedness, traillength, "
            "isnegative, isdipole, glint_trail "
            "FROM alerce_tap.lsst_detection", maxrec=5)
        run("sample_detection",
            "SELECT oid, sid, measurement_id, mjd, ra, dec, band "
            "FROM alerce_tap.detection", maxrec=5)
        run("sample_object",
            "SELECT oid, sid, tid, meanra, meandec, firstmjd, lastmjd, n_det "
            "FROM alerce_tap.object", maxrec=5)
        run("sample_forced_photometry",
            "SELECT oid, sid, mjd, ra, dec, band FROM alerce_tap.forced_photometry",
            maxrec=5)
        run("sample_xmatch",
            "SELECT oid, sid, catid, dist, oid_catalog FROM alerce_tap.xmatch",
            maxrec=5)

        # Distributions of the discriminating keys.  `object` is far smaller than
        # `detection`, so grouping it is cheap and answers the same question.
        run("object_sid_tid",
            "SELECT sid, tid, COUNT(*) AS n FROM alerce_tap.object "
            "GROUP BY sid, tid", maxrec=50)
        run("xmatch_catid",
            "SELECT catid, COUNT(*) AS n FROM alerce_tap.xmatch GROUP BY catid",
            maxrec=50)

        # How current is the data?
        # MIN/MAX on an indexed column is cheap; a full-table COUNT(*) is not,
        # and on a billion-row detection table it would eat the job timeout for
        # a number nothing here needs.
        run("mjd_range_detection",
            "SELECT MIN(mjd) AS mjd_min, MAX(mjd) AS mjd_max "
            "FROM alerce_tap.detection", maxrec=5)
        run("mjd_range_object",
            "SELECT MIN(firstmjd) AS first_min, MAX(lastmjd) AS last_max "
            "FROM alerce_tap.object", maxrec=5)
        run("mjd_range_lsst_join",
            "SELECT MIN(d.mjd) AS mjd_min, MAX(d.mjd) AS mjd_max "
            "FROM alerce_tap.detection AS d "
            "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
            "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id",
            maxrec=5)

        # Add one cut at a time, so the clause that empties the result is named.
        for days in (1, 3, 7, 30, 120):
            run(f"count_detection_last_{days}d",
                "SELECT COUNT(*) AS n FROM alerce_tap.detection "
                f"WHERE mjd >= {float(now_mjd) - days}", maxrec=5)
        run("count_lsst_join_last_30d",
            "SELECT COUNT(*) AS n FROM alerce_tap.detection AS d "
            "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
            "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id "
            f"WHERE d.mjd >= {float(now_mjd) - 30}", maxrec=5)
        run("count_lsst_join_sid1_last_30d",
            "SELECT COUNT(*) AS n FROM alerce_tap.detection AS d "
            "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
            "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id "
            f"WHERE d.sid = 1 AND d.mjd >= {float(now_mjd) - 30}", maxrec=5)

        # Do the joins yield anything at all, with no time filter?
        run("join_gaia_any",
            "SELECT d.oid, d.mjd, g.oid_catalog, g.parallax, x.dist "
            "FROM alerce_tap.detection AS d "
            "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
            "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id "
            f"JOIN alerce_tap.xmatch AS x ON x.oid = d.oid AND x.sid = d.sid "
            f"AND x.catid = {int(gaia_catid)} "
            "JOIN alerce_tap.gaiadr3_source AS g ON g.oid_catalog = x.oid_catalog",
            maxrec=5)
        # The footprint denominator depends on GROUP BY over FLOOR expressions
        # being accepted; if it is not, the channel cannot compute a rate below
        # 1.0 and is inert, so this is a blocking capability rather than a nicety.
        run("footprint_group_by",
            "SELECT FLOOR(d.ra / 1.0) AS rab, FLOOR(d.dec / 1.0) AS decb, "
            "FLOOR(d.mjd - 0.6666666666) AS night, COUNT(*) AS n "
            "FROM alerce_tap.detection AS d "
            f"WHERE d.sid = 1 AND d.mjd >= {float(now_mjd) - 30} "
            f"AND d.mjd < {float(now_mjd) - 29} "
            "GROUP BY FLOOR(d.ra / 1.0), FLOOR(d.dec / 1.0), "
            "FLOOR(d.mjd - 0.6666666666)", maxrec=20)

        run("join_gaia_nearby_any",
            "SELECT d.oid, d.mjd, g.parallax, x.dist "
            "FROM alerce_tap.detection AS d "
            "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
            "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id "
            f"JOIN alerce_tap.xmatch AS x ON x.oid = d.oid AND x.sid = d.sid "
            f"AND x.catid = {int(gaia_catid)} "
            "JOIN alerce_tap.gaiadr3_source AS g ON g.oid_catalog = x.oid_catalog "
            "WHERE g.parallax > 10.0 AND x.dist < 1.5", maxrec=5)
        return out

    def max_available_mjd(self) -> float | None:
        """Newest LSST detection epoch the service actually holds.

        ALeRCE's TAP mirror is **not** live: measured lag was 15.6 days on
        2026-07-30.  A screen that always asks for "the last two nights" would
        therefore return nothing every single night, forever, and look like a
        clean null.  The window is anchored to this value instead of to the
        wall clock.
        """
        rows = self.query(
            "SELECT MAX(d.mjd) AS mjd_max FROM alerce_tap.detection AS d "
            "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
            "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id",
            maxrec=5, retries=2)
        if not rows:
            return None
        v = rows[0].get("mjd_max")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if v == v else None

    # -- stage A: the night's detections on nearby stars -------------------
    def night_detections(self, mjd_lo: float, mjd_hi: float,
                         parallax_min_mas: float | None = 10.0,
                         xmatch_max_arcsec: float = 1.5,
                         min_abs_snr: float = 6.0,
                         gaia_catid: int = 1,
                         sid_diaobject: int | None = ALERCE_SID_DIAOBJECT,
                         extra_where: str = "",
                         maxrec: int | None = None) -> BrokerResult:
        """Every LSST difference-image detection in ``[mjd_lo, mjd_hi)`` on a *nearby* star.

        The reduction that makes this affordable is done server-side: ALeRCE
        carries a Gaia DR3 cross-match table, so a parallax cut selects
        detections on stars inside a distance limit before a single row crosses
        the network.  Without it a night is several million rows.

        ``parallax_min_mas=None`` disables the Gaia join and returns the whole
        night --- expensive, but it is the honest way to *measure* what the join
        loses (Gaia cross-matches are made at the catalogue epoch, so a
        high-proper-motion nearby dwarf, which is exactly this channel's target,
        can be mis-associated).  Runs should periodically do both and compare.
        """
        res = BrokerResult()
        # Every column the funnel's discriminators need, with the spelling the
        # probe measured on 2026-07-30.  Pulling only flux and position would
        # leave the glint, mover, dipole and artefact tests SILENTLY INERT ---
        # they would run, find nothing to test, and pass everything.
        sel = [
            "d.oid", "d.sid", "d.measurement_id", "d.mjd", "d.ra", "d.dec", "d.band",
            # photometry, and the same-band quiescent flux that makes dF/F* a
            # ratio of two Rubin measurements with no passband transformation
            "ld.psfflux", "ld.psffluxerr", "ld.templateflux", "ld.templatefluxerr",
            "ld.scienceflux", "ld.sciencefluxerr",
            "ld.snr", "ld.reliability", "ld.reliabilityversion",
            # astrometry
            "ld.raerr", "ld.decerr",
            # the discriminators
            "ld.isnegative", "ld.isdipole", "ld.dipolelength", "ld.dipolechi2",
            "ld.extendedness", "ld.traillength", "ld.trail_flag", "ld.glint_trail",
            # artefact flags
            "ld.pixelflags_bad", "ld.pixelflags_cr", "ld.pixelflags_crcenter",
            "ld.pixelflags_edge", "ld.pixelflags_offimage",
            "ld.pixelflags_saturated", "ld.pixelflags_saturatedcenter",
            "ld.pixelflags_suspectcenter", "ld.pixelflags_nodatacenter",
            "ld.pixelflags_streak", "ld.pixelflags_streakcenter",
            # synthetic-source injection: must never enter a science sample
            "ld.pixelflags_injected", "ld.pixelflags_injectedcenter",
            "ld.visit", "ld.detector",
        ]
        # NOT selected: `ld.ssobjectid`.  Large integer ids stored as strings are
        # what broke `oid_catalog` serialisation, and the `sid = 1` filter
        # already excludes solar-system-associated detections, so it is redundant
        # risk.
        where = [f"d.mjd >= {float(mjd_lo)}", f"d.mjd < {float(mjd_hi)}"]
        # `sid_diaobject=None` drops the filter entirely.  The inner join to
        # `lsst_detection` already restricts the result to LSST rows, so the sid
        # cut is an optimisation and a solar-system exclusion --- not something
        # worth returning zero rows over if the encoding is not what we assumed.
        if sid_diaobject is not None:
            where.insert(0, f"d.sid = {int(sid_diaobject)}")
        joins = ["JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
                 "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id"]
        if min_abs_snr > 0:
            where.append(f"(ld.snr > {float(min_abs_snr)} "
                         f"OR ld.snr < {-float(min_abs_snr)})")
        if parallax_min_mas is not None:
            joins.append(f"JOIN alerce_tap.xmatch AS x ON x.oid = d.oid "
                         f"AND x.sid = d.sid AND x.catid = {int(gaia_catid)}")
            # The join key is `oid_catalog` on BOTH sides.  ALeRCE's Gaia table
            # has no `source_id` column --- the live probe caught that guess.
            joins.append("JOIN alerce_tap.gaiadr3_source AS g "
                         "ON g.oid_catalog = x.oid_catalog")
            where.append(f"x.dist < {float(xmatch_max_arcsec)}")
            where.append(f"g.parallax > {float(parallax_min_mas)}")
            # NOT `g.oid_catalog`: the service declares it integer but stores
            # string ids for some catalogues (AllWISE), so SELECTing it fails
            # VOTable serialisation with "required argument is not an integer".
            # It is fine in the JOIN condition, which is where it is needed --- and
            # the authoritative star association is this repository's own
            # proper-motion-propagated match anyway, not the broker's id.
            sel += ["g.parallax", "x.dist AS xmatch_dist"]
        if extra_where:
            where.append(f"({extra_where})")
        adql = ("SELECT " + ", ".join(sel) + " FROM alerce_tap.detection AS d "
                + " ".join(joins) + " WHERE " + " AND ".join(where))
        res.rows = self.query(adql, maxrec=maxrec)
        res.reached = True
        res.calls = self.calls
        res.notes.append(f"adql={adql}")
        if maxrec and len(res.rows) >= (maxrec or self.maxrec):
            res.notes.append("maxrec_reached_results_may_be_truncated")
        res.verdict = "OK" if res.rows else "NO_DETECTIONS_IN_WINDOW"
        return res

    # -- stage B: the denominator ------------------------------------------
    def footprint_bins(self, mjd_lo: float, mjd_hi: float, bin_deg: float = 1.0,
                       sid_diaobject: int | None = ALERCE_SID_DIAOBJECT,
                       maxrec: int | None = None) -> BrokerResult:
        """Which patches of sky Rubin actually observed, per night, in the window.

        THE PROBLEM THIS SOLVES.  The recurrence statistic needs to know how many
        times a star was looked at and showed nothing.  Forced photometry is the
        textbook answer, but measurement beats theory: the broker's LSST forced
        photometry covered **0%** of screened star-nights in the first backfill,
        which pins the ensemble rate at 1.0 and makes promotion impossible --- the
        channel becomes scientifically inert.

        THE ANSWER.  Detections themselves trace where the camera pointed.  A
        1-degree sky bin containing any detection on night N was observed on
        night N, so a catalogued star in that bin was screened on night N whether
        or not it produced an alert.  That is a real denominator over *all*
        targets, not just those that happen to have a diaObject.

        The aggregation is done server-side with GROUP BY, so the whole night's
        footprint costs one small result (at most ~360 x 180 bins per night)
        rather than millions of rows crossing the network.

        Deliberately conservative: a target counts only if its OWN bin was
        observed, with no neighbour dilation.  Targets at field edges in empty
        bins are therefore missed, which *under*-counts trials, which
        *over*-estimates the event rate, which makes every p-value larger.  Erring
        toward fewer detections is the right direction for a search.
        """
        res = BrokerResult()
        b = float(bin_deg)
        # The night key must match `schema.night_id` / `ledger.night_of`:
        # floor(mjd - 16/24), i.e. local noon at Cerro Pachon.
        ra_e = f"FLOOR(d.ra / {b})"
        dec_e = f"FLOOR(d.dec / {b})"
        night_e = "FLOOR(d.mjd - 0.6666666666)"
        where = [f"d.mjd >= {float(mjd_lo)}", f"d.mjd < {float(mjd_hi)}"]
        if sid_diaobject is not None:
            where.insert(0, f"d.sid = {int(sid_diaobject)}")
        # The join to `lsst_detection` is DELIBERATELY absent here.  The probe
        # measured sid=1/tid=1 as LSST diaObject and sid=0/tid=0 as ZTF, so
        # `sid = 1` already restricts to LSST on its own — and this query
        # aggregates over the whole night's detections, where an unnecessary
        # join to a second large table is the difference between a footprint
        # that costs seconds and one that costs the job's timeout.  The join is
        # kept only when the sid filter is off and cannot do the work.
        src = "alerce_tap.detection AS d"
        if sid_diaobject is None:
            src += (" JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
                    "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id")
        adql = (
            f"SELECT {ra_e} AS rab, {dec_e} AS decb, {night_e} AS night, "
            f"COUNT(*) AS n FROM {src} "
            "WHERE " + " AND ".join(where) +
            f" GROUP BY {ra_e}, {dec_e}, {night_e}"
        )
        res.rows = self.query(adql, maxrec=maxrec or 500000)
        res.reached = True
        res.calls = self.calls
        res.notes.append(f"adql={adql}")
        res.verdict = "OK" if res.rows else "NO_FOOTPRINT_IN_WINDOW"
        return res


    def forced_photometry_night(self, mjd_lo: float, mjd_hi: float,
                                parallax_min_mas: float | None = 10.0,
                                xmatch_max_arcsec: float = 1.5,
                                gaia_catid: int = 1,
                                sid_diaobject: int | None = ALERCE_SID_DIAOBJECT,
                                maxrec: int | None = None) -> BrokerResult:
        """Forced photometry on *every* tracked nearby star in the night window.

        This is what makes the screen's statistics well posed.  A star with no
        Rubin ``diaObject`` is invisible to the alert stream entirely, so the
        trial space cannot be "all nearby stars"; it is the **tracked** sample:
        nearby stars that have a ``diaObject`` and therefore receive forced
        photometry on every overlapping visit.  Counting distinct
        ``(object, night)`` pairs here gives the exact denominator for the
        ensemble event rate, and the same rows give each target's visit history.

        The Gaia parallax pre-cut is deliberately identical to the one used for
        the detections, so numerator and denominator are filtered by exactly the
        same criterion --- a star that ALeRCE's cross-match misses is absent from
        both, which biases coverage but never the *rate*.
        """
        res = BrokerResult()
        sel = ["fp.oid", "fp.mjd", "fp.ra", "fp.dec"]
        joins = []
        where = [f"fp.mjd >= {float(mjd_lo)}", f"fp.mjd < {float(mjd_hi)}"]
        if sid_diaobject is not None:
            where.insert(0, f"fp.sid = {int(sid_diaobject)}")
        if parallax_min_mas is not None:
            joins.append(f"JOIN alerce_tap.xmatch AS x ON x.oid = fp.oid "
                         f"AND x.sid = fp.sid AND x.catid = {int(gaia_catid)}")
            # The join key is `oid_catalog` on BOTH sides.  ALeRCE's Gaia table
            # has no `source_id` column --- the live probe caught that guess.
            joins.append("JOIN alerce_tap.gaiadr3_source AS g "
                         "ON g.oid_catalog = x.oid_catalog")
            where.append(f"x.dist < {float(xmatch_max_arcsec)}")
            where.append(f"g.parallax > {float(parallax_min_mas)}")
            # See night_detections: `g.oid_catalog` cannot be SELECTed.
        adql = ("SELECT " + ", ".join(sel) +
                " FROM alerce_tap.forced_photometry AS fp " + " ".join(joins) +
                " WHERE " + " AND ".join(where))
        res.rows = self.query(adql, maxrec=maxrec)
        res.reached = True
        res.calls = self.calls
        res.notes.append(f"adql={adql}")
        res.verdict = "OK" if res.rows else "NO_FORCED_PHOTOMETRY_IN_WINDOW"
        return res

    def forced_photometry(self, oids, mjd_lo: float | None = None,
                          mjd_hi: float | None = None, batch: int = 400,
                          ) -> dict[str, list[float]]:
        """Forced-photometry epochs per object --- the ledger's real denominator.

        Forced photometry is measured at the object's position on *every*
        overlapping visit, whether or not anything was detected, which is the
        only way to know how many times a star was looked at and showed nothing.
        The alert stream alone cannot answer that, and without it the recurrence
        p-value has no denominator.

        Rubin ships a 12-month forced history; epochs earlier than the ledger's
        own start are still useful, so no lower bound is imposed by default.
        """
        out: dict[str, list[float]] = {}
        ids = [str(o) for o in dict.fromkeys(str(o) for o in oids)]
        for i in range(0, len(ids), int(batch)):
            chunk = ids[i:i + int(batch)]
            in_list = ", ".join(f"'{o}'" for o in chunk)
            where = [f"fp.sid = {ALERCE_SID_DIAOBJECT}", f"fp.oid IN ({in_list})"]
            if mjd_lo is not None:
                where.append(f"fp.mjd >= {float(mjd_lo)}")
            if mjd_hi is not None:
                where.append(f"fp.mjd < {float(mjd_hi)}")
            rows = self.query(
                "SELECT fp.oid, fp.mjd FROM alerce_tap.forced_photometry AS fp "
                "WHERE " + " AND ".join(where), maxrec=1_000_000)
            for r in rows:
                try:
                    out.setdefault(str(r["oid"]), []).append(float(r["mjd"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return {k: sorted(set(v)) for k, v in out.items()}


def normalize_alerce_rows(rows: list[dict]) -> list[NormalizedAlert]:
    """Map ALeRCE TAP rows onto normalised alerts.

    Tolerant by construction: the live column set is discovered by the probe, so
    every optional field is looked up through :func:`_f`/:func:`_b`, which return
    ``None`` for anything absent.  A column that disappears therefore degrades
    one discriminator to *untestable* instead of crashing the night's run.
    """
    out: list[NormalizedAlert] = []
    for r in rows:
        flux, ferr = _f(r, "psfflux"), _f(r, "psffluxerr")
        if flux is None or ferr is None:
            continue
        band_raw = r.get("band")
        band = ""
        if band_raw is not None:
            try:
                band = ALERCE_BAND.get(int(band_raw), "")
            except (TypeError, ValueError):
                band = str(band_raw).strip()
        bad = any(_b(r, k) for k in (
            "pixelflags_bad", "pixelflags_cr", "pixelflags_crcenter",
            "pixelflags_saturated", "pixelflags_saturatedcenter",
            "pixelflags_suspectcenter", "pixelflags_nodatacenter",
            "pixelflags_streak", "pixelflags_streakcenter", "pixelflags_edge",
            "pixelflags_offimage", "pixelflags_injected",
            "pixelflags_injectedcenter"))
        ra_err, dec_err = _f(r, "raerr"), _f(r, "decerr")
        out.append(NormalizedAlert(
            alert_id=str(r.get("measurement_id", "")),
            object_id=str(r.get("oid", "")),
            mjd=_f(r, "mjd") or float("nan"),
            band=band,
            ra=_f(r, "ra") or float("nan"),
            dec=_f(r, "dec") or float("nan"),
            dflux_njy=flux, dflux_err_njy=ferr, broker="alerce-lsst",
            # UNITS: DEGREES.  Settled by measurement rather than argued ---
            # `diagnostics.sample_lsst_detection` returned raerr = 4.27e-05,
            # which is 0.154" as degrees and physically absurd read any other
            # way.  The earlier milliarcsec guess drove every value below the
            # 0.05" floor in `pos_err_arcsec`, which pinned every source to the
            # floor and inflated sep_sigma about threefold --- over-rejecting
            # genuine matches as astrometric offsets.
            ra_err_arcsec=None if ra_err is None else ra_err * 3600.0,
            dec_err_arcsec=None if dec_err is None else dec_err * 3600.0,
            template_flux_njy=_f(r, "templateflux"),
            template_flux_err_njy=_f(r, "templatefluxerr"),
            science_flux_njy=_f(r, "scienceflux"),
            science_flux_err_njy=_f(r, "sciencefluxerr"),
            snr=_f(r, "snr"), reliability=_f(r, "reliability"),
            reliability_version=(str(r["reliabilityversion"])
                                 if r.get("reliabilityversion") else None),
            is_dipole=_b(r, "isdipole"),
            dipole_length_arcsec=_f(r, "dipolelength"),
            is_negative=_b(r, "isnegative"),
            extendedness=_f(r, "extendedness"),
            trail_length_arcsec=_f(r, "traillength"),
            glint_trail=_b(r, "glint_trail"),
            pixel_flag_bad=bad,
            ss_object_id=None,      # sid = 1 by construction: no SSO association
            visit=r.get("visit"), detector=r.get("detector"),
            raw={"gaia_source_id": r.get("gaia_source_id"),
                 "parallax": _f(r, "parallax"),
                 "xmatch_dist": _f(r, "xmatch_dist")},
        ))
    return out


class LasairLSST:
    """Thin REST client for Lasair-LSST with hourly-quota awareness.

    Uses ``requests`` directly rather than the ``lasair`` pip package: the
    package is a very thin POST wrapper, and depending on it would add a
    dependency whose defaults (``lite=True``, 60 s timeout, never-expiring
    on-disk cache) all have to be overridden anyway.
    """

    def __init__(self, token: str | None = None, endpoint: str = LASAIR_ENDPOINT,
                 timeout: float = 300.0, calls_per_hour: int = 100,
                 max_rows: int = 10000):
        self.token = token or os.environ.get("LASAIR_TOKEN", "")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.calls_per_hour = int(calls_per_hour)
        self.max_rows = int(max_rows)
        self.calls = 0
        self._call_times: list[float] = []

    # -- plumbing ----------------------------------------------------------
    def _throttle(self) -> None:
        """Block until a call fits inside the sliding one-hour quota.

        Exceeding the quota returns HTTP 429 and (at the registered tier) burns
        the rest of the hour, which on a nightly cron means losing the night.
        Waiting is strictly better than being throttled.
        """
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 3600.0]
        if len(self._call_times) >= self.calls_per_hour:
            sleep_s = 3600.0 - (now - self._call_times[0]) + 1.0
            time.sleep(max(0.0, sleep_s))
            self._call_times = [t for t in self._call_times if time.time() - t < 3600.0]
        self._call_times.append(time.time())

    def _post(self, method: str, data: dict, retries: int = 4):
        import requests

        if not self.token:
            raise BrokerError("no LASAIR_TOKEN: register free at "
                              "https://lasair.lsst.ac.uk/register/ and set the secret")
        url = f"{self.endpoint}/{method}/"      # the trailing slash is mandatory
        headers = {"Authorization": f"Token {self.token}"}
        last = None
        for attempt in range(retries):
            self._throttle()
            self.calls += 1
            try:
                r = requests.post(url, data=data, headers=headers, timeout=self.timeout)
            except Exception as exc:                     # network flake
                last = f"transport:{exc}"
                time.sleep(2.0 ** attempt)
                continue
            if r.status_code == 429:
                # Quota exhausted despite throttling (another job shares the
                # account): back off hard rather than hammering.
                last = "http429"
                time.sleep(min(600.0, 60.0 * (attempt + 1)))
                continue
            if r.status_code >= 400:
                # 400 echoes the generated SQL back, which is the useful part.
                raise BrokerError(f"{method} HTTP {r.status_code}: {r.text[:600]}")
            try:
                payload = r.json()
            except ValueError as exc:
                raise BrokerError(f"{method}: non-JSON response: {exc}") from exc
            if isinstance(payload, dict) and payload.get("error"):
                raise BrokerError(f"{method}: {payload['error']}")
            return payload
        raise BrokerError(f"{method}: exhausted retries ({last})")

    # -- stage A -----------------------------------------------------------
    def night_objects(self, lookback_days: float = 1.5,
                      max_separation_arcsec: float = 1.5,
                      stellar_only: bool = True,
                      max_ndia_sources: int = 0,
                      min_abs_glat: float = 0.0,
                      max_pages: int = 40) -> BrokerResult:
        """Bulk pass: objects updated within ``lookback_days``, near a catalogued star.

        ``stellar_only`` uses Sherlock's own catalogue association
        (``catalogue_object_type = "star"``).  That is a *cheap* cut, not a
        trusted one: Sherlock matches against catalogue positions at the
        catalogue epoch, so a high-proper-motion nearby dwarf --- precisely this
        channel's target population --- can be mis-associated or orphaned.  The
        authoritative association is this repository's own proper-motion
        propagated cross-match in ``screen.py``; runs should periodically be
        repeated with ``stellar_only=False`` to measure what the cheap cut loses.

        ``max_ndia_sources`` (0 = no cut) suppresses AGN and high-amplitude
        variables, which dominate the recently-active object list.
        """
        res = BrokerResult()
        cond = [f"mjdnow() - objects.lastDiaSourceMjdTai < {float(lookback_days)}"]
        if max_separation_arcsec > 0:
            cond.append(f"sherlock_classifications.separationArcsec "
                        f"< {float(max_separation_arcsec)}")
        if stellar_only:
            cond.append('sherlock_classifications.catalogue_object_type = "star"')
        if max_ndia_sources > 0:
            cond.append(f"objects.nDiaSources <= {int(max_ndia_sources)}")
        if min_abs_glat > 0:
            cond.append(f"ABS(objects.glat) > {float(min_abs_glat)}")
        conditions = " AND ".join(cond) + " ORDER BY objects.diaObjectId"
        tables = "objects, sherlock_classifications"

        offset = 0
        for page in range(int(max_pages)):
            payload = self._post("query", {
                "selected": _STAGE_A_SELECT, "tables": tables,
                "conditions": conditions, "limit": self.max_rows, "offset": offset,
            })
            if not isinstance(payload, list):
                raise BrokerError(f"query: expected a row list, got {type(payload)}")
            res.rows.extend(payload)
            res.reached = True
            # `limit` is clamped silently to the account tier, so a short page is
            # the only reliable end-of-results signal.
            if len(payload) < self.max_rows:
                break
            offset += len(payload)
            if page == int(max_pages) - 1:
                res.notes.append(f"page_cap_reached_{max_pages}_results_truncated")
        res.calls = self.calls
        res.verdict = "OK" if res.rows else "NO_OBJECTS_IN_WINDOW"
        return res

    # -- stage B -----------------------------------------------------------
    def object_epochs(self, object_id, min_reliability: float | None = None,
                      ) -> tuple[list[NormalizedAlert], list[float]]:
        """Per-epoch ``diaSources`` (as normalised alerts) plus forced-photometry MJDs.

        ``min_reliability`` is applied here, client-side, because Lasair's
        server-side ``reliabilityThreshold`` parameter is inert (see the module
        docstring).  Filtering is left to the funnel by default (``None``).
        """
        payload = self._post("object", {"objectId": str(object_id),
                                        "lite": "False", "lasair_added": "True"})
        if not isinstance(payload, dict):
            raise BrokerError(f"object {object_id}: unexpected payload type")
        alerts = normalize_lasair_diasources(payload, min_reliability=min_reliability)
        forced = [float(f["midpointMjdTai"])
                  for f in payload.get("diaForcedSourcesList") or []
                  if f.get("midpointMjdTai") is not None]
        return alerts, sorted(forced)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _f(row: dict, key: str):
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _b(row: dict, key: str):
    v = row.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "t", "1", "yes")
    return None


# Pixel flags that invalidate a difference-image detection outright.  `_streak`
# and `glint_trail` matter more here than in any other science case: a satellite
# glint IS a brief achromatic reflection, i.e. an exact mimic of the flash
# signature, and it is the dominant instrumental confounder for this channel.
_FATAL_PIXEL_FLAGS = (
    "pixelFlags_bad", "pixelFlags_cr", "pixelFlags_crCenter",
    "pixelFlags_saturated", "pixelFlags_saturatedCenter",
    "pixelFlags_suspectCenter", "pixelFlags_nodataCenter",
    "pixelFlags_streak", "pixelFlags_streakCenter",
    "pixelFlags_edge", "pixelFlags_offimage",
    "pixelFlags_injected", "pixelFlags_injectedCenter",
)


def normalize_lasair_diasources(payload: dict, min_reliability: float | None = None,
                                ) -> list[NormalizedAlert]:
    """Map a Lasair ``/api/object/?lite=False`` payload onto normalised alerts."""
    obj_id = str(payload.get("diaObjectId", ""))
    out: list[NormalizedAlert] = []
    for s in payload.get("diaSourcesList") or []:
        flux, ferr = _f(s, "psfFlux"), _f(s, "psfFluxErr")
        if flux is None or ferr is None:
            continue
        rel = _f(s, "reliability")
        if min_reliability is not None and rel is not None and rel < min_reliability:
            continue
        bad = any(_b(s, k) for k in _FATAL_PIXEL_FLAGS)
        ra_err = _f(s, "raErr")
        dec_err = _f(s, "decErr")
        alert = NormalizedAlert(
            alert_id=str(s.get("diaSourceId", "")),
            object_id=obj_id,
            mjd=_f(s, "midpointMjdTai") or float("nan"),
            band=str(s.get("band") or ""),
            ra=_f(s, "ra") or float("nan"),
            # Lasair renames Rubin's `dec` to `decl`; accept either so a schema
            # revert does not silently produce zero matches.
            dec=(_f(s, "decl") if s.get("decl") is not None else _f(s, "dec"))
            or float("nan"),
            dflux_njy=flux, dflux_err_njy=ferr, broker="lasair-lsst",
            # Same SDM units as the ALeRCE path: raErr/decErr are DEGREES
            # (measured, see normalize_alerce_rows).
            ra_err_arcsec=None if ra_err is None else ra_err * 3600.0,
            dec_err_arcsec=None if dec_err is None else dec_err * 3600.0,
            template_flux_njy=_f(s, "templateFlux"),
            template_flux_err_njy=_f(s, "templateFluxErr"),
            science_flux_njy=_f(s, "scienceFlux"),
            science_flux_err_njy=_f(s, "scienceFluxErr"),
            snr=_f(s, "snr"), reliability=rel,
            reliability_version=(str(s["reliabilityVersion"])
                                 if s.get("reliabilityVersion") else None),
            is_dipole=_b(s, "isDipole"),
            dipole_significance=None,
            dipole_length_arcsec=_f(s, "dipoleLength"),
            is_negative=_b(s, "isNegative"),
            extendedness=_f(s, "extendedness"),
            trail_length_arcsec=_f(s, "trailLength"),
            glint_trail=_b(s, "glint_trail"),
            pixel_flag_bad=bad,
            ss_object_id=(str(s["ssObjectId"]) if s.get("ssObjectId") else None),
            n_prv_sources=None,
            visit=s.get("visit"), detector=s.get("detector"),
            raw={"dipoleChi2": _f(s, "dipoleChi2"),
                 "dipoleAngle": _f(s, "dipoleAngle"),
                 "trail_flag": _b(s, "trail_flag"),
                 "apFlux": _f(s, "apFlux")},
        )
        out.append(alert)
    return out
