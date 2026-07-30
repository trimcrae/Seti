"""ALeRCE TAP access to the LSST solar-system tables.  Runner-only network code.

Egress to the brokers is blocked in the development sandbox
(``docs/channel-brief.md`` §0), so everything here runs on a GitHub Actions
runner via ``.github/workflows/loom*.yml``.

What the stream actually carries
--------------------------------
Two tables matter, and both were verified against the upstream LSST alert-packet
v11.1 Avro schemas (``lsst.v11_1.ssSource`` and ``lsst.v11_1.mpc_orbits``):
ALeRCE exposes the packet fields **verbatim, lower-cased, in schema order** rather
than renaming them.

``alerce_tap.lsst_ss_detection`` (40 columns) is one row per detection associated
with a known minor planet, and it carries the observed-minus-predicted ephemeris
offset already decomposed: ``ephoffsetalongtrack``, ``ephoffsetcrosstrack``,
``ephoffset``, plus ``ephra``/``ephdec`` (the prediction *from the orbit in
mpc_orbits*), ``ephrate``/``ephratera``/``ephratedec``, ``phaseangle``,
``elongation``, ``toporange``, ``heliorange`` and their rates, the full
heliocentric and topocentric state vectors, and ``diadistancerank`` (1 for the
nearest source to the prediction).

``alerce_tap.lsst_mpc_orbits`` (53 columns) is one row per object: the orbital
elements each with an uncertainty twin, ``h`` and ``g``, the fit-quality block
(``arc_length_total``, ``nobs_total``, ``nopp``, ``u_param``,
``normalized_rms``, ``not_normalized_rms``, ``epoch_mjd``), and — decisively —
the non-gravitational block ``yarkovsky``, ``srp``, ``a1``, ``a2``, ``a3``,
``dt``, each with an uncertainty.

Three things are not known from the schema and are therefore *measured* here
rather than assumed, which is the same discipline TOCSIN's probe applies:

1. **Whether the interesting columns are populated at all.**  Every field in the
   non-gravitational block is nullable with ``default: null``, and MPC fits
   non-gravitational terms for only a small minority of objects.  If
   ``yarkovsky``/``srp`` are all-NULL in the mirror, the channel must run off the
   per-detection residuals instead — which it can, and that path is stronger
   anyway because it does not inherit MPC's choice of which objects were
   interesting enough to fit.  :meth:`null_fractions` settles it, and it is the
   single highest-value query in the probe.
2. **The angular unit of the ``ephoffset*`` columns.**  Arcsec is the near-certain
   intent but the schema does not state it.  The distribution measured by
   :meth:`null_fractions` decides it: a well-observed minor planet's residual is
   ~0.1 arcsec, so a median of order 0.1 means arcsec, of order 1e2 means mas,
   and of order 1e-6 means radians.
3. **The join key between the two tables and the epoch source.**  ``ssSource``
   carries ``diasourceid`` and ``ssobjectid``; ALeRCE's generic ``detection``
   table carries ``oid``/``sid``/``measurement_id``/``mjd``.  Which of those the
   solar-system rows key on is not derivable from the column list, so
   :meth:`diagnostics` tries each candidate join and reports which returns rows.

The unit caveat that must not be forgotten
------------------------------------------
``lsst_mpc_orbits.yarkovsky`` is documented in units of ``1e-10 au/day^2`` — so
Bennu's ``A2 = -4.62e-14 au/day^2`` appears in the column as ``-4.6e-4``, and
reading the column as au/day^2 would overstate every acceleration by ten orders
of magnitude.  ``srp`` is in ``m^2/ton``.  The ``a1``/``a2``/``a3`` columns are
*also* documented ``m^2/ton``, which is dimensionally wrong for Marsden
accelerations; they are treated as unit-unverified and are calibrated against
objects with published solutions before use.  All three conversions live in
:mod:`seti.loom.nongrav` so there is exactly one place they can be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tocsin.brokers import ALERCE_TAP, AlerceTAP, BrokerError

SS_DETECTION_TABLE = "alerce_tap.lsst_ss_detection"
MPC_ORBITS_TABLE = "alerce_tap.lsst_mpc_orbits"
SS_OBJECT_TABLE = "alerce_tap.lsst_ss_object"
DETECTION_TABLE = "alerce_tap.detection"

# ALeRCE `sid` for solar-system-associated rows.  TOCSIN measured sid=1 -> LSST
# diaObject and sid=0 -> ZTF on 2026-07-30; the ssObject value is *not* measured
# and is a probe question, so every query takes it as a parameter and `None`
# drops the clause entirely rather than silently returning an empty result.
ALERCE_SID_SSOBJECT = 2

# Verified column names, in schema order, for lsst_ss_detection.
SS_DETECTION_COLUMNS: tuple[str, ...] = (
    "diasourceid", "ssobjectid", "designation",
    "ecllambda", "eclbeta", "gallon", "gallat", "elongation", "phaseangle",
    "toporange", "toporangerate", "heliorange", "heliorangerate",
    "ephra", "ephdec", "ephvmag", "ephrate", "ephratera", "ephratedec",
    "ephoffset", "ephoffsetra", "ephoffsetdec",
    "ephoffsetalongtrack", "ephoffsetcrosstrack",
    "helio_x", "helio_y", "helio_z", "helio_vx", "helio_vy", "helio_vz",
    "helio_vtot",
    "topo_x", "topo_y", "topo_z", "topo_vx", "topo_vy", "topo_vz", "topo_vtot",
    "diadistancerank",
)

# The columns of lsst_mpc_orbits this channel needs.  Not the full 53: the
# per-element uncertainty twins for elements that only enter a clustering
# statistic are not worth the bytes, but every fit-quality and
# non-gravitational field is here because each one gates a decision.
MPC_ORBIT_COLUMNS: tuple[str, ...] = (
    "a", "q", "e", "i", "node", "argperi", "peri_time", "mean_anomaly",
    "period", "mean_motion",
    "a_unc", "e_unc", "i_unc",
    "h", "g", "earth_moid",
    "arc_length_total", "arc_length_sel", "nobs_total", "nobs_total_sel",
    "nopp", "u_param", "not_normalized_rms", "normalized_rms",
    "epoch_mjd",
    "yarkovsky", "yarkovsky_unc", "srp", "srp_unc",
    "a1", "a1_unc", "a2", "a2_unc", "a3", "a3_unc", "dt", "dt_unc",
)

# `ssObject` columns: a SIX-BAND phase-curve fit per object.  This is the second,
# photometric axis of the channel, and it is independent of the dynamical
# selection -- nothing here enters the anomaly cut, so a homogeneity statistic on
# it is a genuine test rather than a restatement of the selection.  `tisserand_j`
# and `moid_earth` are dynamical context (which population the object belongs to,
# and how close it comes to Earth); `extendedness` catches an unrecognised coma,
# which is the outgassing explanation showing itself directly.
SS_OBJECT_COLUMNS: tuple[str, ...] = (
    "ssobjectid", "discoverysubmissiondate", "firstobservationdate",
    "arc", "numobs",
    *[f"{b}_h" for b in "ugrizy"],
    *[f"{b}_g12" for b in "ugrizy"],
    *[f"{b}_herr" for b in "ugrizy"],
    *[f"{b}_ndata" for b in "ugrizy"],
    "medianextendedness", "minextendedness", "maxextendedness",
    "moidearth", "moidearthdeltav", "tisserandj",
    "phaseanglemin", "phaseanglemax",
)

# Columns whose null fraction decides the channel's architecture.
CRITICAL_NULLABLE = ("yarkovsky", "srp", "a1", "a2", "a3", "dt")


@dataclass
class SSOResult:
    """What one solar-system pass returned, with its degradation stated."""

    rows: list[dict] = field(default_factory=list)
    calls: int = 0
    reached: bool = False
    verdict: str = "NOT_RUN"
    notes: list[str] = field(default_factory=list)


class AlerceSSO:
    """Solar-system queries against the same public TAP service TOCSIN uses.

    Composition rather than inheritance: :class:`~seti.tocsin.brokers.AlerceTAP`
    already owns the session with a real per-request socket timeout, the
    exponential-backoff retry, and the fail-fast on ADQL syntax errors — all three
    of which were added because an unattended cron hung on a single stuck query
    and burned a whole job.  Re-deriving them here would mean two places for that
    bug to come back.
    """

    def __init__(self, url: str = ALERCE_TAP, timeout: float = 900.0,
                 maxrec: int = 2_000_000):
        self.tap = AlerceTAP(url=url, timeout=timeout, maxrec=maxrec)

    @property
    def calls(self) -> int:
        return self.tap.calls

    def query(self, adql: str, maxrec: int | None = None, retries: int = 4):
        return self.tap.query(adql, maxrec=maxrec, retries=retries)

    # -- schema discovery --------------------------------------------------
    def describe(self) -> dict:
        """Live column list for the solar-system tables, from ``TAP_SCHEMA``.

        Committed verbatim by the probe so a broker schema change shows up as a
        diff in version control rather than as an unexplained null months later.
        """
        tables = (SS_DETECTION_TABLE, MPC_ORBITS_TABLE, SS_OBJECT_TABLE,
                  DETECTION_TABLE)
        names = ", ".join(f"'{t}'" for t in tables)
        rows = self.query(
            "SELECT table_name, column_name, datatype FROM TAP_SCHEMA.columns "
            f"WHERE table_name IN ({names}) ORDER BY table_name, column_name",
            maxrec=20000)
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(str(r.get("table_name", "")), []).append(
                str(r.get("column_name", "")))
        return out

    def available_columns(self, table: str) -> list[str]:
        """The columns a table actually has, from ``TAP_SCHEMA``.

        Used to build the ``ssObject`` query adaptively.  Unlike
        ``lsst_ss_detection`` and ``lsst_mpc_orbits``, whose ALeRCE spellings were
        verified against the upstream Avro schemas byte for byte, the ``ssObject``
        column names are known only from the LSST science data model, where they
        are camel-cased and irregular (``g_H``, ``MOIDEarth``, ``tisserand_J``).
        Guessing the mirror's lower-cased forms and SELECTing them blind would
        produce an ADQL error and take the whole photometric axis down with it, so
        the query asks the service what it has and uses the intersection.
        """
        rows = self.query(
            "SELECT column_name FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{table}'", maxrec=2000, retries=2)
        return [str(r.get("column_name", "")).lower() for r in rows]

    def ss_objects(self, ss_keys: list | None = None,
                   columns: list[str] | None = None,
                   maxrec: int | None = None) -> SSOResult:
        """Per-object six-band phase-curve fits and dynamical context.

        Feeds :func:`seti.loom.replication.photometric_homogeneity`, which is the
        channel's independent second axis: nothing in this table enters the
        dynamical anomaly cut, so a homogeneity statistic on it is a real test
        rather than a restatement of the selection.
        """
        res = SSOResult()
        try:
            have = set(self.available_columns(SS_OBJECT_TABLE))
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"ssObject schema unavailable: {exc}"[:400])
            return res
        if not have:
            res.verdict = "TABLE_ABSENT"
            res.notes.append(f"{SS_OBJECT_TABLE} has no columns in TAP_SCHEMA; the "
                             f"photometric axis is unavailable in this mirror")
            return res
        want = [c.lower() for c in (columns or SS_OBJECT_COLUMNS)]
        cols = [c for c in want if c in have]
        missing = [c for c in want if c not in have]
        if missing:
            res.notes.append(f"{len(missing)} expected ssObject columns absent: "
                             f"{missing[:12]}")
        if not cols:
            res.verdict = "NO_EXPECTED_COLUMNS"
            return res
        adql = f"SELECT {', '.join(cols)} FROM {SS_OBJECT_TABLE}"
        if ss_keys:
            joined = ", ".join(str(int(k)) for k in ss_keys)
            adql += f" WHERE ssobjectid IN ({joined})"
        try:
            res.rows = self.query(adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "OK" if res.rows else "EMPTY"
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"ssObject query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def null_fractions(self, sample: int = 200_000) -> dict:
        """Are the non-gravitational columns populated, and in what units?

        The highest-value query in the whole channel.  Everything downstream
        branches on the answer: if ``yarkovsky`` and ``srp`` are all-NULL in the
        mirror then the ``mpc_orbits`` path is dead and the channel runs on
        per-detection residuals; if they are populated then both paths run and
        cross-check each other.

        Counts and quantiles are computed server-side so no rows cross the wire,
        and each column is asked for on its own so that one missing column cannot
        take the whole answer down with it.
        """
        out: dict = {}
        for col in CRITICAL_NULLABLE:
            adql = (f"SELECT COUNT(*) AS n_total, COUNT({col}) AS n_present, "
                    f"MIN({col}) AS v_min, MAX({col}) AS v_max, AVG({col}) AS v_mean "
                    f"FROM {MPC_ORBITS_TABLE}")
            try:
                rows = self.query(adql, maxrec=5, retries=2)
                out[f"orbits_{col}"] = {"rows": rows, "adql": adql}
            except Exception as exc:                          # noqa: BLE001
                out[f"orbits_{col}"] = {"error": f"{type(exc).__name__}: {exc}"[:400],
                                        "adql": adql}
        for col in ("ephoffset", "ephoffsetalongtrack", "ephoffsetcrosstrack",
                    "ephrate", "heliorange", "toporange", "phaseangle"):
            adql = (f"SELECT COUNT(*) AS n_total, COUNT({col}) AS n_present, "
                    f"MIN({col}) AS v_min, MAX({col}) AS v_max, AVG({col}) AS v_mean "
                    f"FROM {SS_DETECTION_TABLE}")
            try:
                rows = self.query(adql, maxrec=5, retries=2)
                out[f"ssdet_{col}"] = {"rows": rows, "adql": adql}
            except Exception as exc:                          # noqa: BLE001
                out[f"ssdet_{col}"] = {"error": f"{type(exc).__name__}: {exc}"[:400],
                                       "adql": adql}
        return out

    def diagnostics(self, on_result=None) -> dict:
        """Small independent queries that settle every open schema question.

        Each runs on its own and its error is captured rather than raised, so one
        failure cannot mask the others: the point is to come back with a complete
        picture in a single runner pass instead of iterating blind against a
        service that takes minutes per query.  ``on_result`` is called after every
        query so a job timeout part-way through still leaves every answer obtained
        so far on disk.
        """
        out: dict = {}

        def run(name: str, adql: str, maxrec: int = 5) -> None:
            try:
                rows = self.query(adql, maxrec=maxrec, retries=2)
                out[name] = {"rows": len(rows), "data": rows[:maxrec], "adql": adql}
            except Exception as exc:                          # noqa: BLE001
                out[name] = {"error": f"{type(exc).__name__}: {exc}"[:500],
                             "adql": adql}
            if on_result is not None:
                on_result(name, out[name])

        run("tables", "SELECT table_name FROM TAP_SCHEMA.tables", maxrec=300)

        # What a real solar-system detection row looks like: settles the units of
        # the offset columns and whether they are populated for ordinary objects.
        run("sample_ss_detection",
            f"SELECT {', '.join(SS_DETECTION_COLUMNS[:24])} "
            f"FROM {SS_DETECTION_TABLE}", maxrec=10)
        run("sample_mpc_orbits",
            f"SELECT {', '.join(MPC_ORBIT_COLUMNS)} FROM {MPC_ORBITS_TABLE}",
            maxrec=10)

        # Which key joins the two solar-system tables, and where does mjd live?
        # `lsst_ss_detection` carries no epoch, so an epoch must come from the
        # generic `detection` table and the join is the open question.
        run("join_ss_detection_on_measurement_id",
            "SELECT d.oid, d.sid, d.mjd, d.ra, d.dec, d.band, "
            "ss.ssobjectid, ss.designation, ss.ephoffsetalongtrack "
            f"FROM {SS_DETECTION_TABLE} AS ss "
            f"JOIN {DETECTION_TABLE} AS d ON d.measurement_id = ss.diasourceid",
            maxrec=10)
        run("join_ss_detection_on_oid",
            "SELECT d.oid, d.sid, d.mjd, ss.ssobjectid, ss.ephoffsetalongtrack "
            f"FROM {SS_DETECTION_TABLE} AS ss "
            f"JOIN {DETECTION_TABLE} AS d ON d.oid = ss.ssobjectid", maxrec=10)

        # The photometric axis.  Column spellings here are NOT verified against a
        # primary source (unlike ss_detection and mpc_orbits), so the probe asks
        # the service for the real list rather than guessing.
        run("ss_object_columns",
            "SELECT column_name, datatype FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{SS_OBJECT_TABLE}' ORDER BY column_name",
            maxrec=300)
        run("count_ss_object", f"SELECT COUNT(*) AS n FROM {SS_OBJECT_TABLE}",
            maxrec=5)

        # Does mpc_orbits key on ssobjectid, on a designation, or on oid?
        for key in ("ssobjectid", "designation", "oid", "mpc_orb_jsonb"):
            run(f"orbits_has_{key}",
                f"SELECT {key} FROM {MPC_ORBITS_TABLE}", maxrec=5)

        # Population sizes and epoch coverage: is this table big enough for the
        # matched-null machinery, and how far behind is the mirror?
        run("count_orbits", f"SELECT COUNT(*) AS n FROM {MPC_ORBITS_TABLE}", maxrec=5)
        run("count_ss_detection",
            f"SELECT COUNT(*) AS n FROM {SS_DETECTION_TABLE}", maxrec=5)
        run("orbits_epoch_range",
            f"SELECT MIN(epoch_mjd) AS lo, MAX(epoch_mjd) AS hi, "
            f"MIN(arc_length_total) AS arc_lo, MAX(arc_length_total) AS arc_hi "
            f"FROM {MPC_ORBITS_TABLE}", maxrec=5)

        # Do offsets exist for objects with GOOD orbits?  A residual population
        # made only of short-arc objects is a fit-quality catalogue, not a signal.
        run("ss_offset_with_good_orbit",
            "SELECT ss.designation, ss.ephoffset, ss.ephoffsetalongtrack, "
            "o.normalized_rms, o.arc_length_total, o.nopp, o.h "
            f"FROM {SS_DETECTION_TABLE} AS ss "
            f"JOIN {MPC_ORBITS_TABLE} AS o ON o.ssobjectid = ss.ssobjectid "
            "WHERE o.normalized_rms < 1.5 AND o.nopp >= 3", maxrec=20)

        # How many objects have a fitted non-gravitational solution at all?  This
        # is the size of the mpc_orbits path's parent population.
        run("count_orbits_with_yarkovsky",
            f"SELECT COUNT(*) AS n FROM {MPC_ORBITS_TABLE} "
            "WHERE yarkovsky IS NOT NULL", maxrec=5)
        run("count_orbits_with_srp",
            f"SELECT COUNT(*) AS n FROM {MPC_ORBITS_TABLE} "
            "WHERE srp IS NOT NULL", maxrec=5)
        run("count_orbits_nongrav_snr3",
            f"SELECT COUNT(*) AS n FROM {MPC_ORBITS_TABLE} "
            "WHERE yarkovsky IS NOT NULL AND yarkovsky_unc > 0 "
            "AND ABS(yarkovsky) > 3 * yarkovsky_unc", maxrec=5)

        # Whether a GROUP BY over an expression is accepted decides whether the
        # per-object aggregation can be done server-side; if it is not, every
        # residual row has to cross the wire.
        run("group_by_ssobjectid",
            "SELECT ss.ssobjectid, COUNT(*) AS n, "
            "AVG(ss.ephoffsetalongtrack) AS mean_along "
            f"FROM {SS_DETECTION_TABLE} AS ss GROUP BY ss.ssobjectid", maxrec=20)
        return out

    # -- the parent population --------------------------------------------
    def orbits(self, h_max: float | None = None,
               normalized_rms_max: float | None = None,
               min_oppositions: int | None = None,
               min_arc_days: float | None = None,
               require_nongrav: bool = False,
               key_column: str = "ssobjectid",
               extra_where: str = "",
               maxrec: int | None = None) -> SSOResult:
        """The screened parent population from ``lsst_mpc_orbits``.

        The cuts are all optional and all pushed server-side.  They are *quality*
        cuts, not signal cuts: an object whose orbit is badly determined cannot
        contribute either a detection or a trial, and including it would put the
        channel in the position KNELL documents, ranking objects by how poorly
        they were observed.

        ``require_nongrav`` restricts to objects with a fitted non-gravitational
        solution.  Off by default: that subset is chosen by whoever decided an
        object was worth fitting, so it is not a population a rate can be computed
        against, and the per-detection residual path does not need it.
        """
        res = SSOResult()
        cols = [key_column, *MPC_ORBIT_COLUMNS]
        where: list[str] = []
        if h_max is not None:
            where.append(f"h <= {float(h_max)}")
        if normalized_rms_max is not None:
            where.append(f"normalized_rms <= {float(normalized_rms_max)}")
        if min_oppositions is not None:
            where.append(f"nopp >= {int(min_oppositions)}")
        if min_arc_days is not None:
            where.append(f"arc_length_total >= {float(min_arc_days)}")
        if require_nongrav:
            where.append("yarkovsky IS NOT NULL")
        if extra_where:
            where.append(f"({extra_where})")
        adql = f"SELECT {', '.join(cols)} FROM {MPC_ORBITS_TABLE}"
        if where:
            adql += " WHERE " + " AND ".join(where)
        try:
            res.rows = self.query(adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "OK" if res.rows else "EMPTY"
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"orbits query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    # -- the residual time series -----------------------------------------
    def residual_detections(self, mjd_lo: float | None = None,
                            mjd_hi: float | None = None,
                            designations: list[str] | None = None,
                            ss_keys: list | None = None,
                            join_on: str = "measurement_id",
                            sid_ssobject: int | None = None,
                            nearest_only: bool = True,
                            extra_where: str = "",
                            maxrec: int | None = None) -> SSOResult:
        """Per-detection ephemeris residuals, with epoch and sky position.

        ``join_on`` selects the key the probe found to work
        (``measurement_id`` joins ``detection.measurement_id`` to
        ``lsst_ss_detection.diasourceid``; ``oid`` joins on the object id).  It is
        a parameter rather than a constant precisely so a probe result can be
        applied through config without editing a query.

        ``nearest_only`` keeps ``diadistancerank = 1``, the source nearest the
        prediction.  Without it a crowded field contributes several candidate
        associations for one prediction and the residual distribution acquires a
        tail made entirely of mis-associations — which would be the channel's
        largest false-positive source and the easiest to remove.
        """
        res = SSOResult()
        ss_cols = ", ".join(f"ss.{c}" for c in SS_DETECTION_COLUMNS)
        on = ("d.measurement_id = ss.diasourceid" if join_on == "measurement_id"
              else "d.oid = ss.ssobjectid")
        where: list[str] = []
        if mjd_lo is not None:
            where.append(f"d.mjd >= {float(mjd_lo)}")
        if mjd_hi is not None:
            where.append(f"d.mjd < {float(mjd_hi)}")
        if sid_ssobject is not None:
            where.append(f"d.sid = {int(sid_ssobject)}")
        if nearest_only:
            where.append("ss.diadistancerank = 1")
        if designations:
            quoted = ", ".join("'" + str(d).replace("'", "''") + "'"
                               for d in designations)
            where.append(f"ss.designation IN ({quoted})")
        if ss_keys:
            joined = ", ".join(str(int(k)) for k in ss_keys)
            where.append(f"ss.ssobjectid IN ({joined})")
        if extra_where:
            where.append(f"({extra_where})")
        adql = (f"SELECT d.oid, d.sid, d.measurement_id, d.mjd, d.ra, d.dec, "
                f"d.band, {ss_cols} "
                f"FROM {SS_DETECTION_TABLE} AS ss "
                f"JOIN {DETECTION_TABLE} AS d ON {on}")
        if where:
            adql += " WHERE " + " AND ".join(where)
        try:
            res.rows = self.query(adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "OK" if res.rows else "EMPTY"
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"residual query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def object_residual_summary(self, mjd_lo: float | None = None,
                                mjd_hi: float | None = None,
                                min_detections: int = 6,
                                join_on: str = "measurement_id",
                                nearest_only: bool = True,
                                maxrec: int | None = None) -> SSOResult:
        """Per-object residual aggregates computed server-side.

        The cheap first pass.  Pulling every residual row for a survey that
        detects ~10^4-10^5 minor planets a night is not affordable, and it is not
        necessary: the objects worth a full time-series pull are the ones whose
        *aggregate* along-track offset is large or whose scatter is structured.
        This is the same two-stage shape TOCSIN uses — one bulk query to build a
        shortlist, then expensive per-object work only on the shortlist.
        """
        res = SSOResult()
        on = ("d.measurement_id = ss.diasourceid" if join_on == "measurement_id"
              else "d.oid = ss.ssobjectid")
        where: list[str] = []
        if mjd_lo is not None:
            where.append(f"d.mjd >= {float(mjd_lo)}")
        if mjd_hi is not None:
            where.append(f"d.mjd < {float(mjd_hi)}")
        if nearest_only:
            where.append("ss.diadistancerank = 1")
        adql = ("SELECT ss.ssobjectid, COUNT(*) AS n_det, "
                "MIN(d.mjd) AS mjd_min, MAX(d.mjd) AS mjd_max, "
                "AVG(ss.ephoffsetalongtrack) AS mean_along, "
                "AVG(ss.ephoffsetcrosstrack) AS mean_cross, "
                "MAX(ABS(ss.ephoffsetalongtrack)) AS max_abs_along, "
                "AVG(ss.ephoffset) AS mean_offset, "
                "AVG(ss.heliorange) AS mean_heliorange, "
                "MIN(ss.heliorange) AS min_heliorange, "
                "MAX(ss.heliorange) AS max_heliorange, "
                "AVG(ss.ephrate) AS mean_ephrate "
                f"FROM {SS_DETECTION_TABLE} AS ss "
                f"JOIN {DETECTION_TABLE} AS d ON {on}")
        if where:
            adql += " WHERE " + " AND ".join(where)
        adql += (" GROUP BY ss.ssobjectid "
                 f"HAVING COUNT(*) >= {int(min_detections)}")
        try:
            res.rows = self.query(adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "OK" if res.rows else "EMPTY"
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"summary query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def max_available_mjd(self, join_on: str = "measurement_id") -> float | None:
        """Newest solar-system detection epoch the mirror actually holds.

        ALeRCE's LSST mirror lags: TOCSIN measured 15.6 days on 2026-07-30.  A
        screen that always asks for "the last two nights" would return nothing
        every night forever and look like a clean null, so every window in this
        channel is anchored to this value rather than to the wall clock.
        """
        on = ("d.measurement_id = ss.diasourceid" if join_on == "measurement_id"
              else "d.oid = ss.ssobjectid")
        try:
            rows = self.query(
                f"SELECT MAX(d.mjd) AS mjd_max FROM {SS_DETECTION_TABLE} AS ss "
                f"JOIN {DETECTION_TABLE} AS d ON {on}", maxrec=5, retries=2)
        except BrokerError:
            return None
        if not rows:
            return None
        try:
            v = float(rows[0].get("mjd_max"))
        except (TypeError, ValueError):
            return None
        return v if v == v else None
