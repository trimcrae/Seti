"""Offline tests for the SEXTANT acquisition layer.

No network, per ``docs/channel-brief.md`` §0 and the same rule every other channel
here runs under: the sandbox has no egress to ``gea.esac.esa.int``, so the live
queries are exercised on a GitHub Actions runner by
``.github/workflows/sextant-probe.yml`` and everything else is exercised here.

The suite is organised around the ways this layer could hand the next stage a
plausible, publishable, **wrong** answer:

* **a pull that is silently incomplete** — a truncated query, a chunk plan with a
  gap or an overlap, a bulk pull that came back at exactly the row limit.  Every
  one of those looks like a smaller catalogue rather than like an error, so
  ``test_plan_object_chunks_tiles_the_object_list_exactly_once``,
  ``test_iter_observation_chunks_splits_an_overflowing_chunk`` and
  ``test_fetch_flags_a_result_that_hits_the_row_limit`` are load-bearing;
* **a pull that is missing a column the residual needs** —
  ``position_angle_scan`` is the along-scan axis and ``x_gaia..vz_gaia`` are the
  observer's own state.  Neither is reconstructible downstream, so a pull without
  them must fail here.  ``test_column_inventory_reproduces_the_measured_counts``
  is the regression that catches someone trimming the column tuple: the arithmetic
  32 + 3 = 35 and 32 + 2 = 34 reproduces the live column counts measured on
  2026-08-25, and stops matching the moment a column is dropped;
* **an unverified assumption applied silently** — every entry in
  ``OPEN_QUESTIONS`` must be asked by the probe
  (``test_every_open_question_is_measured_by_the_probe``), and the quality helper
  must say when it applied no flag cut rather than let a caller believe the data
  is filtered;
* **the two releases reconciled by guesswork** — the DR3/FPR relationship is not
  settled, and ``reconcile_observations`` must refuse rather than produce a union
  that double-counts every shared observation.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from seti.sextant import acquire
from seti.sextant.acquire import (
    ALL_COLUMNS,
    DR3_OBSERVATION,
    DR3_ONLY_COLUMNS,
    FPR_OBSERVATION,
    FPR_ONLY_COLUMNS,
    MEASURED_COLUMNS,
    MEASURED_ROWS,
    OPEN_QUESTIONS,
    PUBLISHED_OUTLIER_FRACTION,
    RECORD_KEYS_FROM_PROBE,
    REQUIRED_FOR_REJECTION_SCREEN,
    REQUIRED_FOR_RESIDUALS,
    SHARED_COLUMNS,
    Chunk,
    GaiaSSO,
    QualityCuts,
    apply_quality_cuts,
    check_columns_for_rejection_screen,
    check_columns_for_residuals,
    chunks_cover,
    dedup_key,
    interpret_epoch_zero_point,
    interpret_rejection_fraction,
    interpret_rows_per_transit,
    interpret_state_vector_frame,
    interpret_sync_row_cap,
    partition_by_rejection,
    plan_epoch_chunks,
    plan_object_chunks,
    reconcile_observations,
    rejection_ledger,
    summarise_quality_values,
    transit_groups,
)
from seti.tocsin.brokers import BrokerError


# ---------------------------------------------------------------------------
# Offline stand-ins for the service
# ---------------------------------------------------------------------------
class _FakeTable:
    """The minimum of an ``astropy`` table that :meth:`GaiaSSO.fetch` touches."""

    def __init__(self, rows):
        self._rows = list(rows)
        self.colnames = list(self._rows[0].keys()) if self._rows else []

    def __iter__(self):
        return iter(self._rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def to_table(self):
        return _FakeTable(self._rows)


class _FakeService:
    """A ``pyvo`` TAPService stand-in: records queries, returns canned rows."""

    def __init__(self, handler):
        self.handler = handler
        self.queries: list[tuple[str, int]] = []
        self._session = None

    def run_async(self, adql, maxrec=None):
        self.queries.append((adql, maxrec))
        return _FakeResult(self.handler(adql, maxrec))


def _client(handler=lambda adql, maxrec: []) -> GaiaSSO:
    """A :class:`GaiaSSO` wired to a fake service.  No pyvo, no network."""
    c = GaiaSSO(url="http://stub/tap", timeout=1.0, maxrec=1000)
    svc = _FakeService(handler)
    c.tap._service = lambda: svc          # noqa: SLF001 - that is the seam
    c._fake = svc
    return c


def _obs_row(**kw) -> dict:
    """One observation row carrying every column the residual stage needs."""
    row = {c: 0.0 for c in REQUIRED_FOR_RESIDUALS}
    row.update({
        "number_mp": 1, "observation_id": 1, "transit_id": 1,
        "ra": 10.0, "dec": 5.0,
        "ra_error_random": 0.4, "ra_error_systematic": 0.1,
        "dec_error_random": 0.4, "dec_error_systematic": 0.1,
        "epoch": 1000.0, "epoch_utc": 2456197.5,
        "position_angle_scan": 42.0,
        "x_gaia": 0.99, "y_gaia": 0.01, "z_gaia": 0.0001,
        "vx_gaia": 0.0, "vy_gaia": 0.017, "vz_gaia": 0.0,
    })
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# The column inventory: the regression that stops a column being dropped
# ---------------------------------------------------------------------------
def test_column_inventory_reproduces_the_measured_counts():
    """32 shared + 3 DR3-only = 35, and 32 + 2 FPR-only = 34.

    Those two totals were measured live on 2026-08-25.  They are the only offline
    evidence available that :data:`SHARED_COLUMNS` is *complete*, so if anyone
    trims the tuple — say to save bandwidth on a 46-million-row pull — the
    arithmetic stops matching and this fails, instead of the residual computation
    quietly losing the observer state vector.
    """
    assert len(SHARED_COLUMNS) == 32
    assert len(set(SHARED_COLUMNS)) == len(SHARED_COLUMNS)
    assert len(SHARED_COLUMNS) + len(DR3_ONLY_COLUMNS) == MEASURED_COLUMNS[DR3_OBSERVATION]
    assert len(SHARED_COLUMNS) + len(FPR_ONLY_COLUMNS) == MEASURED_COLUMNS[FPR_OBSERVATION]
    assert set(ALL_COLUMNS["gaiadr3"]) == set(SHARED_COLUMNS) | set(DR3_ONLY_COLUMNS)
    assert set(ALL_COLUMNS["gaiafpr"]) == set(SHARED_COLUMNS) | set(FPR_ONLY_COLUMNS)


def test_the_axis_and_the_observer_state_are_carried_and_required():
    """``position_angle_scan`` and the state vectors are non-negotiable.

    Gaia's precision is anisotropic, so every residual must be projected onto the
    scan direction; and the light-time-corrected prediction needs the observer's
    own position.  Neither can be reconstructed by the next stage.
    """
    assert "position_angle_scan" in SHARED_COLUMNS
    for c in ("x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia"):
        assert c in SHARED_COLUMNS
        assert c in REQUIRED_FOR_RESIDUALS
    for c in ("x_gaia_geocentric", "vz_gaia_geocentric"):
        assert c in SHARED_COLUMNS
    # the error model stays in its two separable parts
    for c in ("ra_error_random", "ra_error_systematic",
              "dec_error_random", "dec_error_systematic",
              "ra_dec_correlation_random", "ra_dec_correlation_systematic"):
        assert c in SHARED_COLUMNS
    assert set(REQUIRED_FOR_RESIDUALS) <= set(SHARED_COLUMNS)


def test_check_columns_for_residuals_rejects_a_pull_missing_the_scan_angle():
    ok = check_columns_for_residuals([_obs_row()])
    assert ok["ok"] and ok["verdict"] == "OK"

    trimmed = {k: v for k, v in _obs_row().items() if k != "position_angle_scan"}
    bad = check_columns_for_residuals([trimmed])
    assert not bad["ok"]
    assert bad["verdict"] == "MISSING_REQUIRED_COLUMNS"
    assert "position_angle_scan" in bad["missing"]

    # also accepts a bare column list, which is how the probe checks the live schema
    assert check_columns_for_residuals(list(SHARED_COLUMNS))["ok"]
    assert not check_columns_for_residuals([])["ok"]


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------
def test_plan_object_chunks_tiles_the_object_list_exactly_once():
    """No gaps, no overlaps: the property that makes a bulk pull believable.

    A gap loses observations and reads as a sparser catalogue; an overlap
    double-counts them and reads as a smaller residual scatter.  Both survive every
    downstream test, so they are checked here.
    """
    numbers = [1, 2, 3, 8, 13, 21, 34, 55, 89, 144, 233, 610, 987]
    chunks = plan_object_chunks(numbers, target_objects=4)
    assert len(chunks) == 4
    cover = chunks_cover(chunks, numbers)
    assert cover["verdict"] == "COVERS_EXACTLY_ONCE"
    assert cover["n_missing"] == 0 and cover["n_duplicated"] == 0
    # boundaries abut exactly, so nothing can fall between two chunks
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.hi == b.lo
    # the largest object is inside the last half-open interval
    assert chunks[-1].hi > max(numbers)
    # each chunk holds the requested number of objects (the last may be short)
    for c in chunks[:-1]:
        assert sum(1 for n in numbers if c.lo <= n < c.hi) == 4


def test_plan_object_chunks_follows_density_not_a_stride():
    """Boundaries come from the object list, not from a stride in the index.

    Gaia's sample is concentrated in the low-numbered bright asteroids while MPC
    numbers run past 600,000.  A fixed stride would put almost every object in the
    first chunk and none in the last; taking boundaries from the sorted list makes
    the object count per chunk exact whatever the density does.
    """
    numbers = list(range(1, 101)) + [500_000, 600_000]
    chunks = plan_object_chunks(numbers, target_objects=51)
    assert len(chunks) == 2
    for c in chunks:
        assert sum(1 for n in numbers if c.lo <= n < c.hi) == 51
    assert chunks_cover(chunks, numbers)["complete"]
    # the expected row count is carried so a runner can size the pull
    assert chunks[0].expected_rows > 0


def test_plan_object_chunks_handles_degenerate_input():
    assert plan_object_chunks([]) == []
    assert len(plan_object_chunks([7], target_objects=100)) == 1
    # duplicates and Nones do not create phantom chunks
    assert len(plan_object_chunks([5, 5, None, 5], target_objects=1)) == 1


def test_plan_epoch_chunks_covers_the_last_observation():
    """The final upper bound is nudged past the end of a half-open tiling."""
    chunks = plan_epoch_chunks(0.0, 100.0, n_chunks=4)
    assert len(chunks) == 4
    assert chunks[-1].hi > 100.0
    assert chunks_cover(chunks, [0.0, 25.0, 50.0, 99.999, 100.0])["complete"]
    by_width = plan_epoch_chunks(0.0, 100.0, width_days=30.0)
    assert len(by_width) == 4
    assert plan_epoch_chunks(10.0, 10.0) == []


def test_chunks_cover_names_gaps_and_overlaps():
    gapped = [Chunk("number_mp", 0, 5), Chunk("number_mp", 6, 10)]
    assert chunks_cover(gapped, [1, 5, 7])["verdict"] == "GAPS"
    overlapping = [Chunk("number_mp", 0, 8), Chunk("number_mp", 4, 10)]
    assert chunks_cover(overlapping, [1, 5, 9])["verdict"] == "OVERLAPS"


def test_chunk_bisection_terminates():
    c = Chunk("number_mp", 10, 11)
    assert c.halves() is None                      # a single object cannot split
    lo, hi = Chunk("number_mp", 0, 100).halves()
    assert (lo.lo, lo.hi, hi.lo, hi.hi) == (0, 50, 50, 100)
    assert lo.depth == 1
    e_lo, e_hi = Chunk("epoch", 0.0, 1.0).halves()
    assert e_lo.hi == e_hi.lo == 0.5
    assert Chunk("epoch", 1.0, 1.0).halves() is None
    assert "number_mp >= 0" in Chunk("number_mp", 0, 50).where()
    assert "o.number_mp" in Chunk("number_mp", 0, 50).where(alias="o")


# ---------------------------------------------------------------------------
# The query paths
# ---------------------------------------------------------------------------
def test_fetch_flags_a_result_that_hits_the_row_limit():
    """A query that returns exactly ``maxrec`` rows is reported truncated.

    Deliberately conservative.  One redundant sub-query costs nothing; an
    unnoticed truncation gives a partial catalogue that reports success.
    """
    c = _client(lambda adql, maxrec: [{"n": i} for i in range(maxrec)])
    rows, truncated = c.fetch("SELECT n FROM t", maxrec=10)
    assert len(rows) == 10 and truncated is True
    assert c.last_truncated is True

    c2 = _client(lambda adql, maxrec: [{"n": 1}])
    rows, truncated = c2.fetch("SELECT n FROM t", maxrec=10)
    assert len(rows) == 1 and truncated is False


def test_fetch_fails_fast_on_an_adql_error_and_retries_transport_errors():
    """An ADQL error fails identically every time; retrying it burns the budget."""
    def bad_adql(adql, maxrec):
        raise RuntimeError("Unknown column 'position_angle_scam'")

    c = _client(bad_adql)
    with pytest.raises(BrokerError):
        c.fetch("SELECT position_angle_scam FROM t", maxrec=5, retries=4)
    assert len(c._fake.queries) == 1          # not retried

    state = {"n": 0}

    def flaky(adql, maxrec):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("connection reset by peer")
        return [{"n": 1}]

    c2 = _client(flaky)
    rows, _ = c2.fetch("SELECT n FROM t", maxrec=5, retries=3)
    assert rows == [{"n": 1}] and state["n"] == 2


def test_fetch_normalises_nulls_and_byte_strings():
    """Masked NULLs become ``None`` and ``bytes`` denominations become ``str``.

    A masked value that leaks through crashes JSON serialisation; a ``bytes``
    denomination never matches the ``str`` designations the JPL side uses, so the
    join would silently find no object at all.
    """
    def handler(adql, maxrec):
        return [{"denomination": b" Ceres ", "number_mp": np.int64(1),
                 "ra": np.float64(10.0), "epoch": np.ma.masked,
                 "is_rejected": np.bool_(False)}]

    rows, _ = _client(handler).fetch("SELECT * FROM t", maxrec=5)
    assert rows[0]["denomination"] == "Ceres"
    assert rows[0]["number_mp"] == 1 and isinstance(rows[0]["number_mp"], int)
    assert rows[0]["epoch"] is None
    assert rows[0]["is_rejected"] is False
    json.dumps(rows)          # must be serialisable, with no NaN and no numpy


def test_observations_for_objects_refuses_to_pull_the_whole_table():
    res = _client().observations_for_objects(release="gaiafpr")
    assert res.verdict == "NO_TARGETS" and res.rows == []


def test_observations_for_objects_ors_number_and_denomination():
    """An object present under one identifier and absent under the other must
    still come back; an AND would return nothing for exactly those objects."""
    c = _client()
    c.observations_for_objects(numbers=[1, 2], denominations=["1998 SH2"],
                               release="gaiafpr")
    adql = c._fake.queries[-1][0]
    assert "number_mp IN (1, 2)" in adql
    assert "denomination IN ('1998 SH2')" in adql
    assert " OR " in adql
    assert "position_angle_scan" in adql and "x_gaia" in adql
    assert "is_rejected" in adql              # the FPR-only column comes through


def test_string_literals_are_escaped():
    c = _client()
    c.observations_for_objects(denominations=["O'Brien"], release="gaiafpr")
    assert "'O''Brien'" in c._fake.queries[-1][0]


def test_iter_observation_chunks_splits_an_overflowing_chunk():
    """A chunk that hits the row limit is bisected, not accepted.

    The plan cannot be trusted to be correctly sized, because the density of the
    sample in ``number_mp`` is wildly non-uniform.  An overflowing chunk is short,
    not empty, and nothing downstream could tell.
    """
    def handler(adql, maxrec):
        # everything below 50 is dense enough to overflow a 4-row limit
        lo = float(adql.split("number_mp >= ")[1].split(" ")[0])
        hi = float(adql.split("number_mp < ")[1].split(" ")[0])
        n = 8 if (lo < 50 and hi <= 100) else 1
        if hi - lo <= 25:
            n = 2
        return [{"number_mp": lo, "i": i} for i in range(n)]

    c = _client(handler)
    results = list(c.iter_observation_chunks([Chunk("number_mp", 0, 100)],
                                             release="gaiafpr", maxrec=4))
    verdicts = [r.verdict for r in results]
    assert "SPLIT" in verdicts
    assert results[0].truncated is True
    # every terminal chunk is complete, and the split halves tile the original
    terminal = [r for r in results if r.verdict in ("OK", "EMPTY")]
    assert terminal
    assert chunks_cover([r.chunk for r in terminal], [0, 10, 40, 60, 99])["complete"]


def test_iter_observation_chunks_reports_irreducible_overflow():
    """A single object that still overflows is named, not silently truncated."""
    c = _client(lambda adql, maxrec: [{"i": i} for i in range(maxrec)])
    results = list(c.iter_observation_chunks([Chunk("number_mp", 7, 8)],
                                             release="gaiafpr", maxrec=3))
    assert results[-1].verdict == "IRREDUCIBLE_OVERFLOW"
    assert "epoch_lo" in " ".join(results[-1].notes)


def test_iter_observation_chunks_survives_a_failed_chunk():
    def handler(adql, maxrec):
        raise RuntimeError("connection reset by peer")

    c = _client(handler)
    results = list(c.iter_observation_chunks([Chunk("number_mp", 0, 10)],
                                             release="gaiafpr", maxrec=5))
    assert results[0].verdict == "NO_DATA_REACHED"


def test_object_numbers_marks_a_truncated_catalogue():
    c = _client(lambda adql, maxrec: [{"number_mp": i} for i in range(maxrec)])
    res = c.object_numbers(release="gaiafpr", maxrec=6)
    assert res.verdict == "TRUNCATED"
    assert "skip objects silently" in " ".join(res.notes)


# ---------------------------------------------------------------------------
# describe() and the schema record
# ---------------------------------------------------------------------------
def _schema_rows():
    rows = []
    for table, release in ((FPR_OBSERVATION, "gaiafpr"),
                           (DR3_OBSERVATION, "gaiadr3")):
        for col in ALL_COLUMNS[release]:
            rows.append({"table_name": table, "column_name": col,
                         "datatype": "double", "unit": "deg", "ucd": "pos.eq.ra",
                         "description": f"the {col} column"})
    return rows


def test_describe_returns_units_and_ucds_per_table():
    """The probe commits units and descriptions, not just names.

    Several open questions — the epoch's time scale, whether ``ra_error_*``
    carries ``cos(dec)``, the frame of the state vectors — may simply be answered
    by the archive's own column metadata, and if they are, the answer belongs in
    the committed record.
    """
    c = _client(lambda adql, maxrec: _schema_rows())
    out = c.describe(tables=(FPR_OBSERVATION, DR3_OBSERVATION))
    assert len(out[FPR_OBSERVATION]) == MEASURED_COLUMNS[FPR_OBSERVATION]
    assert len(out[DR3_OBSERVATION]) == MEASURED_COLUMNS[DR3_OBSERVATION]
    assert {"name", "datatype", "unit", "ucd", "description"} <= set(
        out[FPR_OBSERVATION][0])
    names = c.describe_names(tables=(FPR_OBSERVATION,))
    assert "position_angle_scan" in names[FPR_OBSERVATION]


# ---------------------------------------------------------------------------
# The probe: every open question is actually asked
# ---------------------------------------------------------------------------
def _diagnostic_names() -> set[str]:
    """Build every diagnostic query offline and collect the keys it produces."""
    c = _client()
    return set(c.diagnostics(sample_rows=3).keys())


def test_every_open_question_is_measured_by_the_probe():
    """A question cannot be recorded as unverified and then quietly never asked.

    :data:`OPEN_QUESTIONS` is the module's statement of what it does not know.
    This checks that each entry names a key the probe really produces — either a
    diagnostic query or a top-level field of the record — so the committed probe
    record answers the questions rather than only listing them.
    """
    produced = _diagnostic_names() | set(RECORD_KEYS_FROM_PROBE)
    for name, q in OPEN_QUESTIONS.items():
        assert {"question", "assumed", "why_it_matters", "probe_key"} <= set(q), name
        assert q["probe_key"] in produced, (
            f"open question {name!r} names probe_key {q['probe_key']!r}, which the "
            f"probe does not produce")


def test_every_diagnostic_query_is_well_formed():
    """Smoke-test every ADQL string the probe will send, offline.

    A malformed query costs a runner round trip to discover, and the probe sends
    dozens; balanced quotes and a real FROM clause are checkable here for free.
    """
    c = _client()
    out = c.diagnostics(sample_rows=3)
    n_queries = 0
    for name, entry in out.items():
        adql = entry.get("adql")
        if adql is None:
            assert "see" in entry, name          # a roll-up entry, not a query
            continue
        n_queries += 1
        assert adql.upper().startswith("SELECT"), name
        assert " FROM " in adql.upper(), name
        assert adql.count("'") % 2 == 0, name
        assert adql.count("(") == adql.count(")"), name
    assert n_queries > 25


def test_the_probe_samples_a_real_slice_with_every_column():
    """The verbatim slice must carry the whole column set: TAP_SCHEMA can list a
    column that no row populates, and only rows settle that."""
    c = _client()
    out = c.diagnostics(sample_rows=5)
    adql = out["sample_rows_gaiafpr"]["adql"]
    for col in ALL_COLUMNS["gaiafpr"]:
        assert col in adql
    assert check_columns_for_residuals(
        [x.strip() for x in adql.split("SELECT")[1].split("FROM")[0].split(",")]
    )["ok"]


def test_the_sync_cap_is_measured_synchronously():
    """The synchronous row cap can only be measured with a synchronous query."""
    c = _client()
    out = c.diagnostics(sample_rows=3)
    assert out["sync_row_cap"]["sync"] is True
    assert all(v.get("sync") is False for k, v in out.items()
               if k != "sync_row_cap" and "sync" in v)


# ---------------------------------------------------------------------------
# Reading the probe's measurements
# ---------------------------------------------------------------------------
def test_interpret_sync_row_cap():
    got = interpret_sync_row_cap(2000, 5000)
    assert got["verdict"] == "SYNC_CAP_MEASURED" and got["cap"] == 2000
    assert "asynchronous" in got["note"]
    assert interpret_sync_row_cap(5000, 5000)["verdict"] == "NO_CAP_AT_THIS_SIZE"


def test_interpret_state_vector_frame_separates_ecliptic_from_equatorial():
    """Gaia sits at L2, so barycentric ``z`` is the discriminant.

    In an ecliptic frame it never leaves a thin slab about zero; in an equatorial
    frame the 23.44 deg obliquity sweeps it +/-0.4 AU once a year.  Two orders of
    magnitude apart, so there is no grey zone worth arguing about — and it matters
    because a frame error rotates along-scan into across-scan.
    """
    ecl = interpret_state_vector_frame(-0.003, 0.004)
    assert ecl["verdict"] == "ECLIPTIC_LIKE"
    eq = interpret_state_vector_frame(-0.40, 0.41)
    assert eq["verdict"] == "EQUATORIAL_LIKE"
    assert interpret_state_vector_frame(-0.1, 0.1)["verdict"] == "AMBIGUOUS"
    assert interpret_state_vector_frame(None, None)["verdict"] == "NOT_MEASURED"
    # the physical scale: sin(23.44 deg) x 1 au really is ~0.4
    assert math.sin(math.radians(23.44)) == pytest.approx(0.3977, abs=1e-3)


def test_interpret_epoch_zero_point_reads_a_constant_offset():
    row = {"epoch_min": 500.0, "epoch_max": 2000.0,
           "diff_min": acquire.JD_2010_0, "diff_max": acquire.JD_2010_0 + 1e-9}
    got = interpret_epoch_zero_point(row)
    assert got["verdict"] == "CONSTANT_OFFSET"
    assert "2010-01-01" in got["note"]
    assert got["mjd_if_2010_zero_point"][0] == pytest.approx(acquire.MJD_2010_0 + 500.0)


def test_interpret_epoch_zero_point_flags_a_drifting_offset():
    """A drift means the two columns are in different TIME SCALES.

    TCB leads TDB by ~10 s in the Gaia era, and a main-belt asteroid at 30
    arcsec/day covers 3.5 mas in 10 s — a coherent along-track shift across the
    whole catalogue, which is indistinguishable from the signal being searched for.
    """
    row = {"epoch_min": 0.0, "epoch_max": 2000.0,
           "diff_min": 2455197.5, "diff_max": 2455197.5 + 10.0 / 86400.0}
    got = interpret_epoch_zero_point(row)
    assert got["verdict"] == "DRIFTING_OFFSET"
    # 1e-3 s, not 1e-6: a double at Julian-Date magnitude has a spacing of
    # ~4.7e-10 d = 40 microseconds, so 2455197.5 + 10/86400 is not exact. That
    # floor is four orders of magnitude below the 0.086 s that matters here, so
    # the measurement is not precision-limited — but the test must not pretend
    # the arithmetic is exact.
    assert got["offset_spread_seconds"] == pytest.approx(10.0, abs=1e-3)
    assert "DIFFERENT TIME SCALES" in got["note"]
    assert interpret_epoch_zero_point({})["verdict"] == "NOT_MEASURED"


def test_interpret_epoch_zero_point_refuses_an_unrecognised_offset():
    row = {"diff_min": 12345.0, "diff_max": 12345.0}
    got = interpret_epoch_zero_point(row)
    assert got["verdict"] == "CONSTANT_OFFSET"
    assert "DO NOT assume" in got["note"]


def test_interpret_rows_per_transit_quantifies_the_independence_error():
    """Nine CCD rows per transit are not nine independent measurements.

    They share one attitude solution and one scan angle, so treating them as
    independent understates the transit error by sqrt(N) — three-fold at N = 9,
    which is the arithmetic that manufactured LOOM's first 150 false anomalies.
    """
    rows = [{"n_rows": 9}] * 8 + [{"n_rows": 8}] * 2
    got = interpret_rows_per_transit(rows)
    assert got["verdict"] == "MEASURED"
    assert got["max"] == 9 and got["min"] == 8
    assert got["mean"] == pytest.approx(8.8)
    assert got["sigma_inflation_if_treated_independently"] == pytest.approx(
        math.sqrt(8.8))
    assert got["histogram"]["9"] == 8
    assert interpret_rows_per_transit([])["verdict"] == "NOT_MEASURED"


# ---------------------------------------------------------------------------
# Quality cuts
# ---------------------------------------------------------------------------
def test_a_string_false_is_not_a_rejection():
    """``bool('false')`` is ``True``.

    A naive ``is_rejected`` cut on a column that comes back as the string
    ``'false'`` would discard every FPR observation while looking like a working
    quality cut.  The type of the column is unverified, so the string case is
    decided by content.
    """
    assert acquire._truthy("false") is False
    assert acquire._truthy("False") is False
    assert acquire._truthy("true") is True
    assert acquire._truthy(0) is False
    assert acquire._truthy(1) is True
    assert acquire._truthy(np.bool_(True)) is True
    assert acquire._truthy(None) is False
    # an unparsed value is treated as NOT rejected: deleting data on a flag we
    # could not read is the more damaging error
    assert acquire._truthy("maybe") is False

    rows = [_obs_row(is_rejected="false"), _obs_row(is_rejected="true")]
    kept, report = apply_quality_cuts(rows, QualityCuts(drop_rejected=True))
    assert len(kept) == 1
    assert report["removed_by_rule"] == {"is_rejected": 1}


def test_no_outcome_flag_cut_is_applied_by_default_and_the_report_says_so():
    """A quality helper that silently does nothing is worse than none at all.

    The meanings of ``astrometric_outcome_ccd`` and ``astrometric_outcome_transit``
    are unverified, and a cut on a guessed good-value either removes a few per cent
    or removes 99% — both of which run to completion and report a number.  So no
    cut is applied by default and the report says which rules ran.
    """
    rows = [_obs_row(astrometric_outcome_ccd=v) for v in (0, 0, 1, 5, 5)]
    kept, report = apply_quality_cuts(rows)
    assert len(kept) == 5
    assert report["verdict"] == "NO_OUTCOME_FLAG_CUT_APPLIED_MEANINGS_UNVERIFIED"
    assert "unverified" in report["note"].lower()
    assert report["values_seen"]["astrometric_outcome_ccd"]["0"] == 2


def test_an_explicit_outcome_cut_is_applied_and_reported_per_rule():
    rows = [_obs_row(astrometric_outcome_ccd=v) for v in (0, 0, 1, 5, 5)]
    kept, report = apply_quality_cuts(
        rows, QualityCuts(keep_outcome_ccd=(acquire.ASSUMED_GOOD_OUTCOME,)))
    assert len(kept) == 2
    assert report["verdict"] == "OUTCOME_FLAG_CUT_APPLIED"
    assert report["removed_by_rule"]["astrometric_outcome_ccd"] == 3
    assert report["n_in"] == 5 and report["n_kept"] == 2
    assert report["cuts"]["keep_outcome_ccd"] == [0]


def test_an_observation_without_a_scan_angle_cannot_be_used():
    """No scan angle means no projection onto the axis that carries the precision."""
    rows = [_obs_row(), _obs_row(position_angle_scan=None),
            _obs_row(ra=float("nan"))]
    kept, report = apply_quality_cuts(rows)
    assert len(kept) == 1
    assert report["removed_by_rule"]["non_finite_position_angle_scan"] == 1
    assert report["removed_by_rule"]["non_finite_position"] == 1


def test_an_error_ceiling_can_be_applied():
    rows = [_obs_row(ra_error_random=0.4), _obs_row(ra_error_random=40.0)]
    kept, _ = apply_quality_cuts(rows, QualityCuts(max_error_mas=5.0))
    assert len(kept) == 1


def test_summarise_quality_values_only_reports_columns_that_exist():
    rows = [{"astrometric_outcome_ccd": 0}, {"astrometric_outcome_ccd": 1}]
    got = summarise_quality_values(rows)
    assert set(got) == {"astrometric_outcome_ccd"}
    assert got["astrometric_outcome_ccd"] == {"0": 1, "1": 1}


def test_transit_groups_expose_the_ccd_multiplicity():
    rows = [_obs_row(transit_id=t, observation_id=i)
            for i, t in enumerate([1, 1, 1, 2, 2])]
    g = transit_groups(rows)
    assert g["n_transits"] == 2 and g["n_rows"] == 5
    assert g["max_rows_per_transit"] == 3
    assert g["mean_rows_per_transit"] == pytest.approx(2.5)
    assert g["groups"][1] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Cross-release reconciliation
# ---------------------------------------------------------------------------
def _pair(obs_id, transit, epoch, number=1):
    return {"observation_id": obs_id, "transit_id": transit, "epoch": epoch,
            "number_mp": number, "ra": 1.0, "dec": 1.0}


def test_reconcile_deduplicates_a_genuine_overlap():
    dr3 = [_pair(1, 100, 10.0), _pair(2, 100, 10.0001)]
    fpr = [_pair(1, 100, 10.0), _pair(2, 100, 10.0001), _pair(3, 200, 50.0)]
    res = reconcile_observations(dr3, fpr, policy="prefer_fpr")
    assert res.verdict == "UNION_DEDUPLICATED"
    assert res.n == 3
    assert res.report["n_overlap"] == 2
    # FPR wins a collision: it is the later reduction
    assert all(r["_release"] == "gaiafpr" for r in res.rows)
    assert res.report["n_dr3_only"] == 0 and res.report["n_fpr_only"] == 1


def test_reconcile_refuses_a_union_when_the_key_is_disjoint_but_epochs_overlap():
    """The trap this whole helper exists for.

    If FPR re-minted ``observation_id`` then a key-based union counts every shared
    observation twice, halving the apparent scatter of every object and inflating
    every signal-to-noise by sqrt(2).  If the releases really are disjoint the
    union is right.  Those need opposite handling, so it refuses.
    """
    dr3 = [_pair(1, 100, 10.0), _pair(2, 101, 11.0)]
    fpr = [_pair(9001, 100, 10.0), _pair(9002, 101, 11.0)]
    res = reconcile_observations(dr3, fpr, policy="prefer_fpr")
    assert res.verdict == "KEY_DISJOINT_BUT_EPOCHS_OVERLAP"
    assert res.rows == []
    assert "DOUBLE-COUNTS" in " ".join(res.notes)
    assert res.report["epoch_ranges_overlap"] is True

    forced = reconcile_observations(dr3, fpr, policy="prefer_fpr",
                                    allow_unverified_union=True)
    assert forced.n == 4                       # the caller took responsibility


def test_the_transit_ccd_strategy_survives_reminted_ids():
    """Keying on the physical event is what a re-reduction cannot change.

    A transit crosses several CCDs about 4.4 s apart, so ``transit_id`` alone is
    not unique per row; the epoch rounded to 1e-6 d (0.086 s) separates them.
    """
    dr3 = [_pair(1, 100, 10.000000), _pair(2, 100, 10.000051)]
    fpr = [_pair(9001, 100, 10.000000), _pair(9002, 100, 10.000051),
           _pair(9003, 300, 90.0)]
    res = reconcile_observations(dr3, fpr, strategy="transit_ccd")
    assert res.verdict == "UNION_DEDUPLICATED"
    assert res.n == 3
    assert res.report["n_overlap"] == 2
    # and the two CCDs of one transit are NOT collapsed into each other
    assert res.report["within_release_key_collisions"] == {"gaiadr3": 0,
                                                           "gaiafpr": 0}


def test_reconcile_refuses_a_key_that_is_not_unique_within_a_release():
    """Deduplicating on a repeating key would delete real rows."""
    dr3 = [_pair(1, 100, 10.0), _pair(1, 100, 10.5)]
    fpr = [_pair(1, 100, 10.0)]
    res = reconcile_observations(dr3, fpr)
    assert res.verdict == "KEY_NOT_UNIQUE_WITHIN_RELEASE"
    assert res.rows == []
    assert "transit_ccd" in " ".join(res.notes)


def test_reconcile_single_release_policies_state_their_assumption():
    dr3 = [_pair(1, 100, 10.0)]
    fpr = [_pair(1, 100, 10.0), _pair(2, 200, 50.0)]
    only = reconcile_observations(dr3, fpr, policy="fpr_only")
    assert only.verdict == "FPR_ONLY" and only.n == 2
    assert "superset" in " ".join(only.notes)
    assert reconcile_observations(dr3, fpr, policy="dr3_only").n == 1
    with pytest.raises(ValueError):
        reconcile_observations(dr3, fpr, policy="whatever")


def test_reconcile_disjoint_epochs_concatenate_cleanly():
    dr3 = [_pair(1, 100, 10.0)]
    fpr = [_pair(2, 200, 900.0)]
    res = reconcile_observations(dr3, fpr)
    assert res.verdict == "UNION_NO_OVERLAP" and res.n == 2
    assert res.report["epoch_ranges_overlap"] is False


def test_reconcile_output_is_deterministic():
    """Two runs over the same rows produce the same order, so a diff means the
    data moved rather than the sort did."""
    dr3 = [_pair(i, 100 + i, 100.0 - i) for i in range(6)]
    fpr = [_pair(i, 100 + i, 100.0 - i) for i in range(3, 9)]
    a = reconcile_observations(dr3, fpr)
    b = reconcile_observations(list(reversed(dr3)), list(reversed(fpr)))
    assert [r["observation_id"] for r in a.rows] == \
           [r["observation_id"] for r in b.rows]
    assert [r["epoch"] for r in a.rows] == sorted(r["epoch"] for r in a.rows)


def test_dedup_key_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        dedup_key({"observation_id": 1}, strategy="vibes")


# ---------------------------------------------------------------------------
# The probe entry point
# ---------------------------------------------------------------------------
class _StubClient(GaiaSSO):
    """A GaiaSSO whose network calls are canned, for probing the probe."""

    def __init__(self, *a, fail: bool = False, **kw):
        super().__init__(url="http://stub/tap", timeout=1.0, maxrec=100)
        self.fail = fail

    def fetch(self, adql, maxrec=None, retries=4):
        if self.fail:
            raise BrokerError("no route to host")
        if "TAP_SCHEMA.columns" in adql:
            return _schema_rows(), False
        return [], False

    def query_sync(self, adql, maxrec=None, retries=3):
        return []

    def service_limits(self):
        return {"output_limits": {"default": [2000], "hard": [3000000]}}


def test_probe_writes_a_record_and_answers_the_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "GaiaSSO", _StubClient)
    rec = acquire.probe(out_dir=tmp_path)
    assert rec["verdict"] == "OK"
    written = json.loads((tmp_path / "probe.json").read_text())
    # the open questions travel with the record, so the committed file states what
    # was still unknown when it was written
    assert set(written["open_questions"]) == set(OPEN_QUESTIONS)
    assert written["expected"]["rows"][FPR_OBSERVATION] == MEASURED_ROWS[FPR_OBSERVATION]
    assert written["column_inventory"]["shared"] == list(SHARED_COLUMNS)
    # the live schema is checked against the columns this module selects
    check = written["column_check"][FPR_OBSERVATION]
    assert check["missing_from_live"] == []
    assert check["residuals_ok"]["ok"] is True
    # and the raw measurements are turned into verdicts
    assert {"sync_row_cap", "state_vector_frame", "epoch_zero_point",
            "rows_per_transit", "rejection_fraction",
            "rejection_ledger_sample"} <= set(written["answers"])
    assert "reconciliation_trial" in written


def test_probe_degrades_honestly_when_the_service_is_unreachable(tmp_path,
                                                                monkeypatch):
    """An unreachable archive writes NO_DATA_REACHED, never an empty result set.

    An empty candidate table reads as a clean null on a search that never ran.
    """
    monkeypatch.setattr(acquire, "GaiaSSO",
                        lambda *a, **kw: _StubClient(fail=True))
    rec = acquire.probe(out_dir=tmp_path)
    assert rec["verdict"] == "NO_DATA_REACHED"
    assert "schema_error" in rec
    assert (tmp_path / "probe.json").exists()


def test_probe_checkpoints_so_a_timeout_keeps_what_it_learned(tmp_path,
                                                             monkeypatch):
    """The record is written after every query.

    A cancelled GitHub Actions job never runs its commit step, so a run that
    overshoots loses everything it learned unless the file is already on disk.
    """
    seen: list[int] = []

    class _Counting(_StubClient):
        def diagnostics(self, on_result=None, sample_rows=20, **kw):
            for i in range(3):
                if on_result is not None:
                    on_result(f"q{i}", {"adql": "SELECT 1 FROM t", "rows": 0})
                seen.append((tmp_path / "probe.json").stat().st_size)
            return {}

    monkeypatch.setattr(acquire, "GaiaSSO", lambda *a, **kw: _Counting())
    acquire.probe(out_dir=tmp_path)
    assert len(seen) == 3
    assert all(s > 0 for s in seen)


# ---------------------------------------------------------------------------
# The rejection screen: the rejected rows are the signal
# ---------------------------------------------------------------------------
def test_no_data_pull_ever_filters_on_a_rejection_flag():
    """The load-bearing one for the re-aimed channel.

    Every published search over this dataset works post-fit, so an object that
    fails to fit is invisible to all of them.  SEXTANT screens on the rejection
    pattern instead, which is only possible if the rejected rows are actually
    fetched.  A data pull that quietly narrowed on ``is_rejected`` or on an
    outcome code would make the channel impossible **and the loss would be
    invisible**, because what came back would still look like a clean dataset.
    """
    c = _client()
    c.observations_for_objects(numbers=[1, 2], release="gaiafpr")
    c.observations_in_epoch_range(0.0, 100.0, release="gaiafpr")
    list(c.iter_observation_chunks([Chunk("number_mp", 0, 10)],
                                   release="gaiafpr", maxrec=100))
    assert c._fake.queries
    for adql, _ in c._fake.queries:
        where = adql.split(" WHERE ", 1)[1] if " WHERE " in adql else ""
        for flag in ("is_rejected", "astrometric_outcome_ccd",
                     "astrometric_outcome_transit"):
            assert flag not in where, f"{flag} narrows a data pull: {adql}"
        # but they are still SELECTed, so the caller can label the rows
        assert "astrometric_outcome_ccd" in adql


def test_quality_cuts_do_not_drop_rejected_rows_by_default():
    """The default reverses the obvious one, on purpose."""
    assert QualityCuts().drop_rejected is False
    rows = [_obs_row(is_rejected="true"), _obs_row(is_rejected="false")]
    kept, report = apply_quality_cuts(rows)
    assert len(kept) == 2
    assert "is_rejected" not in report["removed_by_rule"]
    # the flag values are still reported, because they are the observable
    assert report["values_seen"]["is_rejected"] == {"true": 1, "false": 1}


def test_partition_by_rejection_labels_instead_of_discarding():
    rows = [_obs_row(is_rejected="true"), _obs_row(is_rejected="false"),
            _obs_row(is_rejected="false"), _obs_row()]
    got = partition_by_rejection(rows)
    assert got["n_total"] == 4
    assert len(got["rejected"]) == 1 and len(got["fitted"]) == 2
    # an ABSENT flag is not a passed one -- every gaiadr3 row lands here
    assert len(got["unlabelled"]) == 1
    assert got["rejected_fraction"] == pytest.approx(1 / 3)


def test_rejection_census_groups_rather_than_filters():
    """The numerator and the denominator have to come back in one query.

    A query that selected only the failures would return a numerator with no
    denominator, and a rate cannot be recovered from that afterwards.
    """
    c = _client()
    c.rejection_census(numbers=[1, 2, 3], release="gaiafpr",
                       flag_column="is_rejected")
    adql = c._fake.queries[-1][0]
    assert "GROUP BY number_mp, is_rejected" in adql
    assert "COUNT(*)" in adql
    assert "WHERE number_mp IN (1, 2, 3)" in adql
    # the flag appears only in the SELECT/GROUP BY, never as a narrowing predicate
    assert "is_rejected =" not in adql and "is_rejected IS" not in adql

    c2 = _client()
    c2.rejection_census(release="gaiafpr", min_number=100, max_number=200)
    assert "number_mp >= 100" in c2._fake.queries[-1][0]


def test_rejection_census_marks_a_truncated_denominator():
    c = _client(lambda adql, maxrec: [{"number_mp": i, "flag_value": 0, "n": 1}
                                      for i in range(maxrec)])
    res = c.rejection_census(numbers=[1], release="gaiafpr", maxrec=4)
    assert res.verdict == "TRUNCATED"
    assert "wrong rather than noisy" in " ".join(res.notes)


def test_rejection_ledger_preserves_the_denominator():
    """Attempts are the sum over the flag values, and no rate is invented.

    Which codes count as a failure is unverified, so manufacturing a rate here
    would bake in the assumption the channel exists to test.
    """
    census = [
        {"number_mp": 1, "flag_value": 0, "n": 300, "flag_column": "outcome"},
        {"number_mp": 1, "flag_value": 3, "n": 12, "flag_column": "outcome"},
        {"number_mp": 2, "flag_value": 0, "n": 150, "flag_column": "outcome"},
    ]
    led = rejection_ledger(census)
    assert led[1]["attempts"] == 312
    assert led[1]["by_flag"]["outcome"] == {"0": 300, "3": 12}
    assert led[1]["modal_flag"] == "0"
    assert led[1]["n_flag_values"] == 2
    assert led[2]["attempts"] == 150
    # no rate is computed: which codes are failures is not known yet
    assert "rejected_fraction" not in led[1]
    assert rejection_ledger([]) == {}


def test_rejection_ledger_does_not_double_count_two_flag_columns():
    """Summing over two flag columns would count every observation twice."""
    census = [
        {"number_mp": 1, "flag_value": 0, "n": 100, "flag_column": "ccd"},
        {"number_mp": 1, "flag_value": 1, "n": 20, "flag_column": "ccd"},
        {"number_mp": 1, "flag_value": 0, "n": 120, "flag_column": "transit"},
    ]
    led = rejection_ledger(census)
    assert led[1]["attempts"] == 120           # one column's total, not 240
    assert set(led[1]["by_flag"]) == {"ccd", "transit"}


def test_interpret_rejection_fraction_calibrates_against_the_published_rate():
    """~0.58% (DR3) and ~1% (DR2) are what ``is_rejected`` should look like.

    Land near them and the column marks the documented outlier rejection, which
    puts the rejected set at 1e5-1e6 rows — fetchable, which is what makes the
    screen possible.  Land an order of magnitude away and every rate built on the
    column is a rate of something else.
    """
    n_total = MEASURED_ROWS[FPR_OBSERVATION]
    got = interpret_rejection_fraction(int(0.0058 * n_total), n_total)
    assert got["verdict"] == "CONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION"
    assert got["fraction"] == pytest.approx(0.0058, abs=1e-5)
    assert 1e5 < got["n_rejected"] < 1e6      # comfortably fetchable

    high = interpret_rejection_fraction(int(0.30 * n_total), n_total)
    assert high["verdict"] == "INCONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION"
    assert "do not read a rate" in high["note"]

    assert interpret_rejection_fraction(0, n_total)["verdict"] == \
        "ALMOST_NOTHING_REJECTED"
    assert interpret_rejection_fraction(1, 0)["verdict"] == "NOT_MEASURED"
    assert PUBLISHED_OUTLIER_FRACTION["gaiadr3"] == pytest.approx(0.0058)


def test_check_columns_for_rejection_screen_is_separate_from_the_residual_check():
    """A pull can be adequate for one path and useless for the other."""
    assert set(REQUIRED_FOR_REJECTION_SCREEN) <= set(SHARED_COLUMNS)
    fpr = check_columns_for_rejection_screen(list(ALL_COLUMNS["gaiafpr"]),
                                             release="gaiafpr")
    assert fpr["ok"] and fpr["has_is_rejected"] is True

    dr3 = check_columns_for_rejection_screen(list(ALL_COLUMNS["gaiadr3"]),
                                             release="gaiadr3")
    assert dr3["ok"] and dr3["has_is_rejected"] is False
    assert "outcome codes alone" in dr3["note"]

    trimmed = [c for c in ALL_COLUMNS["gaiafpr"]
               if c != "astrometric_outcome_ccd"]
    bad = check_columns_for_rejection_screen(trimmed, release="gaiafpr")
    assert bad["verdict"] == "MISSING_REQUIRED_COLUMNS"
    assert "is_rejected is absent" in check_columns_for_rejection_screen(
        [c for c in ALL_COLUMNS["gaiafpr"] if c != "is_rejected"],
        release="gaiafpr")["note"]


def test_the_probe_measures_the_rejection_fraction_both_ways():
    """The column's TYPE is unverified, so both predicates are tried.

    Whichever one the service accepts also settles whether ``is_rejected`` is a
    string or a number, which no amount of reading the schema from here could.
    """
    out = _client().diagnostics(sample_rows=3)
    assert "is_rejected = 'true'" in out["rejection_fraction_rejected"]["adql"]
    assert "is_rejected = 1" in out["rejection_fraction_rejected_numeric"]["adql"]
    assert out["rejection_fraction"]["published_reference"]["gaiadr3"] == \
        pytest.approx(0.0058)
    # and the per-object census sample is run for every flag column
    for col in ("astrometric_outcome_ccd", "astrometric_outcome_transit",
                "is_rejected"):
        assert f"GROUP BY number_mp, {col}" in out[f"rejection_census_{col}"]["adql"]
