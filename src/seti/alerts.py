"""Decide when a human needs to look, and say why.

The channels run unattended on GitHub's scheduler and commit their results back
to the repository.  That is the easy half.  The hard half is that a search which
reports every week is a search nobody reads, and a search which reports nothing
is indistinguishable from a search that has silently died.  This module is the
part that has to get both right.

Three kinds of alert, and they are genuinely different
------------------------------------------------------
``candidate``
    A channel promoted something.  This is what the whole apparatus exists for
    and it should be rare — if it fires monthly, the thresholds are wrong, not
    the sky.
``health``
    The pipeline is broken in a way that produces *no error*: a screen that
    reaches no data, a positive control the screen failed to recover, results
    that have stopped updating.  These matter more than they look, because every
    one of them turns into "a clean null" if nobody notices — which is the exact
    failure this repository is most exposed to.
``milestone``
    A capability came online.  The one that matters here: LOOM's central
    discriminant needs objects with two apparitions, and the Rubin survey is too
    young to have any.  When that changes, the channel starts being able to
    answer its own question, and that is worth knowing the week it happens
    rather than a year later.

Why alerts are deduplicated
---------------------------
Every alert carries a stable ``key``.  ``results/alerts/state.json`` remembers
the keys already raised, and only *new* keys fire.  Without this the first
promoted candidate would email once a week forever, and the notification would
be trained into noise within a month — at which point the channel has a working
detector and a human who ignores it.

Why nothing here reads like a result
------------------------------------
An alert is a request for attention, not a claim.  The body states what was
measured, what the channel's own verdict was, and which file to read.  It never
says a thing was found.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEVERITIES = ("candidate", "health", "milestone")

# How stale a channel's results may get before silence is treated as failure.
# Nightly channels get a wider window than their cadence so one missed firing is
# not an incident; a week of silence is.
STALE_DAYS = {"tocsin": 4.0, "loom": 10.0,
              # Channels opened while Rubin is off sky.  Each window is a little
              # over twice its cadence, so one missed firing is not an incident
              # and two are.  They are listed here for the same reason the Rubin
              # channels are: an unwatched cron that stops produces exactly the
              # empty directory a quiet sky produces.
              "loom-catalogue": 45.0,      # monthly, 2nd
              "tocsin_altfeeds": 16.0,     # weekly, Wednesdays
              "sextant": 45.0,             # monthly, 3rd
              # Weekly, Mondays.  Watched for the same reason as the rest, and
              # for one of its own: `outage_context` quotes this channel's
              # verdict inside the frontier alerts, so a channel that has quietly
              # stopped keeps an old cause attached to a live alert until the
              # frontier moves.  The expiry there is about the EPOCH the verdict
              # describes; this is about the channel that produces it.
              "rubin_outage": 16.0}

# Which file in a channel's directory carries its run stamp.  Not every channel
# is named `summary.json`/`screen.json`, and a marker that does not exist is
# skipped silently -- so a typo here disables the staleness check for that
# channel while leaving it looking configured.  Pinned by a test.
STALE_MARKER = {"tocsin": "summary.json",
                "loom": "screen.json",
                "loom-catalogue": "catalogue.json",
                "tocsin_altfeeds": "probe.json",
                "sextant": "probe.json",
                "rubin_outage": "brokers.json"}

# How far the DATA may fall behind the wall clock before that is a failure.
#
# This is a different question from "did the channel run", and it is the one
# that catches the failure both channels are most exposed to.  Both read Rubin
# through ALeRCE's public mirror, and that mirror lags: 15.6 days measured, and
# the frontier sat at MJD 61235 (2026-07-14) on 2026-07-30.  If ALeRCE simply
# stops ingesting LSST, both channels keep running on schedule, keep writing a
# fresh run stamp, and keep committing -- so every liveness check above stays
# green while the repository has silently stopped tracking Rubin at all.
#
# 30 days is roughly twice the measured lag, so ordinary mirror latency never
# fires and a mirror that has actually stopped shows up inside a fortnight.
DATA_LAG_LIMIT_DAYS = 30.0

# How long the frontier may sit at the SAME VALUE before that is a failure.
#
# The check above asks whether the data is old.  This asks whether it is
# MOVING, and the two come apart exactly when it matters.  A mirror that stops
# ingesting is caught by the age check only once the freeze has burned through
# the whole 30-day budget -- and because the mirror is already ~16 days behind
# when it stops, that is a fortnight of clean nulls, each of them "no new sky"
# being read as "clean sky".  Watching for the frontier to stop ADVANCING sees
# the same failure from the first week.
#
# WHY 7 DAYS AND NOT 2.  A threshold below the mirror's ingest cadence fires on
# ordinary batching, and an alert that fires on ordinary behaviour is noise
# inside a month -- the failure this module is built to avoid.  ALeRCE's
# cadence has not been measured here yet: the frontier sat at MJD 61235.41918
# unchanged across every run from 2026-07-30 to 2026-07-31, which is the only
# observation there is.  7 days is therefore deliberately conservative: it is
# above any plausible batch interval and still four times faster than the age
# check.  ``observed_advance_days`` in results/alerts/frontier.json accumulates
# the real cadence run by run; tighten this once that has an answer.
FRONTIER_STALL_DAYS = 7.0

# Where each channel reports the frontier, in preference order.
#
# The BROKER's frontier, not the SCREENED one, is what answers "has the data
# stopped".  They are equal while a channel is caught up, but they fail apart:
# if the channel breaks while the mirror keeps advancing, the screened frontier
# freezes and the data has not stopped at all.  Reading the broker's own number
# first keeps a channel bug from being reported as a mirror outage.
FRONTIER_KEYS = {
    "tocsin": (("summary.json", "broker_frontier_mjd"),
               ("ledger.json", "last_mjd_screened")),
    "loom": (("screen.json", "frontier_mjd"),),
}

# How many past frontier values to keep.  This is the record the stall
# threshold is meant to be calibrated against, so it is kept generously: at one
# advance per night it is most of a year, and the file stays a few kB.
FRONTIER_HISTORY_LIMIT = 300

MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass
class Alert:
    """One reason a human should look, with a stable identity for deduplication."""

    key: str
    severity: str
    channel: str
    title: str
    body: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                         # noqa: BLE001
        return None


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


# Every channel stamps the UTC time it ran INSIDE its result file.  That stamp
# is the only usable clock here: a CI runner clones the repository fresh, so
# every file's mtime is the checkout time and a channel dead for a year looks
# like it ran thirty seconds ago.  Checking staleness by mtime would therefore
# be silently, permanently disabled on exactly the machine it has to work on.
RUN_STAMP_KEYS = ("run_at_utc", "screened_at_utc", "assessed_at_utc",
                  "calibrated_at_utc", "checked_at_utc", "run_utc")


def _parse_stamp(value) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(
            timezone.utc)
    except ValueError:
        return None


def _age_days(path: Path, now: datetime) -> float:
    """Days since the result in ``path`` was *produced* (not since checkout)."""
    rec = _load(path)
    if isinstance(rec, dict):
        for key in RUN_STAMP_KEYS:
            stamp = _parse_stamp(rec.get(key))
            if stamp is not None:
                return (now - stamp).total_seconds() / 86400.0
    # No stamp in the file: fall back to mtime, which is right when running
    # against a working tree and useless after a fresh clone.  Returning inf
    # instead would alert on every unstamped file, so mtime is the safer miss.
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return float("inf")
    return (now - mtime).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# The data frontier, and whether it is moving
# ---------------------------------------------------------------------------
def outage_path(root: Path) -> Path:
    return root / "results" / "rubin_outage" / "brokers.json"


def outage_context(root: Path) -> str:
    """The established cause of a frozen frontier, or "" if none is established.

    Both frontier alerts used to end by asking the reader to "check whether
    ALeRCE is still ingesting the LSST alert stream" — one suspect out of two,
    and the wrong one as it turned out.  ``rubin-outage`` answers that question
    from a second broker; once it has, repeating the question wastes the reader's
    attention and points them at a service that is working.

    Deliberately narrow: it reports only what the cross-broker check decided, and
    only while that decision still describes the frontier the alert is about.  A
    verdict older than the epoch it explains is not evidence about the present,
    so it is dropped rather than restated — a stale explanation attached to a
    live alert is worse than no explanation, because it stops the reader looking.
    """
    rec = _load(outage_path(root))
    if not isinstance(rec, dict):
        return ""
    d = rec.get("decision") or {}
    verdict = str(d.get("verdict") or "")
    if verdict not in ("SKY_STOPPED", "MIRROR_STALLED"):
        return ""
    checked_mjd = _f(d.get("alerce_frontier_mjd"))
    # Is the verdict still about the frontier the alert is about?  If the mirror
    # has moved on since the check ran, the old verdict explains an epoch nobody
    # is looking at any more.
    live = [m for m, _ in current_frontiers(root).values()]
    if live and not any(abs(m - checked_mjd) <= 1.0 for m in live):
        return ""

    when = str(rec.get("checked_at_utc") or "")[:10]
    others = ", ".join(
        f"{k} {str(v.get('frontier_utc') or '')[:10]}"
        for k, v in (d.get("other_brokers") or {}).items()) or "none"
    if verdict == "SKY_STOPPED":
        return (f"\n\n**Cause established {when} by `rubin-outage`: "
                f"SKY_STOPPED.** Independent brokers stop on the same night "
                f"(ALeRCE {str(d.get('alerce_frontier_utc') or '')[:10]}; "
                f"{others}), so the alert stream itself stopped and no change to "
                f"the broker path recovers data that was never taken. Do not "
                f"relax the threshold on this — the condition is real. See "
                f"`docs/rubin-outage.md`.")
    return (f"\n\n**Cause established {when} by `rubin-outage`: "
            f"MIRROR_STALLED.** Another broker holds newer LSST epochs "
            f"({others}) than ALeRCE "
            f"({str(d.get('alerce_frontier_utc') or '')[:10]}), so real sky is "
            f"going unscreened. This is urgent: route the channels through a "
            f"current broker. See `docs/rubin-outage.md`.")


def frontier_path(root: Path) -> Path:
    return root / "results" / "alerts" / "frontier.json"


def current_frontiers(root: Path) -> dict[str, tuple[float, str]]:
    """The newest epoch each channel can see, and which file said so.

    Returns ``{channel: (mjd, source)}``, omitting any channel whose frontier is
    missing, unparseable or zero -- an unknown frontier is not a stopped one,
    and treating it as stopped would alert on every channel that has never run.
    """
    out: dict[str, tuple[float, str]] = {}
    for channel, sources in FRONTIER_KEYS.items():
        for name, key in sources:
            rec = _load(root / "results" / channel / name)
            if not isinstance(rec, dict):
                continue
            mjd = _f(rec.get(key))
            if math.isfinite(mjd) and mjd > 0:
                out[channel] = (mjd, f"{name}:{key}")
                break
    return out


def _advance_cadence(entry: dict) -> dict:
    """How often this channel's frontier has actually been seen to advance.

    This is the empirical answer that ``FRONTIER_STALL_DAYS`` is a placeholder
    for.  It measures the gap between the times successive frontier VALUES were
    first seen, which is the mirror's ingest cadence as observed from here --
    not the run cadence, and not the mirror's own claim about itself.
    """
    stamps = [_parse_stamp(h.get("first_seen_utc"))
              for h in (entry.get("history") or [])]
    stamps.append(_parse_stamp(entry.get("first_seen_utc")))
    stamps = [s for s in stamps if s is not None]
    # strict=False is the correct pairing, not a waiver: the two iterables are
    # deliberately of different length (n and n-1) because this walks
    # CONSECUTIVE PAIRS of one list.  strict=True would raise on every call.
    gaps = sorted((b - a).total_seconds() / 86400.0
                  for a, b in zip(stamps, stamps[1:], strict=False) if b > a)
    if not gaps:
        # One frontier value seen so far: no advance has been observed, so the
        # cadence is genuinely unknown.  Reporting a number here would invite
        # calibrating the threshold against nothing.
        return {"n_advances": 0}
    mid = len(gaps) // 2
    median = gaps[mid] if len(gaps) % 2 else 0.5 * (gaps[mid - 1] + gaps[mid])
    return {"n_advances": len(gaps), "median_days": round(median, 3),
            "max_days": round(gaps[-1], 3)}


def record_frontier(root: Path, now: datetime | None = None,
                    path: Path | None = None) -> dict:
    """Fold today's frontier into the history, and return the whole record.

    ``first_seen_utc`` is the load-bearing field: for the CURRENT value it is
    the last time the frontier was observed to move, which is the only clock a
    stall can be measured against.  It must therefore survive a run that sees
    no change -- which is why an unchanged frontier extends ``last_seen_utc``
    and leaves ``first_seen_utc`` alone.

    Called only on the recording pass.  A dry run that wrote here would stamp
    ``first_seen_utc`` at the moment of the check and reset the stall clock on
    every evaluation, so the alert could never fire.
    """
    now = now or datetime.now(timezone.utc)
    path = path or frontier_path(root)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    rec = _load(path) or {}
    channels = dict(rec.get("channels") or {})
    for channel, (mjd, source) in current_frontiers(root).items():
        prev = dict(channels.get(channel) or {})
        history = list(prev.get("history") or [])
        if _f(prev.get("mjd")) == mjd:
            # Same epoch as last time: another sighting of the same freeze.
            entry = dict(prev)
            entry["last_seen_utc"] = stamp
            entry["n_sightings"] = int(prev.get("n_sightings") or 1) + 1
        else:
            if prev.get("mjd") is not None:
                history.append({k: prev.get(k) for k in
                                ("mjd", "first_seen_utc", "last_seen_utc")})
            entry = {"mjd": mjd, "first_seen_utc": stamp, "last_seen_utc": stamp,
                     "n_sightings": 1}
        entry["source"] = source
        entry["history"] = history[-FRONTIER_HISTORY_LIMIT:]
        entry["observed_advance_days"] = _advance_cadence(entry)
        channels[channel] = entry

    out = {"channels": channels, "updated_utc": stamp,
           "stall_limit_days": FRONTIER_STALL_DAYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def frontier_status(root: Path, now: datetime | None = None) -> dict:
    """Per-channel {mjd, source, frozen_days, lag_days} for the run report."""
    now = now or datetime.now(timezone.utc)
    seen = (_load(frontier_path(root)) or {}).get("channels") or {}
    out: dict[str, dict] = {}
    for channel, (mjd, source) in current_frontiers(root).items():
        entry = seen.get(channel) or {}
        since = (_parse_stamp(entry.get("first_seen_utc"))
                 if _f(entry.get("mjd")) == mjd else None)
        out[channel] = {
            "mjd": mjd,
            "source": source,
            "lag_days": round((now - (MJD_EPOCH + timedelta(days=mjd)))
                              .total_seconds() / 86400.0, 3),
            "frozen_days": (None if since is None else
                            round((now - since).total_seconds() / 86400.0, 3)),
            "observed_advance_days": entry.get("observed_advance_days"),
        }
    return out


# ---------------------------------------------------------------------------
# The frontier starts moving again
# ---------------------------------------------------------------------------
def frontier_recovery_alerts(root: Path, now: datetime | None = None) -> list[Alert]:
    """The mirror has advanced again after a stall long enough to have alerted.

    The stall check answers "has the data stopped".  Nothing answered "has it
    STARTED", and the two are not symmetric.  A stall is visible in every run
    that follows it, so it can be noticed late and still be noticed.  A recovery
    is visible in EXACTLY ONE run -- the first whose frontier differs from the
    recorded one -- and after that the mirror simply looks healthy, as though it
    always had been.  Miss that run and the only trace is a number in
    frontier.json that quietly changed.

    Like the stall check this must be evaluated BEFORE ``record_frontier``
    folds today's sighting in, which is the order :func:`check` already
    guarantees; afterwards the old value is gone and there is nothing to compare
    against.

    AN ORDINARY ADVANCE IS DELIBERATELY NOT AN ALERT.  A healthy mirror advances
    every night, and notifying on every night's ingest is exactly the noise this
    module exists to avoid.  The gate is the same threshold the stall check
    uses, so a recovery fires if and only if the stall it ends had itself been
    worth reporting: this alert is the answer to a question already asked, never
    an unprompted one.
    """
    now = now or datetime.now(timezone.utc)
    out: list[Alert] = []
    seen = (_load(frontier_path(root)) or {}).get("channels") or {}
    for channel, (mjd, source) in current_frontiers(root).items():
        entry = seen.get(channel) or {}
        prev = _f(entry.get("mjd"))
        # A frontier never recorded has not been observed to move.  One that
        # went BACKWARDS is not a recovery either -- that is a broker
        # re-indexing or a channel reading a different table, and announcing it
        # as recovery would promise data that is not there.
        if not math.isfinite(prev) or mjd <= prev:
            continue
        first = _parse_stamp(entry.get("first_seen_utc"))
        if first is None:
            continue
        # Measured first-sighting to first-sighting, the same way
        # _advance_cadence measures it, so the number quoted to a human and the
        # number folded into the cadence record are the same number.
        interval = (now - first).total_seconds() / 86400.0
        if interval <= FRONTIER_STALL_DAYS:
            continue

        last = _parse_stamp(entry.get("last_seen_utc")) or first
        confirmed = (last - first).total_seconds() / 86400.0
        sky = mjd - prev
        lag = (now - (MJD_EPOCH + timedelta(days=mjd))).total_seconds() / 86400.0
        caught_up = lag <= DATA_LAG_LIMIT_DAYS
        out.append(Alert(
            key=f"{channel}:frontier_recovered:{mjd:.5f}",
            severity="milestone", channel=channel,
            title=f"{channel.upper()}'s data frontier is moving again "
                  f"(+{sky:.1f} days of sky)",
            body=(f"`{channel}` has advanced from MJD {prev:.5f} to MJD "
                  f"{mjd:.5f} — {sky:.1f} days of sky — after sitting still "
                  f"since {entry.get('first_seen_utc')}.\n\n"
                  f"The stall was confirmed across {confirmed:.1f} days of "
                  f"sightings; the advance itself happened between "
                  f"{entry.get('last_seen_utc')} and now, so the interval "
                  f"between successive frontier values is {interval:.1f} days. "
                  f"That is an UPPER BOUND on the mirror's true ingest gap — "
                  f"this repository samples the frontier about once a day and "
                  f"cannot see a change finer than that.\n\n"
                  + (f"The frontier is now {lag:.1f} days behind the wall "
                     f"clock, inside the {DATA_LAG_LIMIT_DAYS:.0f}-day limit, "
                     f"so the age alert clears. That is not the same as "
                     f"current: the mirror's ordinary lag is ~16 days, and "
                     f"anything above that is still backlog.\n\n"
                     if caught_up else
                     f"The frontier is still {lag:.1f} days behind the wall "
                     f"clock, past "
                     f"the {DATA_LAG_LIMIT_DAYS:.0f}-day limit. One advance is "
                     f"not a recovery: this may be a partial backfill, so do "
                     f"not read the next null as a statement about tonight's "
                     f"sky until the lag comes down.\n\n") +
                  f"This is the cadence datum `FRONTIER_STALL_DAYS` has never "
                  f"had — it was set to {FRONTIER_STALL_DAYS:.0f} by "
                  f"guesswork because no advance had ever been observed here. "
                  f"Recalibrate it against `observed_advance_days` in "
                  f"`results/alerts/frontier.json` now that there is one.\n\n"
                  f"Note that the nights lost to the stall are ABSENT, not "
                  f"null: they were never observed, so they add nothing to the "
                  f"screen's trial count and must not be counted as clean sky."),
            detail={"channel": channel, "source": source,
                    "frontier_mjd": mjd, "previous_mjd": prev,
                    "sky_days_gained": sky,
                    "advance_interval_days": interval,
                    "stall_confirmed_days": confirmed,
                    "lag_days": lag, "caught_up": caught_up,
                    "first_seen_utc": entry.get("first_seen_utc"),
                    "last_seen_utc": entry.get("last_seen_utc")}))
    return out


# ---------------------------------------------------------------------------
# Per-channel conditions
# ---------------------------------------------------------------------------
def tocsin_alerts(root: Path) -> list[Alert]:
    """Conditions on the nightly stellar alert screen."""
    out: list[Alert] = []
    d = root / "results" / "tocsin"

    assess = _load(d / "assessment.json") or _load(d / "summary.json") or {}
    tiers = (assess.get("tier_counts") or (assess.get("ledger") or {}).get("tier_counts")
             or {})
    n_cand = int(tiers.get("candidate") or 0)
    if n_cand:
        # Keyed on the target list, not the count, so one new candidate joining
        # three existing ones still fires while a re-run of the same three does not.
        names = sorted(str(t) for t in (assess.get("candidates") or [])) or [str(n_cand)]
        out.append(Alert(
            key="tocsin:candidates:" + hashlib.sha1(
                "|".join(names).encode()).hexdigest()[:12],
            severity="candidate", channel="tocsin",
            title=f"TOCSIN promoted {n_cand} target(s) to candidate",
            body=(f"The nightly Rubin stellar screen has {n_cand} target(s) at tier "
                  f"`candidate`: repeated achromatic events at a catalogued "
                  f"position, surviving the flare, dipole, mover and glint cuts and "
                  f"the trial-corrected cross-night test.\n\n"
                  f"Read `results/tocsin/assessment.json` and `watchlist.csv`.\n\n"
                  f"Before believing it: check whether the targets share a sky bin "
                  f"(a deep-drilling field raised the local rate 4-75x and produced "
                  f"two false candidates before), and whether any is a catalogued "
                  f"variable."),
            detail={"n_candidate": n_cand, "targets": names[:20]}))

    pop = _load(d / "population.json") or {}
    if str(pop.get("verdict", "")).upper() not in ("", "NO_STRUCTURE",
                                                   "INSUFFICIENT_RESOLUTION",
                                                   "INSUFFICIENT_POPULATION",
                                                   "NO_TEST_COULD_RUN"):
        out.append(Alert(
            key=f"tocsin:population:{pop.get('verdict')}",
            severity="candidate", channel="tocsin",
            title=f"TOCSIN population tests: {pop.get('verdict')}",
            body=("The population-structure tests on the screened stellar sample "
                  "returned something other than a null. This is the test that is "
                  "immune to per-object contamination.\n\n"
                  "Read `results/tocsin/population.json`."),
            detail={"verdict": pop.get("verdict")}))
    return out


def loom_alerts(root: Path) -> list[Alert]:
    """Conditions on the solar-system artefact channel."""
    out: list[Alert] = []
    d = root / "results" / "loom"

    screen = _load(d / "screen.json") or {}
    funnel = screen.get("funnel_final") or {}
    n_cand = int(funnel.get("n_candidate") or 0)
    if n_cand:
        out.append(Alert(
            key=f"loom:candidates:{n_cand}:{screen.get('frontier_mjd')}",
            severity="candidate", channel="loom",
            title=f"LOOM promoted {n_cand} object(s) to candidate",
            body=(f"{n_cand} solar-system object(s) reached tier `candidate`: above "
                  f"the radiation momentum ceiling AND anomalous on an "
                  f"artificiality channel (area-to-mass ratio, or an acceleration "
                  f"independent of heliocentric distance), with the timing, "
                  f"sky-coherence, coma and geometry vetoes all clear.\n\n"
                  f"Read `results/loom/objects.csv` (filter `tier == candidate`) "
                  f"and `results/loom/screen.json`.\n\n"
                  f"Before believing it: a magnitude-only exceedance is a dark "
                  f"comet, and that population is demonstrably incomplete. Run "
                  f"`loom-litcheck` on the designations."),
            detail={"n_candidate": n_cand}))

    assess = _load(d / "assessment.json") or {}
    rep = (assess.get("replication") or {}).get("verdict", "")
    if str(rep).upper() == "REPLICATION_STRUCTURE_DETECTED":
        out.append(Alert(
            key=f"loom:replication:{assess.get('n_anomalous')}",
            severity="candidate", channel="loom",
            title="LOOM replication tests detected population structure",
            body=("The anomalous solar-system objects show structure — element "
                  "clustering, orbital-pole coherence, inclination isotropy, "
                  "resonance concentration or photometric homogeneity — beyond "
                  "matched random subsets of the same screened population.\n\n"
                  "This is the channel's actual decision criterion.\n\n"
                  "Read `results/loom/assessment.json`. Check `quality_independence` "
                  "first: a score correlated with observation count has produced "
                  "this result spuriously before."),
            detail={"n_anomalous": assess.get("n_anomalous")}))

    controls = (assess.get("controls") or {}).get("verdict", "")
    if str(controls).upper() == "SCREEN_INSENSITIVE":
        out.append(Alert(
            key="loom:controls:insensitive",
            severity="health", channel="loom",
            title="LOOM failed its positive control",
            body=("A known artificial object was measured by the screen and NOT "
                  "flagged. The screen was shown the signal it is built to find "
                  "and did not return it, so no null from it is interpretable "
                  "until this is fixed.\n\n"
                  "Read `results/loom/assessment.json` -> `controls`."),
            detail={"verdict": controls}))

    # THE MILESTONE.  LOOM's central discriminant -- which heliocentric-distance
    # law the acceleration follows -- needs objects with two apparitions, and the
    # survey has been running a month.  The week that changes is the week the
    # channel starts being able to answer its own question.
    cov = screen.get("coverage") or {}
    if cov.get("law_discrimination_available"):
        out.append(Alert(
            key="loom:milestone:law_discrimination_available",
            severity="milestone", channel="loom",
            title="LOOM: the law-discrimination test can now run",
            body=(f"Objects with a residual baseline above the minimum AND two or "
                  f"more apparitions now exist in the screened sample "
                  f"({cov.get('n_multi_apparition')} multi-apparition, median arc "
                  f"{cov.get('residual_arc_days_median')} d).\n\n"
                  f"That is the discriminant separating an engineered acceleration "
                  f"from a dark comet, and it is the channel's central novelty "
                  f"claim. It has been returning INSUFFICIENT_R_SPAN on every "
                  f"object since the channel was built.\n\n"
                  f"The screen will now exercise it automatically."),
            detail=cov))

    # A new unexplained exceedance in the monthly SBDB calibration -- but only if
    # the literature check has not already accounted for it.  Most exceedances
    # resolve to the dark-comet population, which is incomplete and growing, so an
    # alert that ignores litcheck would notify on objects already published.
    cal = _load(d / "calibration.json") or {}
    lit = _load(d / "litcheck.json") or {}
    explained = {name for name, o in (lit.get("objects") or {}).items()
                 if o.get("explained_in_literature")}
    survivors = ((cal.get("epsilon") or {}).get("asteroid_rho_2000") or {}
                 ).get("survivors") or []
    for s in survivors:
        name = str(s.get("name", "?"))
        if name in explained:
            continue
        out.append(Alert(
            key=f"loom:exceedance:{name}",
            severity="candidate", channel="loom",
            title=f"LOOM: new unexplained exceedance {name}",
            body=(f"`{name}` exceeds the radiation momentum ceiling "
                  f"(eps = {s.get('epsilon_effective'):.3g}) with a reliable orbit "
                  f"solution and is not in the channel's explained set.\n\n"
                  f"A2 = {s.get('A2_au_day2'):.3g} au/day^2, S/N "
                  f"{s.get('a2_snr'):.1f}, arc {s.get('data_arc_days')} d, "
                  f"diameter {s.get('diameter_m'):.0f} m "
                  f"({'measured' if s.get('diameter_measured') else 'from H'}).\n\n"
                  f"Next step is `loom-litcheck --name \"{name}\"`: the dark-comet "
                  f"population is incomplete and most exceedances resolve to it."),
            detail=s))
    return out


def health_alerts(root: Path, now: datetime | None = None) -> list[Alert]:
    """Has something gone quiet -- the channel, or the data reaching it?

    Silence is the failure mode this repository is worst placed to notice: a
    broken cron and a sky with nothing in it produce the same empty directory,
    and the second is a result while the first is a bug.  The watchdog catches
    runs that FAIL; nothing else catches runs that stop happening.

    Three distinct questions, in the order they can fail:
      * did the channel run at all (result age, read from the run stamp);
      * did it run on anything new (data frontier vs the wall clock) -- a
        channel re-screening a frozen mirror passes every other check;
      * did it reach the broker at all (an explicit NO_DATA_REACHED verdict).
    """
    now = now or datetime.now(timezone.utc)
    out: list[Alert] = []
    for channel, limit in STALE_DAYS.items():
        d = root / "results" / channel
        marker = d / STALE_MARKER[channel]
        if not marker.exists():
            continue
        age = _age_days(marker, now)
        if age > limit:
            # Escalate on a doubling, not on every multiple of the limit.  A
            # channel that stays dead should keep nagging -- silence is the
            # failure this repository is worst placed to notice -- but nagging
            # every four days for a year is how a notification becomes noise.
            octave = int(math.log2(age / limit))
            out.append(Alert(
                key=f"{channel}:stale:{octave}",
                severity="health", channel=channel,
                title=f"{channel.upper()} has not produced results for "
                      f"{age:.1f} days",
                body=(f"`results/{channel}/{marker.name}` was last written "
                      f"{age:.1f} days ago; the schedule expects an update at "
                      f"least every {limit:.0f} days.\n\n"
                      f"A stopped cron and an empty sky look identical from the "
                      f"repository, and only one of them is a result. Check the "
                      f"Actions tab: GitHub disables scheduled workflows after 60 "
                      f"days without repository activity, and a run that fails "
                      f"before its commit step leaves no trace here at all."),
                detail={"age_days": age, "limit_days": limit}))

    # The cross-broker verdict, if one has been reached: appended to both
    # frontier alerts so the reader is not sent to re-diagnose a settled cause.
    cause = outage_context(root)

    # HAS THE DATA STOPPED, AS OPPOSED TO THE CHANNEL?
    #
    # Every check above asks whether the screens ran. This asks whether they
    # ran on anything new -- and it is the only one that can catch the mirror
    # going dark, because a channel screening the same frontier night after
    # night looks perfectly healthy from every other angle: it runs on
    # schedule, writes a fresh stamp, commits, and reports a clean null.
    for channel, marker, key in (("tocsin", "ledger.json", "last_mjd_screened"),
                                 ("loom", "screen.json", "frontier_mjd")):
        rec = _load(root / "results" / channel / marker)
        if not isinstance(rec, dict):
            continue
        mjd = _f(rec.get(key))
        if not math.isfinite(mjd) or mjd <= 0:
            continue
        lag = (now - (MJD_EPOCH + timedelta(days=mjd))).total_seconds() / 86400.0
        if lag > DATA_LAG_LIMIT_DAYS:
            out.append(Alert(
                key=f"{channel}:data_frontier:{int(math.log2(lag / DATA_LAG_LIMIT_DAYS))}",
                severity="health", channel=channel,
                title=f"{channel.upper()} is screening data {lag:.0f} days old",
                body=(f"The most recent Rubin data `{channel}` has reached is "
                      f"MJD {mjd:.2f}, which is {lag:.1f} days behind now. The "
                      f"broker mirror normally lags ~16 days; the limit here is "
                      f"{DATA_LAG_LIMIT_DAYS:.0f}.\n\n"
                      f"The channel itself is fine — it is running on schedule "
                      f"and committing results. What has stopped is the DATA. "
                      f"A screen re-reading the same frontier every night "
                      f"produces a clean null indefinitely and looks healthy "
                      f"from every other angle, which is why this check exists "
                      f"separately from the staleness check.\n\n"
                      f"Check whether ALeRCE is still ingesting the LSST alert "
                      f"stream before reading any recent null from this "
                      f"channel." + cause),
                detail={"frontier_mjd": mjd, "lag_days": lag,
                        "limit_days": DATA_LAG_LIMIT_DAYS}))

    # HAS THE FRONTIER STOPPED MOVING, AS OPPOSED TO MERELY BEING OLD?
    #
    # The check above measures the frontier against the wall clock, so it can
    # only fire once a freeze has eaten the entire 30-day budget -- and the
    # mirror is already ~16 days behind before it stops, so that is a fortnight
    # of nulls that mean "no new sky" being filed as "clean sky".  This one
    # measures the frontier against ITSELF: the same epoch, run after run, is a
    # stall regardless of how recent that epoch happens to be.
    #
    # It needs memory, and results/alerts/frontier.json is that memory.  Nothing
    # else in the repository has it: every channel's result file describes the
    # run that wrote it, so from a single file a frozen mirror and an advancing
    # one are indistinguishable.
    seen = (_load(frontier_path(root)) or {}).get("channels") or {}
    for channel, (mjd, source) in current_frontiers(root).items():
        entry = seen.get(channel) or {}
        # A frontier never recorded, or one that differs from the record, has
        # not been observed to sit still -- the first sighting of a value is
        # the start of its clock, not evidence about how long it has been there.
        if _f(entry.get("mjd")) != mjd:
            continue
        since = _parse_stamp(entry.get("first_seen_utc"))
        if since is None:
            continue
        frozen = (now - since).total_seconds() / 86400.0
        if frozen > FRONTIER_STALL_DAYS:
            src_file, _, src_key = source.partition(":")
            cadence = entry.get("observed_advance_days") or {}
            n_adv = int(cadence.get("n_advances") or 0)
            measured = (f"Observed cadence so far: {n_adv} advance(s), median "
                        f"{cadence.get('median_days')} d, longest gap "
                        f"{cadence.get('max_days')} d."
                        if n_adv else
                        "The frontier has not yet been observed to advance at "
                        "all, so there is no measured cadence to compare this "
                        "against.")
            out.append(Alert(
                # Escalate by doubling, as the staleness check does: a mirror
                # that stays dead should keep nagging without nagging weekly.
                key=f"{channel}:frontier_stalled:"
                    f"{int(math.log2(frozen / FRONTIER_STALL_DAYS))}",
                severity="health", channel=channel,
                title=f"{channel.upper()}'s data frontier has not advanced in "
                      f"{frozen:.0f} days",
                body=(f"`{channel}` has reported the same newest epoch, MJD "
                      f"{mjd:.5f}, on every run since "
                      f"{entry.get('first_seen_utc')} — {frozen:.1f} days, "
                      f"against a limit of {FRONTIER_STALL_DAYS:.0f}. Source: "
                      f"`results/{channel}/{src_file}` → `{src_key}`.\n\n"
                      f"{measured}\n\n"
                      f"The channel is running and committing normally; what "
                      f"has stopped is the mirror. Every null it reports while "
                      f"this holds means *no new sky*, not *clean sky*, and the "
                      f"two are not the same result.\n\n"
                      f"Check whether ALeRCE is still ingesting the LSST alert "
                      f"stream. If the pause turns out to be ordinary batching, "
                      f"raise `FRONTIER_STALL_DAYS` using the cadence recorded "
                      f"in `results/alerts/frontier.json` rather than by "
                      f"guessing again." + cause),
                detail={"frontier_mjd": mjd, "frozen_days": frozen,
                        "limit_days": FRONTIER_STALL_DAYS, "source": source,
                        "first_seen_utc": entry.get("first_seen_utc"),
                        "observed_advance_days": cadence}))

    # A channel that reaches no data is not returning a null; it is not running.
    for channel, name in (("tocsin", "summary.json"), ("loom", "screen.json")):
        rec = _load(root / "results" / channel / name) or {}
        if str(rec.get("verdict", "")).upper() == "NO_DATA_REACHED":
            out.append(Alert(
                key=f"{channel}:no_data_reached:{rec.get('screened_at_utc') or rec.get('run_utc')}",
                severity="health", channel=channel,
                title=f"{channel.upper()} reached no data on its last run",
                body=(f"The last run wrote `verdict: NO_DATA_REACHED` — the broker "
                      f"was unreachable or answered unusably. This is not a null "
                      f"result and must not be read as one.\n\n"
                      f"Read `results/{channel}/{name}`."),
                detail={"verdict": rec.get("verdict")}))
    return out


# ---------------------------------------------------------------------------
# Did the SCHEDULER fire at all?
# ---------------------------------------------------------------------------
def scheduler_alerts(root: Path) -> list[Alert]:
    """A cron that never fired, as reported by ``seti.cronwatch``.

    This is the gap every other check in this module leaves open.  ``STALE_DAYS``
    asks whether results are old, and its windows are deliberately over twice
    each channel's cadence so one missed firing is not an incident -- which is
    correct for a channel that is merely late, and useless for a scheduler that
    dropped the firing outright.  Between the two sits a week in which nothing
    says anything, and for a weekly channel that is the whole interval.

    The finding is read rather than computed here: asking the Actions API needs a
    token and network, both of which the hourly watchdog has and this evaluation
    does not.  Dedup is keyed by the missed firing, so one dropped Wednesday
    notifies once, however many times it is re-read.
    """
    rec = _load(root / "results" / "cronwatch" / "status.json")
    if not isinstance(rec, dict):
        return []
    out: list[Alert] = []
    for wf in rec.get("workflows") or []:
        if not wf.get("overdue"):
            continue
        name = wf.get("name") or wf.get("workflow")
        fired = wf.get("catchup_dispatched_utc")
        out.append(Alert(
            key=f"cron:{wf.get('workflow')}:{wf.get('expected_last_fire_utc')}",
            severity="health", channel="cronwatch",
            title=f"{name} did not fire at {wf.get('expected_last_fire_utc')}",
            body=(f"GitHub's scheduler did not start `{wf.get('workflow')}` at its "
                  f"scheduled firing of {wf.get('expected_last_fire_utc')} "
                  f"(cron `{wf.get('cron_matched')}`, cadence "
                  f"{wf.get('cadence_hours')} h). Its last scheduled run was "
                  f"{wf.get('last_scheduled_run_utc') or 'never'}, "
                  f"{wf.get('hours_late')} h ago.\n\n"
                  + ("A catch-up run was dispatched automatically at "
                     f"{fired}; check that it finished.\n\n" if fired else
                     "NO catch-up was dispatched" + (
                         " -- this workflow has no `workflow_dispatch` trigger, so "
                         "the firing can only be recovered by hand.\n\n"
                         if not wf.get("has_dispatch") else
                         f" ({wf.get('catchup_error', 'the dispatch was not attempted')}).\n\n")
                  )
                  + "Scheduled runs on GitHub are best effort and are dropped "
                    "under load. A dropped firing leaves no failed run for the "
                    "watchdog to retry and no stale result until the staleness "
                    "window expires days later, which is why it is watched "
                    "separately."),
            detail={k: wf.get(k) for k in (
                "workflow", "cron_matched", "cadence_hours", "grace_hours",
                "expected_last_fire_utc", "last_scheduled_run_utc", "hours_late",
                "has_dispatch", "catchup_dispatched_utc", "catchup_error")}))
    return out


# ---------------------------------------------------------------------------
# Orchestration and deduplication
# ---------------------------------------------------------------------------
def evaluate(root: Path, now: datetime | None = None) -> list[Alert]:
    """Every alert condition, evaluated against whatever results exist."""
    return [*tocsin_alerts(root), *loom_alerts(root), *health_alerts(root, now),
            *frontier_recovery_alerts(root, now), *scheduler_alerts(root)]


def check(root: Path, state_path: Path | None = None,
          now: datetime | None = None, record: bool = True) -> dict:
    """Evaluate, drop anything already raised, and record what is new.

    Returns a report with ``new`` — the alerts a human has not yet been told
    about.  ``record=False`` evaluates without consuming them, which is what a
    dry run wants.
    """
    now = now or datetime.now(timezone.utc)
    state_path = state_path or (root / "results" / "alerts" / "state.json")
    state = _load(state_path) or {"seen": {}}
    seen = dict(state.get("seen") or {})

    # EVALUATE BEFORE RECORDING THE FRONTIER, always.  record_frontier stamps
    # the moment a frontier value was first seen, and that stamp is what the
    # stall check measures against; folding today's sighting in first would
    # move the clock forward on every run and the stall could never fire.
    alerts = evaluate(root, now)
    new = [a for a in alerts if a.key not in seen]
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if record:
        for a in new:
            seen[a.key] = {"first_seen_utc": stamp, "severity": a.severity,
                           "title": a.title}

        # RE-ARM THE STALL KEYS THE RECOVERY JUST ANSWERED.
        #
        # Without this the stall detector is ONE-SHOT for the life of the
        # repository.  The stall key escalates by doubling --
        # `<channel>:frontier_stalled:0`, `:1`, `:2` -- so once a stall has been
        # reported those keys are consumed forever, and the NEXT outage is
        # silent until it grows past the longest escalation already seen.  This
        # repository has already consumed `:0` and `:1` on the July 2026 Rubin
        # outage, so the next stall would say nothing for its first 28 days,
        # and the one after that nothing for 56.  A detector whose blind spot
        # doubles every time it fires is worse than no detector, because the
        # silence still reads as health.
        #
        # A recovery is the honest moment to clear them: the condition those
        # keys stand for has demonstrably ended, so re-arming announces nothing
        # and loses nothing.
        #
        # Only keys NOT currently active are cleared.  Dropping a key whose
        # condition is still true would re-raise it on the very next run --
        # re-notifying a human about something they were already told, which is
        # the noise this module is built to avoid.
        active_keys = {a.key for a in alerts}
        for a in alerts:
            if ":frontier_recovered:" not in a.key:
                continue
            for prefix in (f"{a.channel}:frontier_stalled:",
                           f"{a.channel}:data_frontier:"):
                for k in [k for k in seen
                          if k.startswith(prefix) and k not in active_keys]:
                    seen.pop(k, None)

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"seen": seen}, indent=1,
                                         sort_keys=True) + "\n")
        record_frontier(root, now)

    report = {
        "checked_at_utc": stamp,
        "n_active": len(alerts),
        "n_new": len(new),
        "alert": bool(new),
        "by_severity": {s: sum(1 for a in new if a.severity == s)
                        for s in SEVERITIES},
        "new": [a.as_dict() for a in new],
        "active": [a.as_dict() for a in alerts],
        # Reported whether or not it alerts.  Below the threshold a stalling
        # mirror is invisible in every other field here, and "how old is the
        # newest sky we have seen" is the first thing to check before reading
        # any null on this page as a statement about the sky.
        "frontier": frontier_status(root, now),
    }
    out = root / "results" / "alerts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    return report


def issue_title(alerts: list[dict]) -> str:
    """One line, safe to pass through a shell, that says what happened."""
    if not alerts:
        return "No new alerts"
    if len(alerts) == 1:
        title = str(alerts[0].get("title") or "Alert")
    else:
        counts = {s: sum(1 for a in alerts if a.get("severity") == s)
                  for s in SEVERITIES}
        parts = [f"{n} {s}" for s, n in counts.items() if n]
        title = "Automated screens: " + ", ".join(parts)
    # Newlines and backticks in an issue title are how a `gh issue create`
    # invocation turns into a broken command instead of a notification.
    title = " ".join(title.replace("`", "").split())
    return title[:250]


def issue_labels(alerts: list[dict]) -> list[str]:
    """Labels for the issue: severity plus the channels involved."""
    sev = {str(a.get("severity")) for a in alerts if a.get("severity") in SEVERITIES}
    chan = {str(a.get("channel")) for a in alerts if a.get("channel")}
    return sorted(sev) + sorted(chan)


def issue_body(alerts: list[dict], repo_url: str = "") -> str:
    """Markdown for a GitHub issue.  States what was measured, never a finding."""
    lines = ["The automated screens raised the following. Each is a request to "
             "look, not a claim that anything was found.\n"]
    for a in alerts:
        lines.append(f"### [{a['severity']}] {a['title']}")
        lines.append(a["body"])
        lines.append("")
    lines.append("---")
    lines.append("Raised by `seti.alerts`. Every alert is deduplicated by a stable "
                 "key in `results/alerts/state.json`, so this will not repeat for "
                 "the same finding. Delete the key to re-arm it.")
    if repo_url:
        lines.append(f"\nFull state: {repo_url}/blob/main/results/alerts/latest.json")
    return "\n".join(lines)
