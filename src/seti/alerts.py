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
from datetime import datetime, timezone
from pathlib import Path

SEVERITIES = ("candidate", "health", "milestone")

# How stale a channel's results may get before silence is treated as failure.
# Nightly channels get a wider window than their cadence so one missed firing is
# not an incident; a week of silence is.
STALE_DAYS = {"tocsin": 4.0, "loom": 10.0}


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
    """Has a channel gone quiet?

    Silence is the failure mode this repository is worst placed to notice: a
    broken cron and a sky with nothing in it produce the same empty directory,
    and the second is a result while the first is a bug.  The watchdog catches
    runs that FAIL; nothing else catches runs that stop happening.
    """
    now = now or datetime.now(timezone.utc)
    out: list[Alert] = []
    for channel, limit in STALE_DAYS.items():
        d = root / "results" / channel
        marker = d / ("summary.json" if channel == "tocsin" else "screen.json")
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
# Orchestration and deduplication
# ---------------------------------------------------------------------------
def evaluate(root: Path, now: datetime | None = None) -> list[Alert]:
    """Every alert condition, evaluated against whatever results exist."""
    return [*tocsin_alerts(root), *loom_alerts(root), *health_alerts(root, now)]


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

    alerts = evaluate(root, now)
    new = [a for a in alerts if a.key not in seen]
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if record:
        for a in new:
            seen[a.key] = {"first_seen_utc": stamp, "severity": a.severity,
                           "title": a.title}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"seen": seen}, indent=1,
                                         sort_keys=True) + "\n")

    report = {
        "checked_at_utc": stamp,
        "n_active": len(alerts),
        "n_new": len(new),
        "alert": bool(new),
        "by_severity": {s: sum(1 for a in new if a.severity == s)
                        for s in SEVERITIES},
        "new": [a.as_dict() for a in new],
        "active": [a.as_dict() for a in alerts],
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
