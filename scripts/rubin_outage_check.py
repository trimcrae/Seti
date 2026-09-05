"""Is the Rubin data frontier frozen because the MIRROR stopped, or because the SKY did?

Runner-only network code (``docs/channel-brief.md`` §0: broker egress is
403-blocked in the sandbox).

WHY THIS EXISTS.  Both Rubin channels (``tocsin``, ``loom``) read the LSST alert
stream through ALeRCE's TAP mirror, and both have reported the same newest epoch
--- MJD 61235.419 = 2026-07-14T10:03Z --- on every run since 2026-07-30.  The
alert module notices the freeze and says so, but its text asks the reader to
"check whether ALeRCE is still ingesting the LSST alert stream", which names one
suspect out of two.  The two have opposite consequences and the apparatus cannot
tell them apart from a single broker:

* **MIRROR_STALLED** --- Rubin is observing, ALeRCE has stopped mirroring.  We are
  blind to sky that exists; the fix is to switch or add a broker, and every night
  spent waiting is a night of real data going unscreened.
* **SKY_STOPPED** --- Rubin is not observing.  Nothing is being missed, no code
  change helps, and both channels are correct to report nothing.  The only error
  would be to read the resulting run of nulls as a statement about the sky.

The discriminator is a SECOND, INDEPENDENT broker.  If Fink or Lasair holds LSST
epochs newer than ALeRCE's, the mirror is the problem.  If every broker stops on
the same night, the stream itself stopped, and the last night held in common is
the last night Rubin observed.

Every query is guarded and its error captured rather than raised: the point is to
come back from one runner pass with a complete picture, not to iterate blind.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from seti.tocsin.brokers import (  # noqa: E402
    ALERCE_TAP,
    FINK_LSST_API,
    LASAIR_ENDPOINT,
    AlerceTAP,
)

MJD_UNIX_EPOCH = 40587.0
# A broker is "ahead" only by more than one night.  Brokers ingest a night in
# batches and their newest epoch drifts by hours within the same night, so a
# sub-day difference is bookkeeping, not evidence of a stalled mirror.
AHEAD_TOLERANCE_DAYS = 1.0

# Fink's ZTF portal, used only as a control on whether Fink itself is alive.
FINK_ZTF_API = "https://api.fink-portal.org"


def _now_mjd() -> float:
    return MJD_UNIX_EPOCH + datetime.datetime.now(
        datetime.timezone.utc).timestamp() / 86400.0


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(mjd: float | None) -> str | None:
    if mjd is None:
        return None
    dt = datetime.datetime(1858, 11, 17, tzinfo=datetime.timezone.utc) + \
        datetime.timedelta(days=float(mjd))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


# --------------------------------------------------------------------------
# ALeRCE
# --------------------------------------------------------------------------
def probe_alerce(timeout: float) -> dict:
    """Newest epoch, and the nightly detection counts that show HOW it stopped.

    The nightly histogram is the part a single MAX() cannot give: a mirror that
    was cut off mid-ingest tapers (a short final night), whereas a stream that
    stopped at the source ends on a night of ordinary size.
    """
    tap = AlerceTAP(ALERCE_TAP, timeout=timeout)
    out: dict = {"url": ALERCE_TAP, "queries": {}}

    def run(name: str, adql: str, maxrec: int = 400):
        try:
            rows = tap.query(adql, maxrec=maxrec, retries=2)
            out["queries"][name] = {"rows": len(rows), "data": rows[:maxrec]}
            return rows
        except Exception as exc:                                  # noqa: BLE001
            out["queries"][name] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
            return None

    rows = run("max_mjd_lsst_join",
               "SELECT MAX(d.mjd) AS mjd_max FROM alerce_tap.detection AS d "
               "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
               "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id",
               maxrec=5)
    out["frontier_mjd"] = _num(rows[0].get("mjd_max")) if rows else None

    # `lsst_ss_detection` has no `oid`: it carries `ssobjectid`, and the join is
    # (measurement_id, ssobjectid) --- `seti.loom.acquire._join_clause`.  Run 1
    # guessed `ss.oid` and lost the loom frontier to an ADQL error.
    rows = run("max_mjd_ss_detection",
               "SELECT MAX(d.mjd) AS mjd_max FROM alerce_tap.lsst_ss_detection AS ss "
               "JOIN alerce_tap.detection AS d "
               "ON d.measurement_id = ss.measurement_id AND d.oid = ss.ssobjectid",
               maxrec=5)
    out["frontier_ss_mjd"] = _num(rows[0].get("mjd_max")) if rows else None

    # Does ANY table in the mirror hold something newer?  If ALeRCE's ZTF side is
    # current while its LSST side is frozen, the service is alive and the LSST
    # ingest specifically is not --- a different failure from ALeRCE being down.
    rows = run("max_mjd_any_detection",
               "SELECT MAX(mjd) AS mjd_max FROM alerce_tap.detection", maxrec=5)
    out["frontier_any_mjd"] = _num(rows[0].get("mjd_max")) if rows else None

    # THE SELF-CONTROL.  Run 1 found the bare `detection` table maxing at exactly
    # the LSST frontier, which is consistent with two very different things: a
    # mirror that ingests only LSST (so the two are the same number by
    # construction) and a mirror that has stopped ingesting everything.  Grouping
    # the far smaller `object` table by survey separates them: if ALeRCE is
    # current on another survey while LSST sits at 2026-07-14, the service is
    # alive and its LSST feed specifically is not.
    rows = run("survey_currency",
               "SELECT sid, tid, COUNT(*) AS n, MAX(lastmjd) AS last_max "
               "FROM alerce_tap.object GROUP BY sid, tid", maxrec=50)
    if rows:
        surveys = []
        for r in rows:
            last = _num(r.get("last_max"))
            surveys.append({"sid": r.get("sid"), "tid": str(r.get("tid")),
                            "n": _num(r.get("n")), "last_mjd": last,
                            "last_utc": _iso(last)})
        out["survey_currency"] = surveys
        newest = [s["last_mjd"] for s in surveys if s["last_mjd"] is not None]
        out["newest_any_survey_mjd"] = max(newest) if newest else None

    lo = _now_mjd() - 120.0
    rows = run("nightly_counts_120d",
               "SELECT FLOOR(d.mjd) AS night, COUNT(*) AS n "
               "FROM alerce_tap.detection AS d "
               "JOIN alerce_tap.lsst_detection AS ld ON d.oid = ld.oid "
               "AND d.sid = ld.sid AND d.measurement_id = ld.measurement_id "
               f"WHERE d.mjd >= {lo} GROUP BY FLOOR(d.mjd) ORDER BY night",
               maxrec=400)
    if rows:
        hist = []
        for r in rows:
            night, n = _num(r.get("night")), _num(r.get("n"))
            if night is not None:
                hist.append({"night_mjd": int(night), "date": (_iso(night) or "")[:10],
                             "n": int(n or 0)})
        out["nightly_counts"] = hist
        out["last_nights"] = hist[-10:]
    return out


# --------------------------------------------------------------------------
# Fink --- the independent second opinion
# --------------------------------------------------------------------------
def probe_fink(timeout: float) -> dict:
    """Ask Fink for its newest LSST night --- and for its newest ZTF night.

    Run 1 established the endpoint set the hard way: ``/api/v1/latests`` is 404 on
    the LSST portal, and ``/api/v1/statistics`` returns the whole per-night
    history --- one row per observing night, with ``f:night`` as ``YYYYMMDD`` and
    ``f:alerts`` as that night's alert count.  That is a **second, independent
    nightly histogram** of the same stream ALeRCE mirrors, which is exactly the
    comparison this check was written to make.

    The ZTF portal is queried as a **control on Fink itself**.  If Fink's LSST
    history stops on the same night as ALeRCE's while its ZTF history runs to
    this week, then two independent brokers are alive and neither is being handed
    LSST alerts --- which no broker-side explanation covers.
    """
    import requests

    out: dict = {"url": FINK_LSST_API, "attempts": {}}

    def stats(label: str, base: str) -> dict:
        url = base.rstrip("/") + "/api/v1/statistics"
        rec: dict = {"method": "POST", "url": url}
        try:
            resp = requests.post(url, json={"date": "", "output-format": "json"},
                                 timeout=timeout)
            rec["status"] = resp.status_code
            rec["bytes"] = len(resp.text or "")
            payload = resp.json()
            nights = _nights_from_fink_stats(payload)
            rec["n_nights"] = len(nights)
            rec["last_nights"] = nights[-10:]
            if nights:
                rec["last_night"] = nights[-1]["date"]
                rec["frontier_mjd"] = nights[-1]["mjd"]
        except Exception as exc:                                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        out["attempts"][label] = rec
        return rec

    lsst = stats("lsst_statistics", FINK_LSST_API)
    ztf = stats("ztf_statistics", FINK_ZTF_API)

    out["frontier_mjd"] = lsst.get("frontier_mjd")
    out["last_night"] = lsst.get("last_night")
    out["nightly_counts"] = lsst.get("last_nights")
    # The control, kept as its own field: it must never be mistaken for an LSST
    # frontier, or a live ZTF feed would read as live Rubin data.
    out["ztf_control"] = {"frontier_mjd": ztf.get("frontier_mjd"),
                          "last_night": ztf.get("last_night"),
                          "last_nights": ztf.get("last_nights")}
    return out


def _nights_from_fink_stats(payload) -> list[dict]:
    """Fink's per-night statistics rows -> [{date, mjd, n_alerts}], oldest first.

    ``f:night`` is a ``YYYYMMDD`` string labelling the observing night, so the
    epoch it stands for is a date, not an instant.  It is placed at 12:00 UT of
    that date: a Chilean night's alerts carry timestamps from about 00:00 to
    10:00 UT on the labelled date, so noon is within half a day of every one of
    them and no rounding choice can make one broker look a night ahead of
    another when they hold the same night.
    """
    rows = payload if isinstance(payload, list) else (payload or {}).get("results") or []
    nights = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        raw = str(r.get("f:night") or r.get("night") or "").strip()
        if len(raw) != 8 or not raw.isdigit():
            continue
        try:
            d = datetime.date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
        except ValueError:
            continue
        mjd = (d.toordinal() - datetime.date(1858, 11, 17).toordinal()) + 0.5
        nights.append({"date": d.isoformat(), "mjd": float(mjd),
                       "n_alerts": _num(r.get("f:alerts"))})
    nights.sort(key=lambda x: x["mjd"])
    return nights


def _max_time_in(payload, depth: int = 0) -> float | None:
    """Largest plausible epoch anywhere in a JSON payload, normalised to MJD.

    Written scan-any-shape on purpose: the exact column naming of the LSST Fink
    portal is unverified here, and a hard-coded key that turns out to be spelled
    differently would report "no data" for a broker that is in fact current --
    the precise mistake this whole check exists to catch.  JD (~2.46e6) and MJD
    (~6.1e4) are separated by five orders of magnitude, so they are safe to tell
    apart by magnitude; anything outside both ranges is not a time and is ignored.
    """
    if depth > 6:
        return None
    best: float | None = None

    def offer(v):
        nonlocal best
        f = _num(v)
        if f is None:
            return
        if 2_400_000.0 < f < 2_600_000.0:          # Julian Date
            f -= 2_400_000.5
        elif not (50_000.0 < f < 80_000.0):        # not an MJD either
            return
        best = f if best is None else max(best, f)

    if isinstance(payload, dict):
        for k, v in payload.items():
            kl = str(k).lower()
            if isinstance(v, (dict, list)):
                sub = _max_time_in(v, depth + 1)
                if sub is not None:
                    best = sub if best is None else max(best, sub)
            elif "jd" in kl or "mjd" in kl or kl.endswith("time"):
                offer(v)
    elif isinstance(payload, list):
        for item in payload[:200]:
            sub = _max_time_in(item, depth + 1)
            if sub is not None:
                best = sub if best is None else max(best, sub)
    return best


# --------------------------------------------------------------------------
# Lasair --- only if a token happens to be configured
# --------------------------------------------------------------------------
def probe_lasair(timeout: float) -> dict:
    import os

    import requests

    token = os.environ.get("LASAIR_TOKEN", "").strip()
    out: dict = {"url": LASAIR_ENDPOINT, "have_token": bool(token)}
    if not token:
        out["skipped"] = "no LASAIR_TOKEN in the environment"
        out["frontier_mjd"] = None
        return out
    try:
        resp = requests.post(
            LASAIR_ENDPOINT.rstrip("/") + "/query/",
            headers={"Authorization": f"Token {token}"},
            data={"selected": "MAX(objects.lastDiaSourceMjdTai) AS mjd_max",
                  "tables": "objects", "conditions": "", "limit": 1,
                  "format": "json"},
            timeout=timeout)
        out["status"] = resp.status_code
        payload = resp.json()
        out["json_head"] = json.dumps(payload)[:800]
        out["frontier_mjd"] = _max_time_in(payload)
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
        out["frontier_mjd"] = None
    return out


# --------------------------------------------------------------------------
# ZTF --- is the OTHER public wide-field stream alive at all?
# --------------------------------------------------------------------------
# WHY THIS IS HERE.  With Rubin dark, ZTF is the one public wide-field survey
# whose alert stream is shaped like Rubin's.  Two facts about it were on record
# before 2026-09-05, and both were about BROKERS, not about ZTF: ALeRCE's non-LSST
# table stopped on 2026-04-30, and Fink's ZTF portal timed out.  From those the
# repository inferred that "the live stream appears to have ended", which is one
# hypothesis out of two -- exactly the mirror-vs-sky ambiguity this script exists
# to break for Rubin.  So the same discipline is applied to ZTF: ask several
# independent public endpoints for their newest ZTF epoch, and name the newest.
#
# The strongest source is the archive ZTF publishes ITSELF: one tarball per
# night at ztf.uw.edu/alerts/public/, so its directory listing is a nightly
# histogram that depends on no broker.  A night with alerts is a tarball of
# hundreds of megabytes; a night with none is a few tens of bytes, which is why
# size is read as well as date.
ZTF_PUBLIC_ARCHIVE = "https://ztf.uw.edu/alerts/public/"
ALERCE_ZTF_API = "https://api.alerce.online/ztf/v1"
ANTARES_API = "https://api.antares.noirlab.edu/v1"
LASAIR_ZTF_ENDPOINT = "https://lasair-ztf.lsst.ac.uk/api"
#: A stream is LIVE if some public endpoint holds an epoch this recent.  A week
#: absorbs weather, the bright-of-moon gap and a broker's ingest lag together.
ZTF_LIVE_WITHIN_DAYS = 7.0
#: Below this a nightly tarball held no alerts (an empty archive member).
ZTF_TARBALL_MIN_BYTES = 1_000_000

_TARBALL_NAME = re.compile(r"ztf_public_(\d{8})\.tar\.gz", re.I)
_SIZE_TOKEN = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)$", re.I)
_SIZE_MULT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def parse_ztf_archive_listing(html: str) -> list[dict]:
    """The nightly tarballs in an autoindex page -> [{date, mjd, size_bytes}], oldest first.

    Parsed line by line, keyed on the file NAME: the date is the eight digits in
    it, and the size is the last token on the line that reads like an autoindex
    size (``1.3G``, ``45``).  A line whose size cannot be read keeps the date
    and carries ``size_bytes=None`` --- "unknown", never "empty".  One entry per
    night: the listing shows each name twice (href and text).
    """
    out: dict[str, dict] = {}
    for line in (html or "").splitlines():
        m = _TARBALL_NAME.search(line)
        if not m:
            continue
        raw = m.group(1)
        try:
            d = datetime.date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
        except ValueError:
            continue
        tail = line[line.rfind("</a>") + 4:] if "</a>" in line else ""
        tail = re.sub(r"<[^>]*>", " ", tail)        # table cells, if any
        size = None
        for tok in reversed(tail.split()):
            sm = _SIZE_TOKEN.match(tok.strip())
            if sm:
                size = int(float(sm.group(1)) * _SIZE_MULT[sm.group(2).upper()])
                break
        mjd = (d.toordinal() - datetime.date(1858, 11, 17).toordinal()) + 0.5
        rec = out.get(d.isoformat())
        if rec is None or (rec.get("size_bytes") is None and size is not None):
            out[d.isoformat()] = {"date": d.isoformat(), "mjd": float(mjd),
                                  "size_bytes": size}
    return sorted(out.values(), key=lambda r: r["mjd"])


def ztf_archive_frontier(entries: list[dict],
                         min_bytes: int = ZTF_TARBALL_MIN_BYTES) -> dict:
    """The newest night whose tarball actually held alerts.

    A tarball smaller than ``min_bytes`` is an empty night (weather, or the
    camera off); it proves the archive job ran and says nothing about the sky,
    so it does not move the frontier.  A tarball of unknown size counts, because
    "unknown" must not read as "empty".
    """
    with_alerts = [e for e in entries
                   if e.get("size_bytes") is None or int(e["size_bytes"]) >= int(min_bytes)]
    newest_any = entries[-1] if entries else None
    newest = with_alerts[-1] if with_alerts else None
    return {"frontier_mjd": newest["mjd"] if newest else None,
            "last_night": newest["date"] if newest else None,
            "last_tarball_any_size": newest_any["date"] if newest_any else None,
            "n_tarballs": len(entries),
            "n_empty_recent": sum(1 for e in entries[-30:]
                                  if e.get("size_bytes") is not None
                                  and int(e["size_bytes"]) < int(min_bytes))}


def ztf_status(sources: dict, now: float | None = None,
               live_within_days: float = ZTF_LIVE_WITHIN_DAYS) -> dict:
    """LIVE / DARK_OR_UNSERVED / UNREACHED from whatever the sources returned.

    ``DARK_OR_UNSERVED`` is deliberately not ``DARK``: every source reached
    agreeing on an old epoch is strong evidence the stream stopped, but it is
    still evidence about what is SERVED.  The archive listing is the one source
    that comes from ZTF itself, and when it is among those reached the ``why``
    says so, because that is the case in which the two readings collapse.
    """
    now = _now_mjd() if now is None else float(now)
    by_source = {k: v.get("frontier_mjd") for k, v in (sources or {}).items()}
    reached = {k: v for k, v in by_source.items() if v is not None}
    if not reached:
        return {"status": "UNREACHED", "newest_mjd": None, "newest_utc": None,
                "newest_source": None, "days_behind_now": None, "by_source": by_source,
                "why": "No public ZTF endpoint returned an epoch; this says nothing "
                       "about ZTF."}
    src, newest = max(reached.items(), key=lambda kv: kv[1])
    behind = round(now - newest, 2)
    live = behind <= float(live_within_days)
    archive_reached = "archive" in reached
    if live:
        why = (f"ZTF is LIVE: {src} holds an epoch {behind} d old"
               + (" and the archive ZTF publishes itself agrees" if archive_reached
                  and src != "archive" and now - reached["archive"] <= live_within_days
                  else "") + ".")
    else:
        why = (f"No public ZTF endpoint holds an epoch newer than {behind} d ago "
               f"({src}). ")
        why += ("That includes ZTF's own nightly alert archive, so the stream itself "
                "has stopped or stopped being published." if archive_reached else
                "ZTF's own archive listing was not reached, so this is what the "
                "BROKERS serve; the stream itself is not established either way.")
    return {"status": "LIVE" if live else "DARK_OR_UNSERVED",
            "newest_mjd": newest, "newest_utc": _iso(newest), "newest_source": src,
            "days_behind_now": behind, "by_source": by_source, "why": why}


def probe_ztf(timeout: float) -> dict:
    """Ask every public ZTF endpoint for its newest epoch.  Nothing here raises."""
    import os

    import requests

    out: dict = {"sources": {}}

    def rec_for(name: str, url: str) -> dict:
        r: dict = {"url": url}
        out["sources"][name] = r
        return r

    # 1. The archive ZTF publishes itself.
    r = rec_for("archive", ZTF_PUBLIC_ARCHIVE)
    try:
        resp = requests.get(ZTF_PUBLIC_ARCHIVE, timeout=timeout)
        r["status"] = resp.status_code
        entries = parse_ztf_archive_listing(resp.text)
        r.update(ztf_archive_frontier(entries))
        r["last_entries"] = entries[-10:]
    except Exception as exc:                                      # noqa: BLE001
        r["error"] = f"{type(exc).__name__}: {exc}"[:400]
        r["frontier_mjd"] = None

    # 2. ALeRCE's ZTF API (a different service from the TAP mirror the Rubin
    #    channels use, whose non-LSST table stopped on 2026-04-30).
    url = f"{ALERCE_ZTF_API}/objects"
    r = rec_for("alerce_ztf", url)
    try:
        resp = requests.get(url, params={"order_by": "lastmjd", "order_mode": "DESC",
                                         "page_size": 1, "page": 1}, timeout=timeout)
        r["status"] = resp.status_code
        payload = resp.json()
        r["json_head"] = json.dumps(payload)[:600]
        r["frontier_mjd"] = _max_time_in(payload)
    except Exception as exc:                                      # noqa: BLE001
        r["error"] = f"{type(exc).__name__}: {exc}"[:400]
        r["frontier_mjd"] = None

    # 3. ANTARES (NOIRLab), public, no token.  JSON:API; the newest-alert time
    #    is an MJD under `properties`, found by the shape-agnostic scanner.
    url = f"{ANTARES_API}/loci"
    r = rec_for("antares", url)
    try:
        resp = requests.get(url, params={"sort": "-properties.newest_alert_observation_time",
                                         "page[limit]": 1}, timeout=timeout)
        r["status"] = resp.status_code
        payload = resp.json()
        r["json_head"] = json.dumps(payload)[:600]
        r["frontier_mjd"] = _max_time_in(payload)
    except Exception as exc:                                      # noqa: BLE001
        r["error"] = f"{type(exc).__name__}: {exc}"[:400]
        r["frontier_mjd"] = None

    # 4. Lasair's ZTF instance, only with a token.
    token = (os.environ.get("LASAIR_ZTF_TOKEN") or os.environ.get("LASAIR_TOKEN") or "").strip()
    r = rec_for("lasair_ztf", LASAIR_ZTF_ENDPOINT)
    if not token:
        r["skipped"] = "no LASAIR_ZTF_TOKEN / LASAIR_TOKEN in the environment"
        r["frontier_mjd"] = None
    else:
        try:
            resp = requests.post(LASAIR_ZTF_ENDPOINT.rstrip("/") + "/query/",
                                 headers={"Authorization": f"Token {token}"},
                                 data={"selected": "MAX(objects.jdmax) AS jdmax",
                                       "tables": "objects", "conditions": "",
                                       "limit": 1, "format": "json"},
                                 timeout=timeout)
            r["status"] = resp.status_code
            payload = resp.json()
            r["json_head"] = json.dumps(payload)[:600]
            r["frontier_mjd"] = _max_time_in(payload)
        except Exception as exc:                                  # noqa: BLE001
            r["error"] = f"{type(exc).__name__}: {exc}"[:400]
            r["frontier_mjd"] = None

    out["status"] = ztf_status(out["sources"])
    return out


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def decide(alerce: dict, fink: dict, lasair: dict, ztf_probe: dict | None = None) -> dict:
    """Name the cause, or say plainly that the evidence cannot name it.

    ``UNDETERMINED_SINGLE_SOURCE`` is a real outcome, not a failure to try: a
    verdict of SKY_STOPPED resting on the only broker we can reach would be the
    same single-source claim this check exists to break.
    """
    a = alerce.get("frontier_mjd")
    others = {k: v for k, v in (("fink", fink.get("frontier_mjd")),
                                ("lasair", lasair.get("frontier_mjd")))
              if v is not None}

    ahead = {k: round(v - a, 3) for k, v in others.items()
             if a is not None and v - a > AHEAD_TOLERANCE_DAYS}

    # The controls: is each service demonstrably current on data that is NOT the
    # LSST stream?  A broker that is stale everywhere proves nothing about Rubin.
    controls = {}
    ztf = (fink.get("ztf_control") or {}).get("frontier_mjd")
    if ztf is not None:
        controls["fink_ztf"] = {"frontier_mjd": ztf, "frontier_utc": _iso(ztf),
                                "days_behind_now": round(_now_mjd() - ztf, 2)}
    other_survey = alerce.get("newest_any_survey_mjd")
    if other_survey is not None:
        controls["alerce_newest_survey"] = {
            "frontier_mjd": other_survey, "frontier_utc": _iso(other_survey),
            "days_behind_now": round(_now_mjd() - other_survey, 2)}
    # The ZTF probe's sources are controls of the same kind: a public endpoint
    # current on ZTF proves the wider alert infrastructure is alive.
    for name, fr in ((ztf_probe or {}).get("status") or {}).get("by_source", {}).items():
        if fr is not None:
            controls[f"ztf_{name}"] = {"frontier_mjd": fr, "frontier_utc": _iso(fr),
                                       "days_behind_now": round(_now_mjd() - fr, 2)}
    live_controls = [k for k, c in controls.items() if c["days_behind_now"] <= 7.0]

    if a is None and not others:
        verdict, why = "NO_BROKER_REACHED", (
            "No broker answered. This is a network or service failure on our "
            "side of the question and says nothing about Rubin.")
    elif ahead:
        verdict, why = "MIRROR_STALLED", (
            "Another broker holds LSST epochs newer than ALeRCE's by "
            + ", ".join(f"{k} +{d} d" for k, d in ahead.items())
            + ". Rubin is producing alerts that our channels are not seeing; "
              "the mirror, not the sky, is what stopped.")
    elif others:
        verdict, why = "SKY_STOPPED", (
            "Every broker reached stops on the same night. The alert stream "
            "itself stopped, so the channels' nulls mean 'no new sky', and no "
            "change to the broker path recovers data that was never taken.")
        if live_controls:
            why += (" Corroborated: " + ", ".join(live_controls) + " "
                    + ("is" if len(live_controls) == 1 else "are")
                    + " current on non-LSST data, so the brokers themselves are "
                      "alive and simply have no LSST alerts to serve.")
    else:
        verdict, why = "UNDETERMINED_SINGLE_SOURCE", (
            "Only ALeRCE answered, so a stalled mirror and a stopped stream "
            "remain indistinguishable. Do not read this as SKY_STOPPED.")

    out = {"verdict": verdict, "why": why,
           "alerce_frontier_mjd": a, "alerce_frontier_utc": _iso(a),
           "other_brokers": {k: {"frontier_mjd": v, "frontier_utc": _iso(v)}
                             for k, v in others.items()},
           "brokers_ahead_days": ahead,
           "controls": controls, "live_controls": live_controls,
           "lag_days": (round(_now_mjd() - a, 2) if a is not None else None)}
    if ztf_probe is not None and ztf_probe.get("status"):
        # Kept as its own block, never folded into the Rubin verdict: a live ZTF
        # says nothing about LSST, and the two questions have different readers.
        out["ztf"] = dict(ztf_probe["status"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/rubin_outage")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[outage] querying ALeRCE ...", flush=True)
    alerce = probe_alerce(args.timeout)
    print(f"[outage] ALeRCE frontier: {alerce.get('frontier_mjd')} "
          f"({_iso(alerce.get('frontier_mjd'))})", flush=True)

    print("[outage] querying Fink LSST ...", flush=True)
    fink = probe_fink(min(args.timeout, 120.0))
    print(f"[outage] Fink LSST last night: {fink.get('last_night')} "
          f"(mjd {fink.get('frontier_mjd')})", flush=True)
    print(f"[outage] Fink ZTF control last night: "
          f"{(fink.get('ztf_control') or {}).get('last_night')}", flush=True)

    print("[outage] querying Lasair ...", flush=True)
    lasair = probe_lasair(min(args.timeout, 120.0))

    print("[outage] asking the public ZTF endpoints for their newest epoch ...", flush=True)
    ztf = probe_ztf(min(args.timeout, 120.0))
    zs = ztf.get("status") or {}
    print(f"[outage] ZTF: {zs.get('status')} newest {zs.get('newest_utc')} "
          f"via {zs.get('newest_source')} ({zs.get('days_behind_now')} d behind now)",
          flush=True)
    for name, src in (ztf.get("sources") or {}).items():
        print(f"[outage]   ztf/{name}: status={src.get('status')} "
              f"frontier={_iso(src.get('frontier_mjd'))} "
              f"{('error=' + str(src.get('error'))[:120]) if src.get('error') else ''}"
              f"{('skipped=' + str(src.get('skipped'))) if src.get('skipped') else ''}",
              flush=True)

    rec = {"checked_at_utc": _utc(), "now_mjd": round(_now_mjd(), 5),
           "decision": decide(alerce, fink, lasair, ztf),
           "alerce": alerce, "fink": fink, "lasair": lasair, "ztf": ztf}

    path = out_dir / "brokers.json"
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(f"[outage] wrote {path}")
    d = rec["decision"]
    print(f"[outage] VERDICT {d['verdict']}: {d['why']}")
    for name, c in (d.get("controls") or {}).items():
        print(f"[outage] control {name}: {c['frontier_utc']} "
              f"({c['days_behind_now']} d behind now)")
    print("[outage] ALeRCE nightly LSST detections:")
    for night in (alerce.get("last_nights") or [])[-6:]:
        print(f"[outage]   {night['date']}  n={night['n']}")
    print("[outage] Fink nightly LSST alerts:")
    for night in (fink.get("nightly_counts") or [])[-6:]:
        print(f"[outage]   {night['date']}  n={night['n_alerts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
