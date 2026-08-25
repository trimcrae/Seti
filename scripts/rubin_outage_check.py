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

    rows = run("max_mjd_ss_detection",
               "SELECT MAX(d.mjd) AS mjd_max FROM alerce_tap.lsst_ss_detection AS ss "
               "JOIN alerce_tap.detection AS d ON d.oid = ss.oid "
               "AND d.sid = ss.sid AND d.measurement_id = ss.measurement_id",
               maxrec=5)
    out["frontier_ss_mjd"] = _num(rows[0].get("mjd_max")) if rows else None

    # Does ANY table in the mirror hold something newer?  If ALeRCE's ZTF side is
    # current while its LSST side is frozen, the service is alive and the LSST
    # ingest specifically is not --- a different failure from ALeRCE being down.
    rows = run("max_mjd_any_detection",
               "SELECT MAX(mjd) AS mjd_max FROM alerce_tap.detection", maxrec=5)
    out["frontier_any_mjd"] = _num(rows[0].get("mjd_max")) if rows else None

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
    """Ask Fink's LSST portal for its newest alert.

    Fink's REST API is POST-with-JSON, but the endpoint set of the LSST portal
    (as opposed to the long-standing ZTF one) is not verified from inside the
    sandbox, so each candidate endpoint is tried both ways and the raw response
    recorded.  A 404 here is itself informative and must not look like an outage.
    """
    import requests

    out: dict = {"url": FINK_LSST_API, "attempts": {}}
    attempts = [
        ("latests_POST", "POST", "/api/v1/latests",
         {"class": "allclasses", "n": 20, "output-format": "json"}),
        ("statistics_POST", "POST", "/api/v1/statistics",
         {"date": "", "output-format": "json"}),
        ("latests_GET", "GET", "/api/v1/latests",
         {"class": "allclasses", "n": 20, "output-format": "json"}),
        ("schema_GET", "GET", "/api/v1/schema", None),
    ]
    newest: float | None = None
    for name, method, path, body in attempts:
        url = FINK_LSST_API.rstrip("/") + path
        rec: dict = {"method": method, "url": url}
        try:
            if method == "POST":
                resp = requests.post(url, json=body, timeout=timeout)
            else:
                resp = requests.get(url, params=body, timeout=timeout)
            rec["status"] = resp.status_code
            text = resp.text or ""
            rec["bytes"] = len(text)
            try:
                payload = resp.json()
            except Exception:                                     # noqa: BLE001
                payload = None
                rec["text_head"] = text[:800]
            if payload is not None:
                rec["json_head"] = json.dumps(payload)[:2000]
                cand = _max_time_in(payload)
                if cand is not None:
                    rec["newest_mjd"] = cand
                    newest = cand if newest is None else max(newest, cand)
        except Exception as exc:                                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        out["attempts"][name] = rec
    out["frontier_mjd"] = newest
    return out


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
# Verdict
# --------------------------------------------------------------------------
def decide(alerce: dict, fink: dict, lasair: dict) -> dict:
    """Name the cause, or say plainly that one broker cannot name it.

    ``UNDETERMINED_SINGLE_SOURCE`` is a real outcome, not a failure to try: a
    verdict of SKY_STOPPED resting on the only broker we can reach would be the
    same single-source claim the check was written to break.
    """
    a = alerce.get("frontier_mjd")
    others = {k: v for k, v in (("fink", fink.get("frontier_mjd")),
                                ("lasair", lasair.get("frontier_mjd")))
              if v is not None}

    ahead = {k: round(v - a, 3) for k, v in others.items()
             if a is not None and v - a > AHEAD_TOLERANCE_DAYS}

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
    else:
        verdict, why = "UNDETERMINED_SINGLE_SOURCE", (
            "Only ALeRCE answered, so a stalled mirror and a stopped stream "
            "remain indistinguishable. Do not read this as SKY_STOPPED.")

    return {"verdict": verdict, "why": why,
            "alerce_frontier_mjd": a, "alerce_frontier_utc": _iso(a),
            "other_brokers": {k: {"frontier_mjd": v, "frontier_utc": _iso(v)}
                              for k, v in others.items()},
            "brokers_ahead_days": ahead,
            "lag_days": (round(_now_mjd() - a, 2) if a is not None else None)}


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
    print(f"[outage] Fink frontier: {fink.get('frontier_mjd')} "
          f"({_iso(fink.get('frontier_mjd'))})", flush=True)

    print("[outage] querying Lasair ...", flush=True)
    lasair = probe_lasair(min(args.timeout, 120.0))

    rec = {"checked_at_utc": _utc(), "now_mjd": round(_now_mjd(), 5),
           "decision": decide(alerce, fink, lasair),
           "alerce": alerce, "fink": fink, "lasair": lasair}

    path = out_dir / "brokers.json"
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(f"[outage] wrote {path}")
    d = rec["decision"]
    print(f"[outage] VERDICT {d['verdict']}: {d['why']}")
    for night in (alerce.get("last_nights") or [])[-6:]:
        print(f"[outage]   {night['date']}  n={night['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
