"""What else can the two Rubin-starved channels be fed, and what does each source ACTUALLY serve?

Runner-only (every endpoint below is egress-blocked in the sandbox).

WHY A PROBE RATHER THAN A SHORTLIST.  A list of candidate surveys is worth very
little: what decides whether a channel can run on one is whether the specific
*column* carrying its observable exists, is populated, and is reachable in bulk
without a human in the loop.  This repository has already paid for that lesson
once --- TOCSIN's entire ADQL layer was inferred from published broker source and
had to be corrected against a live schema on the runner.  So this asks each
archive directly and commits the answer verbatim.

THE TWO QUESTIONS BEING SOURCED
-------------------------------
* **LOOM** screens the *ephemeris residual* --- observed minus predicted position
  of a known minor planet --- as a population observable.  A substitute must
  therefore deliver per-detection astrometry of known objects, densely enough to
  test a population.  Precision matters more than volume: Rubin's ``ephOffset*``
  is arcsecond-scale, and Gaia's asteroid astrometry is milliarcsecond-scale, so
  a smaller Gaia sample can carry a *stronger* version of the same test.
* **TOCSIN** screens fast achromatic flash/dip events on nearby stars against a
  cross-night ledger whose denominator is forced photometry. A substitute needs
  short exposures, a real revisit history, and bulk access to both.

Each probe is guarded and reports what it found rather than raising, so one dead
endpoint cannot cost the rest of the pass.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from seti.tocsin.brokers import ALERCE_TAP, _timeout_session  # noqa: E402

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

REST_ENDPOINTS = {
    # name: (url, what a 200 would mean for us)
    "atlas_forced_photometry": (
        "https://fallingstar-data.com/forcedphot/",
        "ATLAS forced photometry: full-sky, ~1 d cadence, 30 s exposures, "
        "southern coverage Rubin also has. Free account + token."),
    "asassn_skypatrol_v2": (
        "http://asas-sn.ifa.hawaii.edu/skypatrol/",
        "ASAS-SN Sky Patrol v2: full-sky light curves, ~1 d cadence, g<~18, "
        "public bulk API, no token."),
    "ztf_forced_photometry_ipac": (
        "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi",
        "ZTF forced photometry service (IPAC): the per-position revisit history "
        "that a ledger denominator needs. Account required."),
    "mast_catalogs": (
        "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
        "Pan-STARRS DR2 detections via MAST: deep northern per-epoch photometry."),
    "mpc_obs": (
        "https://www.minorplanetcenter.net/",
        "Minor Planet Center: every reported asteroid observation since the "
        "19th century --- the longest possible baseline for a residual test."),
    "jpl_sbdb": (
        "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=433",
        "JPL SBDB: non-gravitational parameters A1/A2/A3. Already LOOM's source "
        "for the momentum-ceiling screen, and NOT dependent on Rubin at all."),
}


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mjd_to_utc(mjd):
    try:
        v = float(mjd)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    dt = datetime.datetime(1858, 11, 17, tzinfo=datetime.timezone.utc) + \
        datetime.timedelta(days=v)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def tap_query(url: str, adql: str, timeout: float, maxrec: int = 50) -> dict:
    try:
        import pyvo
        svc = pyvo.dal.TAPService(url, session=_timeout_session(timeout))
        tab = svc.search(adql, maxrec=maxrec).to_table()
        return {"rows": len(tab),
                "data": [{c: str(row[c]) for c in tab.colnames} for row in tab][:maxrec]}
    except Exception as exc:                                      # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:600], "adql": adql}


def probe_alerce_ztf(timeout: float) -> dict:
    """Is a second survey already sitting in the broker TOCSIN reads?

    If ALeRCE's mirror carries a current ZTF feed, TOCSIN's substitute costs a
    parameter change rather than an acquisition layer: same TAP service, same
    ADQL, same normalisation, different survey id. That makes it the cheapest
    thing on this list by a wide margin, and the first thing worth knowing.
    """
    out = {"service": ALERCE_TAP}
    out["survey_currency"] = tap_query(
        ALERCE_TAP,
        "SELECT sid, tid, COUNT(*) AS n, MIN(firstmjd) AS first_min, "
        "MAX(lastmjd) AS last_max FROM alerce_tap.object GROUP BY sid, tid",
        timeout, maxrec=50)
    rows = out["survey_currency"].get("data") or []
    surveys = []
    for r in rows:
        surveys.append({"sid": r.get("sid"), "tid": r.get("tid"), "n": r.get("n"),
                        "last_mjd": r.get("last_max"),
                        "last_utc": _mjd_to_utc(r.get("last_max"))})
    out["surveys"] = surveys
    return out


def probe_gaia_sso(timeout: float) -> dict:
    """Gaia's asteroid astrometry --- LOOM's observable at ~1000x the precision.

    Rubin delivers the residual pre-computed and arcsecond-scale; Gaia delivers
    positions at milliarcsecond scale, from which the residual is computed
    against the same JPL orbits LOOM already uses. What has to be established
    here is which tables exist across DR3 and the Focused Product Release, what
    columns they carry (an epoch, a position, an uncertainty, and ideally a
    residual or a predicted position), and how many rows there are.
    """
    out = {"service": GAIA_TAP}
    out["tables"] = tap_query(
        GAIA_TAP,
        "SELECT table_name FROM TAP_SCHEMA.tables "
        "WHERE table_name LIKE '%sso%' ORDER BY table_name", timeout, maxrec=50)
    for tbl in ("gaiadr3.sso_observation", "gaiadr3.sso_source",
                "gaiafpr.sso_observation", "gaiafpr.sso_source"):
        out[f"columns:{tbl}"] = tap_query(
            GAIA_TAP,
            "SELECT column_name, datatype, unit, ucd, description "
            f"FROM TAP_SCHEMA.columns WHERE table_name = '{tbl}' "
            "ORDER BY column_name", timeout, maxrec=200)
        out[f"count:{tbl}"] = tap_query(
            GAIA_TAP, f"SELECT COUNT(*) AS n FROM {tbl}", timeout, maxrec=5)
    return out


def probe_vizier_sso(timeout: float) -> dict:
    """Published residual catalogues, if any, without a bespoke pipeline."""
    return {"service": VIZIER_TAP,
            "tables": tap_query(
                VIZIER_TAP,
                "SELECT table_name, description FROM TAP_SCHEMA.tables "
                "WHERE description LIKE '%asteroid%' "
                "AND (description LIKE '%astrometr%' OR description LIKE '%residual%') "
                "ORDER BY table_name", timeout, maxrec=60)}


def probe_rest(timeout: float) -> dict:
    out = {}
    for name, (url, why) in REST_ENDPOINTS.items():
        rec = {"url": url, "why_it_matters": why}
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "seti-research/1.0"})
            rec["status"] = resp.status_code
            rec["bytes"] = len(resp.text or "")
            rec["head"] = (resp.text or "")[:300]
        except Exception as exc:                                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out[name] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/survey_recon")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = {"probed_at_utc": _utc()}
    print("[recon] ALeRCE: which surveys, how current ...", flush=True)
    rec["alerce_surveys"] = probe_alerce_ztf(args.timeout)
    print("[recon] Gaia SSO tables ...", flush=True)
    rec["gaia_sso"] = probe_gaia_sso(args.timeout)
    print("[recon] VizieR asteroid-astrometry catalogues ...", flush=True)
    rec["vizier_sso"] = probe_vizier_sso(args.timeout)
    print("[recon] REST endpoints ...", flush=True)
    rec["rest"] = probe_rest(min(args.timeout, 60.0))

    path = out_dir / "recon.json"
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(f"[recon] wrote {path}")

    print("\n=== ALeRCE surveys ===")
    for s in rec["alerce_surveys"].get("surveys") or []:
        print(f"  sid={s['sid']} tid={s['tid']} n={s['n']} newest={s['last_utc']}")
    err = (rec["alerce_surveys"].get("survey_currency") or {}).get("error")
    if err:
        print("  error:", err[:300])

    print("\n=== Gaia SSO ===")
    for row in (rec["gaia_sso"].get("tables") or {}).get("data") or []:
        print("  table:", row.get("table_name"))
    for tbl in ("gaiadr3.sso_observation", "gaiadr3.sso_source",
                "gaiafpr.sso_observation", "gaiafpr.sso_source"):
        cnt = rec["gaia_sso"].get(f"count:{tbl}", {})
        n = (cnt.get("data") or [{}])[0].get("n") if not cnt.get("error") else cnt["error"][:80]
        cols = (rec["gaia_sso"].get(f"columns:{tbl}") or {}).get("data") or []
        print(f"  {tbl}: n={n}, {len(cols)} columns")
        for c in cols:
            name, ucd = c.get("column_name", ""), (c.get("ucd") or "")
            if any(k in name.lower() for k in ("ra", "dec", "epoch", "time",
                                               "residual", "error", "number",
                                               "denomination", "outcome")):
                print(f"      {name:<28} {c.get('unit','') or '':<10} {ucd}")

    print("\n=== VizieR asteroid astrometry ===")
    for row in (rec["vizier_sso"].get("tables") or {}).get("data") or []:
        print(f"  {row.get('table_name')}: {str(row.get('description'))[:110]}")

    print("\n=== REST endpoints ===")
    for name, r in rec["rest"].items():
        print(f"  {name}: {r.get('status') or r.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
