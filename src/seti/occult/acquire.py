"""OCCULT acquisition: event catalogues and per-event photometry.

Formats and hosts are the ones the runner probe measured (runs 36003872955
and the round-2 probe, ``results/occult/probe.json``) --- not remembered:

* OGLE-IV EWS.  The per-year catalogue is
  ``https://www.astrouw.edu.pl/ogle/ogle4/ews/<yr>/lenses.par`` (whitespace
  table: Event Field StarNo RA Dec Tmax(HJD) Tmax(UT) tau umin Amax Dmag fbl
  I_bl I0); the photometry is ``.../<yr>/blg-NNNN/phot.dat`` (HJD, I, sigma_I,
  seeing, sky), I band only, with several seasons of baseline.  The
  ``ogle.astrouw.edu.pl`` host serves the HTML pages only (404 on the data).
  EWS ran 2011-2019 and 2022-2025 (no 2020/2021 season pages).
* KMTNet.  ``https://kmtnet.kasi.re.kr/~ulens/event/<yr>/listpage.dat`` lists
  every event of a season (name, star id, two classification codes, RA, Dec,
  t0 [HJD-2450000], tE, u0, Isource, Ibase, Icat, a code, A_I, related
  OGLE/MOA/PRIME events); ``.../<yr>/data/KB<yy><nnnn>/pysis/pysis.tar.gz``
  holds every site/field/band pySIS file (``KMT{A,C,S}<ff>_{I,V}.pysis``:
  HJD, Delta_flux, flux_err, mag, mag_err, fwhm, sky, secz).  Directory
  listings are forbidden (403); the tarball needs no listing.
* MOA.  The alert pages that the literature cites 404 or do not resolve from
  a runner (``www.massey.ac.nz/~iabond``: 404; ``it019909.massey.ac.nz``:
  no DNS).  MOA is recorded as unreached, not silently dropped.

KMTNet data policy: a season is proprietary until 1 July of the next year.
Seasons inside that window are excluded from the search (``public_seasons``).
"""

from __future__ import annotations

import io
import math
import re
import tarfile
import time
from datetime import datetime, timezone

import numpy as np

OGLE_DATA = "https://www.astrouw.edu.pl/ogle/ogle4/ews/"
KMT_ROOT = "https://kmtnet.kasi.re.kr/~ulens/event/"
OGLE_SEASONS = tuple(y for y in range(2011, 2026) if y not in (2020, 2021))
KMT_FIRST_SEASON = 2016


def public_seasons(now: datetime | None = None, last: int = 2025) -> list:
    """KMTNet seasons whose proprietary period (until 1 July of the next year) is over."""
    now = now or datetime.now(timezone.utc)
    out = []
    for y in range(KMT_FIRST_SEASON, last + 1):
        if now >= datetime(y + 1, 7, 1, tzinfo=timezone.utc):
            out.append(y)
    return out


# --------------------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------------------

def _sexa(s: str, hours: bool) -> float | None:
    try:
        sign = -1.0 if s.strip().startswith("-") else 1.0
        p = [abs(float(x)) for x in s.strip().lstrip("+-").split(":")]
        v = p[0] + p[1] / 60.0 + p[2] / 3600.0
        return sign * v * (15.0 if hours else 1.0)
    except (ValueError, IndexError):
        return None


def hjd_prime(t):
    """Any HJD convention to HJD - 2450000 (the photometry's convention).

    KMTNet ``listpage.dat`` quotes t0 as HJD - 2450000 up to 2022 (``8540.21``)
    and as HJD - 2400000 from 2023 (``60748.36``); OGLE quotes full HJD.
    Run 36015751048 lost every 2023-2025 KMTNet control to this: the event
    window was cut 50,000 d away from the data.
    """
    if t is None:
        return None
    t = float(t)
    if t > 2400000.0:
        return t - 2450000.0
    if t > 40000.0:
        return t - 50000.0
    return t


def _hjd_prime_arr(t):
    t = np.asarray(t, dtype=float)
    return np.where(t > 2400000.0, t - 2450000.0, np.where(t > 40000.0, t - 50000.0, t))


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_ogle_lenses(text: str, season: int) -> list:
    """Rows of an OGLE-IV ``lenses.par``; t0 in HJD-2450000."""
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 14 or not re.match(r"^\d{4}-BLG-\d{3,4}$", p[0]):
            continue
        t0 = _f(p[5])
        rows.append({
            "survey": "OGLE", "season": int(season), "name": f"OGLE-{p[0]}",
            "id": p[0].split("-")[-1], "field": p[1],
            "ra": _sexa(p[3], True), "dec": _sexa(p[4], False),
            "t0": hjd_prime(t0),
            "tE": _f(p[7]), "u0": _f(p[8]), "Amax": _f(p[9]), "fbl": _f(p[11]),
            "I0": _f(p[13]),
        })
    return rows


def parse_kmt_listpage(text: str, season: int) -> list:
    """Rows of a KMTNet ``listpage.dat``.

    Columns (by position): name, field.starid, class_a, class_b, RA, Dec, t0,
    tE, u0, Isource, Ibase, Icat, code, A_I, then any related-event tokens
    (OB..., MB..., PB..., KB...).  The two class codes are kept verbatim.
    """
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 10 or not re.match(r"^KMT-\d{4}-BLG-\d{4}$", p[0]):
            continue
        # The class columns are not always both present, so the coordinates are
        # located by their shape and every later column is read relative to them.
        ira = next((i for i in range(2, len(p) - 1)
                    if re.match(r"^\d{1,2}:\d{2}:\d{2}(\.\d*)?$", p[i])
                    and re.match(r"^[+-]?\d{1,2}:\d{2}:\d{2}(\.\d*)?$", p[i + 1])), None)
        if ira is None:
            continue
        num = p[0].split("-")[-1]
        q = p[ira + 2:]
        related = [x for x in q if re.match(r"^[A-Z]{2}\d{5,6}$", x)]
        rows.append({
            "survey": "KMT", "season": int(season), "name": p[0], "id": num,
            "field": p[1].split(".")[0], "class_a": p[2] if ira > 2 else None,
            "class_b": p[3] if ira > 3 else None,
            "ra": _sexa(p[ira], True), "dec": _sexa(p[ira + 1], False),
            "t0": hjd_prime(_f(q[0])) if len(q) > 0 else None,
            "tE": _f(q[1]) if len(q) > 1 else None,
            "u0": _f(q[2]) if len(q) > 2 else None,
            "Isource": _f(q[3]) if len(q) > 3 else None, "Ibase": _f(q[4]) if len(q) > 4 else None,
            "A_I": _f(q[7]) if len(q) > 7 else None,
            "related": related,
        })
    return rows


def ogle_name_from_token(tok: str) -> str | None:
    """``OB190161`` -> ``OGLE-2019-BLG-0161`` (KMTNet's related-event shorthand)."""
    m = re.match(r"^OB(\d{2})(\d{4})$", tok)
    if not m:
        return None
    return f"OGLE-20{m.group(1)}-BLG-{m.group(2)}"


def parse_phot_dat(text: str) -> dict:
    """OGLE ``phot.dat``: HJD, I, sigma, seeing, sky (whitespace); t in HJD-2450000."""
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 3 or line.lstrip().startswith("#"):
            continue
        v = [_f(x) for x in p[:5]]
        if v[0] is None or v[1] is None or v[2] is None:
            continue
        rows.append(v + [None] * (5 - len(v)))
    if not rows:
        return {"t": np.zeros(0), "mag": np.zeros(0), "err": np.zeros(0),
                "seeing": np.zeros(0), "sky": np.zeros(0)}
    a = np.array([[np.nan if x is None else x for x in r] for r in rows], dtype=float)
    t = _hjd_prime_arr(a[:, 0])
    return {"t": t, "mag": a[:, 1], "err": a[:, 2], "seeing": a[:, 3], "sky": a[:, 4]}


def parse_pysis(text: str) -> dict:
    """KMTNet pySIS: HJD, Delta_flux, flux_err, mag, mag_err, fwhm, sky, secz."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 3:
            continue
        v = [_f(x) for x in p[:8]]
        if v[0] is None or v[1] is None or v[2] is None:
            continue
        rows.append(v + [None] * (8 - len(v)))
    keys = ("t", "flux", "err", "mag", "mag_err", "fwhm", "sky", "secz")
    if not rows:
        return {k: np.zeros(0) for k in keys}
    a = np.array([[np.nan if x is None else x for x in r] for r in rows], dtype=float)
    out = {k: a[:, i] for i, k in enumerate(keys)}
    out["t"] = _hjd_prime_arr(out["t"])
    return out


_PYSIS_NAME = re.compile(r"KMT([ACS])(\d{2})_([IV])\.pysis$")


def parse_pysis_tar(blob: bytes) -> list:
    """Every pySIS file in a KMTNet ``pysis.tar.gz``: [(dataset, table), ...]."""
    out = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            g = _PYSIS_NAME.search(m.name)
            if not g:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            tab = parse_pysis(fh.read().decode("utf-8", errors="replace"))
            site = "KMT" + g.group(1)
            out.append(({"name": f"{site}{g.group(2)}_{g.group(3)}", "site": site,
                         "field": g.group(2), "band": g.group(3)}, tab))
    return out


def quality_mask_kmt(tab: dict, fwhm_max_factor: float = 1.8, sky_max_factor: float = 3.0) -> np.ndarray:
    """Standard KMTNet cuts: finite, positive error, no extreme seeing or sky."""
    ok = np.isfinite(tab["t"]) & np.isfinite(tab["flux"]) & np.isfinite(tab["err"]) & (tab["err"] > 0)
    fw = tab.get("fwhm")
    if fw is not None and np.isfinite(fw).sum() > 10:
        med = np.nanmedian(fw[ok]) if ok.any() else np.nan
        if np.isfinite(med):
            ok &= ~(fw > fwhm_max_factor * med)
    sk = tab.get("sky")
    if sk is not None and np.isfinite(sk).sum() > 10:
        med = np.nanmedian(sk[ok]) if ok.any() else np.nan
        if np.isfinite(med) and med > 0:
            ok &= ~(sk > sky_max_factor * med)
    return ok


def quality_mask_ogle(tab: dict, seeing_max_factor: float = 1.8) -> np.ndarray:
    ok = np.isfinite(tab["t"]) & np.isfinite(tab["mag"]) & np.isfinite(tab["err"]) & (tab["err"] > 0)
    ok &= (tab["mag"] > 5) & (tab["mag"] < 25)
    se = tab.get("seeing")
    if se is not None and np.isfinite(se).sum() > 10:
        med = np.nanmedian(se[ok]) if ok.any() else np.nan
        if np.isfinite(med):
            ok &= ~(se > seeing_max_factor * med)
    return ok


def window_mask(t, t0, te, n_te: float = 8.0, min_days: float = 120.0, max_days: float = 400.0):
    """The event window: |t - t0| < clip(n_te tE, min_days, max_days)."""
    if t0 is None or not np.isfinite(t0):
        return np.ones(np.size(t), dtype=bool)
    half = min(max(n_te * (te or 30.0), min_days), max_days)
    return np.abs(np.asarray(t) - t0) < half


# --------------------------------------------------------------------------------------
# Network (runner only)
# --------------------------------------------------------------------------------------

def session():
    import requests
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    s.headers["User-Agent"] = "seti-occult/1.0 (research; github.com/trimcrae/Seti)"
    s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
    return s


def get(s, url: str, tries: int = 4, timeout: float = 60.0, binary: bool = False):
    """GET with retries; returns (status, payload or None, error or None)."""
    err = None
    for k in range(tries):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code == 200:
                return 200, (r.content if binary else r.text), None
            if r.status_code in (403, 404):
                return r.status_code, None, f"HTTP {r.status_code}"
            err = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            err = repr(exc)[:200]
        time.sleep(1.5 * (2 ** k))
    return None, None, err


def fetch_catalogues(s, kmt_seasons, ogle_seasons=OGLE_SEASONS) -> dict:
    """Both surveys' event lists; per-season status recorded (never silently empty)."""
    rows, status = [], {}
    for y in ogle_seasons:
        code, txt, err = get(s, f"{OGLE_DATA}{y}/lenses.par")
        got = parse_ogle_lenses(txt, y) if txt else []
        status[f"OGLE_{y}"] = {"http": code, "n": len(got), "error": err}
        rows += got
    for y in kmt_seasons:
        code, txt, err = get(s, f"{KMT_ROOT}{y}/listpage.dat")
        got = parse_kmt_listpage(txt, y) if txt else []
        status[f"KMT_{y}"] = {"http": code, "n": len(got), "error": err}
        rows += got
    return {"rows": rows, "status": status}


def build_units(rows: list, match_arcsec: float = 2.0, match_days: float = 20.0) -> list:
    """One search unit per physical event: a KMT event with its OGLE partner, or OGLE alone.

    The partner comes from KMTNet's related-event token (``OByynnnn``) when it
    names an OGLE row of the catalogue; otherwise from a positional match
    (``match_arcsec``) with |t0 difference| < ``match_days``.
    """
    ogle = {r["name"]: r for r in rows if r["survey"] == "OGLE"}
    ogle_list = [r for r in rows if r["survey"] == "OGLE" and r["ra"] is not None
                 and r["dec"] is not None]
    used = set()
    units = []
    if ogle_list:
        ora = np.array([r["ra"] for r in ogle_list])
        odec = np.array([r["dec"] for r in ogle_list])
        ot0 = np.array([r["t0"] if r["t0"] is not None else np.nan for r in ogle_list])
    for r in rows:
        if r["survey"] != "KMT":
            continue
        partner = None
        for tok in r.get("related", []):
            nm = ogle_name_from_token(tok)
            if nm in ogle:
                partner = nm
                break
        if partner is None and ogle_list and r["ra"] is not None and r["dec"] is not None:
            d = np.hypot((ora - r["ra"]) * math.cos(math.radians(r["dec"])), odec - r["dec"]) * 3600
            dt = np.abs(ot0 - (r["t0"] or np.nan))
            ok = (d < match_arcsec) & (dt < match_days)
            if ok.any():
                partner = ogle_list[int(np.argmin(np.where(ok, d, np.inf)))]["name"]
        if partner is not None:
            used.add(partner)
        units.append({"unit": r["name"], "kmt": r, "ogle": ogle.get(partner) if partner else None})
    for nm, r in ogle.items():
        if nm not in used:
            units.append({"unit": nm, "kmt": None, "ogle": r})
    return units


def kmt_tar_url(row: dict) -> str:
    y = int(row["season"])
    return f"{KMT_ROOT}{y}/data/KB{str(y)[2:]}{row['id']}/pysis/pysis.tar.gz"


def ogle_phot_url(row: dict) -> str:
    return f"{OGLE_DATA}{int(row['season'])}/blg-{row['id']}/phot.dat"


def fetch_unit_series(s, unit: dict, conf: dict | None = None) -> dict:
    """Download and parse one unit's photometry; returns series for ``detect.build_event``.

    Every KMT pySIS dataset becomes one series in Delta_flux units; OGLE I
    becomes one series in flux relative to I = 18.  Points are restricted to
    the event window around the catalogue t0.  ``status`` records each
    download so a missing survey is visible, never silent.
    """
    from .detect import mag_to_flux

    conf = conf or {}
    series, status = [], {}
    ref = unit.get("kmt") or unit.get("ogle")
    t0 = ref.get("t0")
    te = ref.get("tE") or 30.0
    if unit.get("kmt"):
        code, blob, err = get(s, kmt_tar_url(unit["kmt"]), binary=True)
        status["kmt"] = {"http": code, "error": err, "bytes": len(blob) if blob else 0}
        if blob:
            try:
                tabs = parse_pysis_tar(blob)
                status["kmt"]["n_files"] = len(tabs)
                status["kmt"]["n_points_raw"] = int(sum(tab["t"].size for _, tab in tabs))
                kept = 0
                for d, tab in tabs:
                    ok = quality_mask_kmt(tab) & window_mask(tab["t"], t0, te)
                    if ok.sum() >= 5:
                        series.append((d, tab["t"][ok], tab["flux"][ok], tab["err"][ok]))
                        kept += int(ok.sum())
                status["kmt"]["n_points_kept"] = kept
            except (tarfile.TarError, OSError, EOFError) as exc:
                status["kmt"]["error"] = f"tar: {exc!r}"[:200]
    if unit.get("ogle"):
        code, txt, err = get(s, ogle_phot_url(unit["ogle"]))
        status["ogle"] = {"http": code, "error": err, "bytes": len(txt) if txt else 0}
        if txt:
            tab = parse_phot_dat(txt)
            ok = quality_mask_ogle(tab) & window_mask(tab["t"], t0, te)
            status["ogle"]["n_points_raw"] = int(tab["t"].size)
            status["ogle"]["n_points_kept"] = int(ok.sum())
            if ok.sum() >= 5:
                f, e = mag_to_flux(tab["mag"][ok], tab["err"][ok])
                series.append(({"name": "OGLE_I", "site": "OGLE", "field": unit["ogle"]["field"],
                                "band": "I"}, tab["t"][ok], f, e))
    return {"series": series, "status": status}
