"""Runner-only acquisition for SLAG-WD: PEWDD, and the diffusion-timescale tables.

Nothing here runs in a test: every network function takes an injectable
``fetch_fn(url) -> str`` or ``query_fn(adql) -> DataFrame`` so the offline
suite can script a healthy, an empty and a dead archive.  The pure pieces
(column-role resolution, atmosphere resolution, panel building, the
timescale-table parser) are what the tests exercise.

Routes to PEWDD (Williams et al. 2024, A&A 691, A352), in order:

1. VizieR ``J/A+A/691/A352/pewdd`` over the route ladder of
   ``seti.metronome.acquire.vizier_table`` (TAP mirrors → ASU → astroquery),
   with the table's column names, units and descriptions read from the ASU
   metadata first so that no column is addressed by a guessed name;
2. the GitHub repository ``jamietwilliams/PEWDD`` (tree listing through the
   API, then the raw CSV), which carries the provenance columns VizieR may
   not.

Both are recorded; the one that served the rows is named in
``acquire.json["pewdd"]["route"]``.  PEWDD logs abundances as log(Z/H) in
hydrogen atmospheres and log(Z/He) in helium atmospheres; the reference is
resolved per row (an explicit atmosphere column when the table has one, the
spectral type otherwise) and recorded --- the science only ever uses element
RATIOS, in which the reference cancels, but the sinking timescales do not.
"""

from __future__ import annotations

import io
import json
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..metronome.acquire import (
    ROUTE_NONE,
    VIZIER_ASU,
    VizierRouteError,
    asu_catalogue_tables,
    asu_table_columns,
    list_tables,
    reset_route_state,
    route_log_summary,
    vizier_table,
)
from .misfit import Panel
from .sinking import parse_timescale_table

STATUS_OK = "OK"
STATUS_FAILED = "QUERY_FAILED"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"

ATM_H = "H"
ATM_HE = "He"
ATM_UNKNOWN = "unknown"

#: VizieR column prefixes that mark a companion column, never a value.
_VIZIER_PREFIXES = ("e_", "l_", "f_", "n_", "q_", "r_", "u_", "E_", "x_")


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------
@dataclass
class AcquisitionLog:
    stages: list[dict] = field(default_factory=list)
    prefix: str = "slag/acquire"

    def record(self, stage: str, what: str, *, rows: int | None = None,
               error: str | None = None, extra: dict | None = None) -> str:
        if error is not None or rows is None:
            status = STATUS_FAILED
        elif rows == 0:
            status = STATUS_ZERO
        else:
            status = STATUS_OK
        rec = {"stage": stage, "status": status, "rows": int(rows or 0), "what": str(what)[:2000]}
        if error:
            rec["error"] = str(error)[:2000]
        if extra:
            rec.update(extra)
        self.stages.append(rec)
        print(f"[{self.prefix}] {stage}: {status} rows={rec['rows']}"
              + (f" error={rec.get('error')}" if error else ""))
        return status

    def as_dict(self) -> dict:
        n_fail = sum(1 for s in self.stages if s["status"] == STATUS_FAILED)
        n_zero = sum(1 for s in self.stages if s["status"] == STATUS_ZERO)
        n_ok = sum(1 for s in self.stages if s["status"] == STATUS_OK)
        return {"stages": self.stages, "n_stages": len(self.stages), "n_ok": n_ok,
                "n_query_failed": n_fail, "n_query_returned_zero_rows": n_zero,
                "any_query_failed": bool(n_fail > 0), "routes": route_log_summary()}

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, default=str))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def http_text(url: str, *, timeout: float = 180.0) -> str:
    """GET a URL and return its text (runner only)."""
    import requests  # noqa: PLC0415  runner-only; keeps the module importable offline

    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "seti-slag/1.0 (+github actions; astronomy)",
                              "Accept": "application/vnd.github+json, text/plain, */*"})
    r.raise_for_status()
    return r.text


def fetch_text(url: str, *, fetch_fn=None, retries: int = 3, timeout: float = 180.0,
               log: AcquisitionLog | None = None, stage: str = "fetch") -> str | None:
    fn = fetch_fn or (lambda u: http_text(u, timeout=timeout))
    last = None
    for attempt in range(max(int(retries), 1)):
        try:
            text = fn(url)
            if text is None:
                raise RuntimeError("fetch returned None")
            text = str(text)
            if log is not None:
                log.record(stage, url, rows=len(text.splitlines()),
                           extra={"bytes": len(text.encode("utf-8", "replace"))})
            return text
        except Exception as exc:                                  # noqa: BLE001
            last = exc
            if attempt + 1 < retries:
                _time.sleep(min(2.0 * (attempt + 1), 10.0))
    if log is not None:
        log.record(stage, url, error=repr(last))
    return None


# ---------------------------------------------------------------------------
# GitHub (second route for PEWDD; the only route for PyllutedWD's files)
# ---------------------------------------------------------------------------
def github_tree(repo: str, *, fetch_fn=None, log: AcquisitionLog | None = None,
                branches=("main", "master", "HEAD")) -> tuple[list[dict], str | None]:
    """Every path in a repository, via the git-trees API; ``(entries, branch)``."""
    for br in branches:
        url = f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1"
        text = fetch_text(url, fetch_fn=fetch_fn, retries=2, log=log, stage=f"github_tree:{repo}")
        if not text:
            continue
        try:
            d = json.loads(text)
        except Exception:                                         # noqa: BLE001
            continue
        tree = d.get("tree") or []
        if tree:
            return [{"path": t.get("path"), "size": t.get("size"), "type": t.get("type")}
                    for t in tree], br
    return [], None


def github_raw_url(repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def pick_pewdd_csv(entries: list[dict], patterns=(r"(?i)pewdd.*\.csv$", r"(?i)\.csv$")
                   ) -> list[dict]:
    """CSV files of the PEWDD repository, the most PEWDD-looking and largest first."""
    out = []
    for pat in patterns:
        hits = [e for e in entries if e.get("type") == "blob" and re.search(pat, str(e["path"]))]
        hits.sort(key=lambda e: -(e.get("size") or 0))
        for h in hits:
            if h not in out:
                out.append(h)
    return out


def pick_timescale_files(entries: list[dict], max_bytes: int = 5_000_000,
                         pattern=r"(?i)(timescale|diffusion|sinking|tau|settling)") -> list[dict]:
    hits = [e for e in entries if e.get("type") == "blob" and re.search(pattern, str(e["path"]))
            and (e.get("size") or 0) <= max_bytes
            and re.search(r"(?i)\.(csv|txt|dat|tsv|tab|json)$", str(e["path"]))]
    hits.sort(key=lambda e: (e.get("size") or 0))
    return hits


# ---------------------------------------------------------------------------
# column-role resolution
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def _is_companion(col: str) -> bool:
    return any(str(col).startswith(p) for p in _VIZIER_PREFIXES)


def element_value_column(el: str, columns: list[str], descriptions: dict | None = None
                         ) -> str | None:
    """The column carrying log(el/H(e)), by name first, then by description."""
    descriptions = descriptions or {}
    name_pats = [
        rf"^{el}$", rf"^log{el}$", rf"^log\(?{el}/?H\(?e\)?\)?$", rf"^{el}/H\(?e\)?$",
        rf"^{el}_?H\(?e\)?$", rf"^\[{el}/H\(?e\)?\]$", rf"^{el}He$", rf"^{el}/He$",
        rf"^{el}_He$", rf"^log\(?{el}/He\)?$", rf"^{el}Hx$", rf"^{el}_x$", rf"^{el}/X$",
        rf"^log{el}H$", rf"^log{el}He$", rf"^{el}abund$", rf"^A{el}$",
    ]
    for col in columns:
        if _is_companion(col):
            continue
        n = _norm(col)
        if any(re.fullmatch(p, n, flags=re.IGNORECASE) for p in name_pats):
            return col
    for col in columns:
        if _is_companion(col):
            continue
        d = str(descriptions.get(col, ""))
        if re.search(rf"(?i)(log\s*\(?\s*|abundance[^,;]*\b){el}\s*/\s*H", d) \
                or re.search(rf"(?i)\b{el}\b\s*/\s*H\s*\(?\s*e\s*\)?", d) \
                or re.search(rf"(?i)abundance\s+of\s+{el}\b", d):
            if re.search(r"(?i)(error|uncertaint|limit|flag)", d):
                continue
            return col
    return None


def companion_column(value_col: str, columns: list[str], kind: str,
                     descriptions: dict | None = None) -> str | None:
    """The error (``kind='error'``) or limit-flag (``kind='limit'``) column of ``value_col``."""
    descriptions = descriptions or {}
    base = value_col
    if kind == "error":
        cands = [f"e_{base}", f"E_{base}", f"{base}_err", f"{base}err", f"err_{base}",
                 f"e{base}", f"{base}_e", f"sigma_{base}", f"{base}_error", f"{base} error",
                 f"d{base}", f"{base}_unc"]
        rx = re.compile(r"(?i)(error|uncertaint|sigma)")
    else:
        cands = [f"l_{base}", f"{base}_flag", f"f_{base}", f"{base}_ul", f"{base}_limit",
                 f"lim_{base}", f"{base}_lim", f"u_{base}", f"{base} upper limit",
                 f"{base}_upper_limit", f"{base}ul"]
        rx = re.compile(r"(?i)(limit|flag)")
    norm_cols = {_norm(c).lower(): c for c in columns}
    for c in cands:
        hit = norm_cols.get(_norm(c).lower())
        if hit is not None:
            return hit
    el = re.sub(r"(?i)^log", "", _norm(base)).split("/")[0].strip("[]()")
    for col in columns:
        d = str(descriptions.get(col, ""))
        if rx.search(d) and re.search(rf"\b{re.escape(el)}\b", d) and col != base:
            return col
    return None


def resolve_roles(columns: list[str], elements: list[str], *, units: dict | None = None,
                  descriptions: dict | None = None) -> dict:
    """Every role the channel needs, resolved from the table's own column names."""
    units = units or {}
    descriptions = descriptions or {}
    cols = [str(c) for c in columns]

    def first(pats, *, desc_pats=()):
        for c in cols:
            if _is_companion(c):
                continue
            n = _norm(c)
            if any(re.fullmatch(p, n, flags=re.IGNORECASE) for p in pats):
                return c
        for c in cols:
            d = str(descriptions.get(c, ""))
            if any(re.search(p, d, flags=re.IGNORECASE) for p in desc_pats):
                return c
        return None

    roles = {
        "name": first([r"name", r"wd", r"wdname", r"object", r"star", r"id", r"source", r"wdj",
                       r"designation", r"objname"],
                      desc_pats=[r"^(white dwarf |object |star )?name", r"designation"]),
        "ra": first([r"ra", r"raj2000", r"_ra", r"radeg", r"ra_deg"], desc_pats=[r"right ascension"]),
        "dec": first([r"dec", r"dej2000", r"_de", r"dedeg", r"de", r"dec_deg"],
                     desc_pats=[r"declination"]),
        "spt": first([r"spt", r"sptype", r"type", r"spectraltype", r"spectral_type", r"class",
                      r"sp"], desc_pats=[r"spectral (type|class)"]),
        "teff": first([r"teff", r"t_eff", r"temp", r"temperature"], desc_pats=[r"effective temperature"]),
        "logg": first([r"logg", r"log_g", r"log\(g\)", r"gravity"], desc_pats=[r"surface gravity"]),
        "ref": first([r"ref", r"reference", r"bibcode", r"source", r"paper", r"refs", r"r_",
                      r"provenance", r"cite"], desc_pats=[r"reference", r"bibcode"]),
        "atm": first([r"atm", r"atmos", r"atmosphere", r"comp", r"composition", r"dominant",
                      r"hhe", r"h/he", r"atmtype", r"atm_type"],
                     desc_pats=[r"dominant (element|atmospheric)", r"atmosphere (type|composition)"]),
        "hhe": first([r"log\(?h/he\)?", r"h/he", r"logh/he", r"hhe", r"logh", r"h_he"],
                     desc_pats=[r"log\s*\(?\s*H\s*/\s*He"]),
    }
    if roles["atm"] == roles["hhe"] and roles["atm"] is not None:
        roles["atm"] = None
    els = {}
    for el in elements:
        v = element_value_column(el, cols, descriptions)
        if v is None:
            continue
        els[el] = {"value": v, "error": companion_column(v, cols, "error", descriptions),
                   "limit": companion_column(v, cols, "limit", descriptions),
                   "unit": str(units.get(v, "")), "description": str(descriptions.get(v, ""))}
    roles["elements"] = els
    roles["n_elements_resolved"] = len(els)
    # what the abundance columns say the reference is
    refs = set()
    for r in els.values():
        d = (r["description"] + " " + r["value"]).lower()
        if "h(e)" in d or "h/he" in d or "he)" in d:
            refs.add("H(e)")
        elif "/he" in d:
            refs.add("He")
        elif "/h" in d:
            refs.add("H")
    roles["reference_convention"] = sorted(refs) or ["unstated"]
    return roles


# ---------------------------------------------------------------------------
# atmosphere and panels
# ---------------------------------------------------------------------------
def atmosphere_from_spt(spt: str) -> str:
    """H if the first letter after the D is A (DA, DAZ, DAB, DAO); He for DB/DZ/DQ/DC/DO."""
    s = str(spt or "").strip().upper()
    if not s or s in ("NAN", "NONE", "--", ""):
        return ATM_UNKNOWN
    s = s.lstrip("(")
    m = re.match(r"^D\s*([A-Z])", s)
    if not m:
        return ATM_UNKNOWN
    return ATM_H if m.group(1) == "A" else ATM_HE


def atmosphere_of_row(row: pd.Series, roles: dict) -> tuple[str, str]:
    """``(atmosphere, how)`` from an explicit column when present, the spectral type otherwise."""
    if roles.get("atm"):
        v = str(row.get(roles["atm"], "")).strip()
        vl = v.lower()
        if vl in ("h", "hydrogen", "da", "h-rich", "h-dominated", "hydrogen-rich"):
            return ATM_H, "atm_column"
        if vl in ("he", "helium", "db", "dz", "he-rich", "he-dominated", "helium-rich"):
            return ATM_HE, "atm_column"
        if vl.startswith("h") and not vl.startswith("he"):
            return ATM_H, "atm_column"
        if vl.startswith("he"):
            return ATM_HE, "atm_column"
    if roles.get("spt"):
        a = atmosphere_from_spt(row.get(roles["spt"], ""))
        if a != ATM_UNKNOWN:
            return a, "spectral_type"
    if roles.get("hhe"):
        try:
            if np.isfinite(float(row.get(roles["hhe"]))):
                return ATM_HE, "hhe_column_present"
        except (TypeError, ValueError):
            pass
    return ATM_UNKNOWN, "unresolved"


def _to_float(v) -> float:
    try:
        if v is None:
            return np.nan
        s = str(v).strip()
        if s in ("", "--", "nan", "NaN", "None", "..."):
            return np.nan
        s = s.replace("−", "-").lstrip("<>~=")
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _is_limit(v) -> bool:
    s = str(v if v is not None else "").strip()
    return s.startswith("<") or s.lower() in ("ul", "u", "upper", "limit", "true", "1", "y", "yes",
                                               "<=", "≤")


def normalise_name(name: str) -> str:
    s = str(name or "").replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[\s_]+", "", s).upper()
    return s


def build_panels(df: pd.DataFrame, roles: dict, *, default_error_dex: float = 0.2,
                 error_floor_dex: float = 0.02, elements=None) -> tuple[list[Panel], list[dict]]:
    """One panel per table row (one white dwarf, one source) from the resolved roles."""
    panels, diag = [], []
    els = roles.get("elements", {})
    order = [e for e in (elements or list(els)) if e in els]
    for i, row in df.iterrows():
        name = str(row.get(roles["name"], f"row{i}")) if roles.get("name") else f"row{i}"
        ref = str(row.get(roles["ref"], "")) if roles.get("ref") else ""
        atm, how = atmosphere_of_row(row, roles)
        teff = _to_float(row.get(roles["teff"])) if roles.get("teff") else np.nan
        logg = _to_float(row.get(roles["logg"])) if roles.get("logg") else np.nan
        meas_e, meas_v, meas_s, lim_e, lim_v, assumed = [], [], [], [], [], []
        for el in order:
            r = els[el]
            raw = row.get(r["value"])
            v = _to_float(raw)
            if not np.isfinite(v):
                continue
            limit = _is_limit(raw) or (r["limit"] is not None and _is_limit(row.get(r["limit"])))
            if limit:
                lim_e.append(el)
                lim_v.append(v)
                continue
            err = _to_float(row.get(r["error"])) if r["error"] else np.nan
            if not np.isfinite(err) or err <= 0:
                err = float(default_error_dex)
                assumed.append(el)
            meas_e.append(el)
            meas_v.append(v)
            meas_s.append(max(float(err), float(error_floor_dex)))
        p = Panel(name=name, elements=meas_e, values=np.array(meas_v), errors=np.array(meas_s),
                  atmosphere=atm if atm != ATM_UNKNOWN else ATM_HE, teff=teff, logg=logg,
                  reference=ref, limit_elements=lim_e, limit_values=np.array(lim_v),
                  meta={"row": int(i), "atmosphere_how": how, "atmosphere_raw": atm,
                        "errors_assumed_for": assumed, "spt": str(row.get(roles["spt"], ""))
                        if roles.get("spt") else "", "name_key": normalise_name(name)})
        panels.append(p)
        diag.append({"row": int(i), "name": name, "reference": ref, "atmosphere": atm,
                     "atmosphere_how": how, "n_measured": len(meas_e), "n_limits": len(lim_e),
                     "n_errors_assumed": len(assumed)})
    return panels, diag


# ---------------------------------------------------------------------------
# PEWDD probe and fetch
# ---------------------------------------------------------------------------
def probe_pewdd(cfg: dict, *, fetch_fn=None, query_fn=None, log: AcquisitionLog | None = None
                ) -> dict:
    """What VizieR and GitHub hold for PEWDD, with the resolved column roles."""
    log = log or AcquisitionLog()
    src = cfg["sources"]["pewdd"]
    out: dict = {"vizier": {}, "github": {}, "roles": None, "status": STATUS_FAILED}
    # VizieR: which tables exist under the catalogue, and the one we want
    pattern = src["vizier_catalogue"]
    tables = []
    try:
        df = list_tables(pattern, query_fn=query_fn, fetch_fn=fetch_fn)
        tables = df.to_dict("records") if len(df) else []
        log.record("vizier_tables", pattern, rows=len(tables), extra={"route": df.attrs.get("route")})
    except Exception as exc:                                      # noqa: BLE001
        log.record("vizier_tables", pattern, error=repr(exc))
    out["vizier"]["tables"] = [{k: (str(v)[:200] if not isinstance(v, list) else v)
                                for k, v in t.items() if k in ("table_name", "description")}
                               for t in tables]
    table = src["vizier_table"]
    cols, units, descs = [], {}, {}
    try:
        cdf, attempts = asu_table_columns(table, fetch_fn=fetch_fn)
        cols = [str(c) for c in cdf["column_name"].tolist()]
        units = dict(zip(cols, cdf["unit"].astype(str).tolist(), strict=True))
        descs = dict(zip(cols, cdf["description"].astype(str).tolist(), strict=True))
        log.record("vizier_columns", table, rows=len(cols), extra={"n_attempts": len(attempts)})
    except VizierRouteError as exc:
        log.record("vizier_columns", table, error=str(exc), extra={"route_attempts": exc.attempts})
    except Exception as exc:                                      # noqa: BLE001
        log.record("vizier_columns", table, error=repr(exc))
    if not cols:
        # the column census can also come from the catalogue's own listing
        try:
            tdf, _ = asu_catalogue_tables(pattern, fetch_fn=fetch_fn)
            for _, r in tdf.iterrows():
                if str(r["table_name"]).lower().endswith(table.split("/")[-1].lower()):
                    cols = [str(c) for c in (r["columns"] or [])]
                    units = dict(r.get("units") or {})
                    descs = dict(r.get("descriptions") or {})
            log.record("vizier_columns_from_listing", pattern, rows=len(cols))
        except Exception as exc:                                  # noqa: BLE001
            log.record("vizier_columns_from_listing", pattern, error=repr(exc))
    out["vizier"]["table"] = table
    out["vizier"]["columns"] = cols
    out["vizier"]["units"] = units
    out["vizier"]["descriptions"] = descs
    if cols:
        roles = resolve_roles(cols, cfg["elements"]["all"], units=units, descriptions=descs)
        out["roles"] = roles
        out["status"] = STATUS_OK if roles["n_elements_resolved"] >= 3 else STATUS_ZERO
    # GitHub: the repository's files
    entries, branch = github_tree(src["github_repo"], fetch_fn=fetch_fn, log=log)
    out["github"] = {"repo": src["github_repo"], "branch": branch, "n_entries": len(entries),
                     "csv_files": pick_pewdd_csv(entries)[:10]}
    return out


def fetch_pewdd(cfg: dict, out_dir: Path, *, fetch_fn=None, tap_fn=None, log: AcquisitionLog | None = None,
                probe: dict | None = None) -> dict:
    """The PEWDD table over the VizieR ladder, then GitHub; writes ``data/pewdd_<route>.csv``."""
    log = log or AcquisitionLog()
    src = cfg["sources"]["pewdd"]
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    rec: dict = {"route": ROUTE_NONE, "status": STATUS_FAILED, "n_rows": 0, "columns": [],
                 "routes_tried": []}
    table = src["vizier_table"]
    res = vizier_table(table, max_rows=int(src.get("max_rows", 20000)), tap_fn=tap_fn,
                       fetch_fn=fetch_fn, log=log, stage="vizier_rows")
    rec["routes_tried"].append({"route": "vizier", "status": res.status, "served_by": res.route,
                                "endpoint": res.endpoint, "n_attempts": len(res.attempts)})
    df = None
    if res.route != ROUTE_NONE and len(res.rows):
        df = res.rows
        rec.update(route=f"vizier:{res.route}", status=STATUS_OK, n_rows=int(len(df)),
                   endpoint=res.endpoint)
        path = out_dir / "data" / "pewdd_vizier.csv"
        df.to_csv(path, index=False)
        rec["path"] = str(path)
    # GitHub, also fetched when VizieR served (it carries provenance VizieR may not)
    entries, branch = github_tree(src["github_repo"], fetch_fn=fetch_fn, log=log)
    gh = {"branch": branch, "n_entries": len(entries), "files": []}
    for e in pick_pewdd_csv(entries)[:3]:
        url = github_raw_url(src["github_repo"], branch, e["path"])
        text = fetch_text(url, fetch_fn=fetch_fn, retries=2, log=log, stage="github_csv")
        if not text:
            gh["files"].append({"path": e["path"], "status": STATUS_FAILED})
            continue
        try:
            gdf = pd.read_csv(io.StringIO(text), low_memory=False)
        except Exception as exc:                                  # noqa: BLE001
            gh["files"].append({"path": e["path"], "status": STATUS_FAILED, "error": repr(exc)[:300]})
            continue
        path = out_dir / "data" / ("pewdd_github_" + re.sub(r"[^A-Za-z0-9_.-]", "_", e["path"]))
        gdf.to_csv(path, index=False)
        gh["files"].append({"path": e["path"], "status": STATUS_OK, "n_rows": int(len(gdf)),
                            "n_columns": int(gdf.shape[1]), "columns": [str(c) for c in gdf.columns][:120],
                            "local": str(path)})
        if df is None:
            df = gdf
            rec.update(route="github_csv", status=STATUS_OK, n_rows=int(len(df)),
                       endpoint=url, path=str(path))
            break
    rec["github"] = gh
    if df is not None:
        rec["columns"] = [str(c) for c in df.columns]
        rec["n_rows"] = int(len(df))
    return rec


def discover_timescale_tables(cfg: dict, out_dir: Path, *, fetch_fn=None,
                              log: AcquisitionLog | None = None) -> dict:
    """PyllutedWD's (or Koester's) diffusion-timescale files, fetched and parsed if they exist."""
    log = log or AcquisitionLog()
    src = cfg["sources"]["pyllutedwd"]
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    out: dict = {"repo": src["github_repo"], "files": [], "parsed": {}, "koester_pages": []}
    entries, branch = github_tree(src["github_repo"], fetch_fn=fetch_fn, log=log)
    out["branch"] = branch
    out["n_entries"] = len(entries)
    for e in pick_timescale_files(entries)[: int(src.get("max_files", 12))]:
        url = github_raw_url(src["github_repo"], branch, e["path"])
        text = fetch_text(url, fetch_fn=fetch_fn, retries=2, log=log, stage="pyllutedwd_file")
        rec = {"path": e["path"], "size": e.get("size"), "url": url}
        if not text:
            rec["status"] = STATUS_FAILED
            out["files"].append(rec)
            continue
        rec["status"] = STATUS_OK
        rec["head"] = text[:600]
        tab = parse_timescale_table(text)
        if tab is not None:
            rec["parsed_columns"] = [str(c) for c in tab.columns]
            rec["parsed_rows"] = int(len(tab))
            local = out_dir / "data" / ("timescales_" + re.sub(r"[^A-Za-z0-9_.-]", "_", e["path"]))
            tab.to_csv(local, index=False)
            rec["local"] = str(local)
            atm = ATM_H if re.search(r"(?i)(^|[^a-z])(da|h[-_]?rich|hydrogen)([^a-z]|$)", e["path"]) \
                else ATM_HE if re.search(r"(?i)(^|[^a-z])(db|dz|he[-_]?rich|helium)([^a-z]|$)", e["path"]) \
                else ATM_UNKNOWN
            rec["atmosphere_guess"] = atm
            out["parsed"].setdefault(atm, []).append(rec["local"])
        out["files"].append(rec)
    for url in src.get("koester_urls", []):
        text = fetch_text(url, fetch_fn=fetch_fn, retries=1, log=log, stage="koester_page")
        out["koester_pages"].append({"url": url, "status": STATUS_OK if text else STATUS_FAILED,
                                     "bytes": len(text) if text else 0,
                                     "links": re.findall(r'href="([^"]+)"', text)[:60] if text else []})
    return out


__all__ = ["ATM_H", "ATM_HE", "ATM_UNKNOWN", "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO",
           "VIZIER_ASU", "AcquisitionLog", "atmosphere_from_spt", "atmosphere_of_row",
           "build_panels", "companion_column", "discover_timescale_tables",
           "element_value_column", "fetch_pewdd", "fetch_text", "github_raw_url",
           "github_tree", "http_text", "normalise_name", "pick_pewdd_csv",
           "pick_timescale_files", "probe_pewdd", "reset_route_state", "resolve_roles"]
