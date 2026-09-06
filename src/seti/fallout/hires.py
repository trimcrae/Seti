"""FALLOUT high-resolution tier: the elements that DECIDE the fission vector.

The GALAH tier (v2, 2026-09-06) is honest about what it can and cannot do:
the dwarf sample is 20% testable with 1.5% completeness even at Nd +1.0 dex,
and no GALAH element separates fission from an s+r mixture *decisively*. The
elements that do are not in GALAH:

* **Pb** -- the s-process makes it (the strong component ends at Pb-208);
  fission never does (no fragment reaches A = 208). An s-process star with
  [Nd/H] up has Pb up; a fission-polluted star has Pb at the *scaled-solar*
  level or below. A Pb **upper limit** below the s-prediction is therefore
  evidence *for* fission-not-s, and is treated as a censored value, not a
  missing one.
* **Ag, Pd (and Cd, Sn)** -- the fission valley at A ~ 108-125, three
  decades below the peaks. The r-process is not suppressed there (solar Ag is
  80% r), so an r-process enrichment that reproduces the heavy peak brings
  Ag/Pd up with it; fission does not.
* **Eu** -- fission makes almost none; the r-process makes most of it.

The public compilations that carry those elements are literature compilations
of high-resolution work: **JINAbase** (Abohalima & Frebel 2018, ApJS 238, 36;
VizieR ``J/ApJS/238/36``: ~1,900 metal-poor stars, up to ~20 n-capture
elements, upper limits flagged) and the **Hypatia Catalog** (Hinkel et al.
2014, AJ 148, 54; VizieR ``J/AJ/148/54``, newer releases preferred where
VizieR has them: FGK stars, up to ~80 elements in a subset). Both are
discovered at runtime -- tables may be split into element groups and joined
on the star name -- and every column that was looked for is reported as
found-or-not.

Two things are specific to compilations and are handled as first-class
vetoes: the abundances mix analyses, so each element's error is floored at
the measured star-to-star scatter of **duplicate entries** where the
compilation has them (else at the peer scatter, as in the GALAH tier), and a
star whose own duplicates disagree by more than ``hetero_max_dex`` in a
pattern element is ``literature_heterogeneity``-vetoed. And metal-poor stars
are r-process dominated with a known weak-r spread at Sr-Ag, so the natural
alternative always includes the r template with a free amplitude: the valley
elements are compared against an r-process prediction, never against solar.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..tailings import acquire as A
from ..tailings import manifold as M
from . import pattern as P
from . import yields as Y

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_DEGRADED = "DEGRADED_SOURCE"
VERDICT_NO_PATTERN = "NO_FISSION_PATTERN"
VERDICT_CANDIDATES = "FISSION_PATTERN_CANDIDATES_PENDING_VET"

HIRES_STAGES: tuple[str, ...] = ("hires-probe", "hires-acquire", "hires-screen", "hires-assess", "hires-all")

#: Default source registry; ``config/fallout.yaml`` ``hires.sources`` overrides.
DEFAULT_SOURCES: dict[str, dict] = {
    "JINABASE": {
        "keywords": ["J/ApJS/238/36", "JINAbase"],
        "prefer": ["J/ApJS/238/36"],
        "note": "Abohalima & Frebel 2018, ApJS 238, 36 -- metal-poor literature compilation",
    },
    "HYPATIA": {
        "keywords": ["Hypatia"],
        "prefer": ["J/AJ/148/54"],
        "note": "Hinkel et al. 2014, AJ 148, 54 -- FGK literature compilation; newer VizieR releases preferred",
    },
}

_ELEMENT_CASE = {e.lower(): e for e in Y.SOLAR_LOGEPS if e not in ("Li", "Fe")}
_ELEMENT_CASE.update({"fe": "Fe"})

#: Identifier columns a split catalogue can be joined on, in preference order.
_ID_CANDIDATES = ("name", "star", "starname", "object", "id", "simbad", "simbadname", "jina_id",
                  "hip", "hd", "gaia", "source_id", "recno")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canon(name: str) -> str:
    """Lowercase; fold VizieR's ``[X/H]`` / ``logeps(X)`` / ``__X_H_`` forms onto one shape."""
    s = str(name).strip().lower()
    s = re.sub(r"[\[\]/() \-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _params(columns) -> dict[str, str]:
    """Stellar-parameter columns via TAILINGS' resolver, tolerant of ``[Fe/H]`` literals."""
    cols = [str(c) for c in columns]
    back = {_canon(c): c for c in cols}
    found = A.resolve_param_columns(list(back))
    return {k: back.get(v, v) for k, v in found.items()}


# ---------------------------------------------------------------------------
# Column resolution: [X/H], [X/Fe] or log eps, with VizieR limit flags
# ---------------------------------------------------------------------------
_VALUE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^([a-z]{1,2})_h$", "xh"),
    (r"^([a-z]{1,2})_fe$", "xfe"),
    (r"^(?:log_?eps|loge|eps|a)_?([a-z]{1,2})$", "logeps"),
    (r"^([a-z]{1,2})2?_?(?:i|ii)_h$", "xh"),
    (r"^([a-z]{1,2})2?_?(?:i|ii)_fe$", "xfe"),
)


def resolve_hires_columns(columns) -> dict[str, dict[str, str]]:
    """Map a compilation's columns onto ``{El: {value, kind, err, limit}}``.

    Understands ``[X/H]`` (VizieR ``__X_H_``), ``[X/Fe]``, ``logeps(X)`` /
    ``A(X)`` value columns; VizieR ``e_`` errors and ``l_`` limit flags
    (``"<"``); and ``f_`` flag columns as a fallback limit flag. A bare
    element-symbol column is accepted only when it has an ``e_`` or ``l_``
    companion, so ``Y`` the element and ``Y`` a coordinate cannot be confused.
    """
    cols = [str(c) for c in columns]
    canon = {_canon(c): c for c in cols}
    out: dict[str, dict[str, str]] = {}

    def put(el: str, kind: str, key: str, orig: str):
        d = out.setdefault(el, {})
        if key == "value":
            if "value" not in d:
                d["value"] = orig
                d["kind"] = kind
        else:
            d.setdefault(key, orig)

    for c in cols:
        cc = _canon(c)
        prefix, rest = None, cc
        for pfx, kind in (("e_", "err"), ("l_", "limit"), ("f_", "flag"), ("n_", "note"),
                          ("r_", "ref"), ("q_", "quality")):
            if cc.startswith(pfx):
                prefix, rest = kind, cc[len(pfx):]
                break
        m_el, m_kind = None, None
        for pat, kind in _VALUE_PATTERNS:
            m = re.match(pat, rest)
            if m and m.group(1) in _ELEMENT_CASE:
                m_el, m_kind = _ELEMENT_CASE[m.group(1)], kind
                break
        if m_el is None and rest in _ELEMENT_CASE and (f"e_{rest}" in canon or f"l_{rest}" in canon):
            m_el, m_kind = _ELEMENT_CASE[rest], "xfe_bare"
        if m_el is None or m_el == "Fe":
            continue
        if prefix is None:
            put(m_el, m_kind, "value", c)
        elif prefix in ("err", "limit", "flag"):
            put(m_el, m_kind, prefix, c)
    return {el: d for el, d in out.items() if "value" in d}


def _limit_code(v) -> int:
    s = str(v).strip().lower() if v is not None else ""
    if s in ("", "nan", "none", "0", "0.0", "false"):
        return P.DETECTION
    if "<" in s or s in ("u", "ul", "upper", "lim", "l"):
        return P.UPPER_LIMIT
    if ">" in s or s in ("ll", "lower", "g"):
        return P.LOWER_LIMIT
    return P.DETECTION


def normalise_hires(df: pd.DataFrame, *, source: str) -> tuple[pd.DataFrame, dict]:
    """Raw compilation rows -> canonical ``star_id, teff, logg, fe_h, <El> ([X/Fe]), e_, lim_``.

    Values in ``[X/H]`` or ``log eps`` are converted to ``[X/Fe]`` with
    Asplund 2021 solar values, so the same peer regression, vetoes and vetting
    code as the GALAH tier apply. Every conversion is recorded per element.
    """
    params = _params(df.columns)
    res = resolve_hires_columns(df.columns)
    out = pd.DataFrame(index=df.index)
    log: dict = {"params": params, "elements": {}, "n_rows": int(len(df))}
    for canon, orig in params.items():
        out[canon] = df[orig]
    if "star_id" not in out.columns:
        for cand in _ID_CANDIDATES:
            hit = next((c for c in df.columns if _canon(c) == cand), None)
            if hit is not None:
                out["star_id"] = df[hit]
                break
    if "star_id" not in out.columns:
        out["star_id"] = [f"{source}_{i}" for i in range(len(df))]
    out["star_id"] = out["star_id"].astype(str).str.strip()
    for c in ("teff", "logg", "fe_h"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    feh = pd.to_numeric(out["fe_h"], errors="coerce").to_numpy(dtype=float) if "fe_h" in out.columns \
        else np.full(len(df), np.nan)
    for el, d in res.items():
        v = pd.to_numeric(df[d["value"]], errors="coerce").to_numpy(dtype=float)
        kind = d.get("kind", "xfe")
        if kind == "xh":
            xfe = v - feh
        elif kind == "logeps":
            xfe = v - Y.SOLAR_LOGEPS.get(el, np.nan) - feh
        else:
            xfe = v
        out[el] = xfe
        if "err" in d:
            out[f"e_{el}"] = pd.to_numeric(df[d["err"]], errors="coerce")
        lim = np.zeros(len(df), dtype=int)
        lcol = d.get("limit") or d.get("flag")
        if lcol is not None:
            lim = np.array([_limit_code(x) for x in df[lcol].tolist()], dtype=int)
        out[f"lim_{el}"] = lim
        log["elements"][el] = {"value": d["value"], "kind": kind, "err": d.get("err"),
                               "limit": lcol,
                               "n_values": int(np.isfinite(xfe).sum()),
                               "n_upper_limits": int((lim == P.UPPER_LIMIT).sum())}
    out["source"] = source
    return out, log


# ---------------------------------------------------------------------------
# Duplicates: collapse and measure heterogeneity
# ---------------------------------------------------------------------------
def collapse_duplicates(df: pd.DataFrame, elements: list[str], *, id_col: str = "star_id",
                        ) -> tuple[pd.DataFrame, dict[str, float], dict]:
    """One row per star; the spread among a star's duplicate entries is kept.

    For each element: detections take precedence over limits; the value is
    the median of the detections (or the *smallest* upper limit if there are
    only limits); the within-star range goes to ``hetero_<El>`` and its
    maximum over the pattern elements to ``hetero_max_dex``. The per-element
    robust scatter of (entry - star median) over multi-entry stars is the
    compilation's measured heterogeneity, returned as the second value and
    used as an error floor.
    """
    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = [f"row{i}" for i in range(len(df))]
    groups = df.groupby(id_col, sort=False)
    rows = []
    devs: dict[str, list[float]] = {el: [] for el in elements}
    for sid, g in groups:
        rec = {id_col: sid, "n_entries": int(len(g))}
        for c in ("teff", "logg", "fe_h"):
            if c in g.columns:
                rec[c] = float(pd.to_numeric(g[c], errors="coerce").median())
        for c in g.columns:
            if c in (id_col, "teff", "logg", "fe_h") or c in elements or c.startswith(("e_", "lim_", "hetero_")):
                continue
            rec[c] = g[c].iloc[0]
        het_max = np.nan
        for el in elements:
            if el not in g.columns:
                continue
            v = pd.to_numeric(g[el], errors="coerce").to_numpy(dtype=float)
            lim = pd.to_numeric(g[f"lim_{el}"], errors="coerce").fillna(0).to_numpy(dtype=int) \
                if f"lim_{el}" in g.columns else np.zeros(len(g), dtype=int)
            e = pd.to_numeric(g[f"e_{el}"], errors="coerce").to_numpy(dtype=float) \
                if f"e_{el}" in g.columns else np.full(len(g), np.nan)
            det = np.isfinite(v) & (lim == 0)
            if det.any():
                vals = v[det]
                rec[el] = float(np.median(vals))
                rec[f"lim_{el}"] = 0
                rec[f"e_{el}"] = float(np.nanmedian(e[det])) if np.isfinite(e[det]).any() else np.nan
                if det.sum() > 1:
                    rng_ = float(vals.max() - vals.min())
                    rec[f"hetero_{el}"] = rng_
                    het_max = rng_ if not np.isfinite(het_max) else max(het_max, rng_)
                    devs[el].extend((vals - np.median(vals)).tolist())
            elif (np.isfinite(v) & (lim == P.UPPER_LIMIT)).any():
                ul = v[np.isfinite(v) & (lim == P.UPPER_LIMIT)]
                rec[el] = float(ul.min())
                rec[f"lim_{el}"] = P.UPPER_LIMIT
                rec[f"e_{el}"] = float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan
            elif (np.isfinite(v) & (lim == P.LOWER_LIMIT)).any():
                ll = v[np.isfinite(v) & (lim == P.LOWER_LIMIT)]
                rec[el] = float(ll.max())
                rec[f"lim_{el}"] = P.LOWER_LIMIT
                rec[f"e_{el}"] = float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan
            else:
                rec[el] = np.nan
                rec[f"lim_{el}"] = 0
                rec[f"e_{el}"] = np.nan
        rec["hetero_max_dex"] = het_max
        rows.append(rec)
    out = pd.DataFrame(rows)
    scatter = {}
    for el, d in devs.items():
        arr = np.asarray(d, dtype=float)
        arr = arr[np.isfinite(arr)]
        scatter[el] = float(M.robust_sigma(arr)) if arr.size >= 6 else float("nan")
    notes = {"n_rows_in": int(len(df)), "n_stars": int(len(out)),
             "n_multi_entry": int((out["n_entries"] > 1).sum()) if len(out) else 0,
             "elements_with_duplicate_scatter": sorted(k for k, v in scatter.items() if np.isfinite(v))}
    return out, scatter, notes


# ---------------------------------------------------------------------------
# Discovery and acquisition (VizieR TAP through seti.tailings.acquire)
# ---------------------------------------------------------------------------
@dataclass
class SourcePull:
    source: str
    table: pd.DataFrame
    verdict: str
    tables_used: list[str] = field(default_factory=list)
    discovery: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    columns_found: dict = field(default_factory=dict)
    degraded: bool = False
    degradation: str = ""

    @property
    def n_rows(self) -> int:
        return int(len(self.table))

    def provenance(self) -> dict:
        return {"source": self.source, "verdict": self.verdict, "n_rows": self.n_rows,
                "tables_used": self.tables_used, "discovery": self.discovery,
                "columns_found": self.columns_found, "degraded": self.degraded,
                "degradation": self.degradation, "log": list(self.log)}


def discover_source(name: str, spec: dict, *, query_fn=None) -> dict:
    """Which VizieR tables belong to this compilation, and what each carries.

    Keyword discovery over ``TAP_SCHEMA`` (table name and description), the
    preferred catalogue ids added explicitly, then the full column list of
    every candidate from ``TAP_SCHEMA.columns``. Each table is scored by how
    many n-capture elements it resolves and whether it has stellar parameters,
    so a split catalogue (parameters in one table, elements in others) is
    recognised as a set to be joined rather than as several bad tables.
    """
    query_fn = query_fn or (lambda q: A.tap_query(q))
    found = A.discover_tables(tuple(spec.get("keywords") or [name]), query_fn=query_fn)
    for pref in spec.get("prefer") or []:
        found += A.discover_tables((pref,), query_fn=query_fn)
    found = list(dict.fromkeys(found))
    cols = A.fetch_table_columns(found, query_fn=query_fn) if found else {}
    tables = {}
    for t, cl in cols.items():
        res = resolve_hires_columns(cl)
        params = _params(cl)
        ids = [c for c in cl if _canon(c) in _ID_CANDIDATES]
        tables[t] = {"n_columns": len(cl), "elements": sorted(res), "n_elements": len(res),
                     "params": params, "id_columns": ids,
                     "has_params": all(k in params for k in ("teff", "logg", "fe_h"))}
    # group by catalogue prefix (everything before the last '/'), score groups
    groups: dict[str, list[str]] = {}
    for t in tables:
        groups.setdefault(t.rsplit("/", 1)[0] if "/" in t else t, []).append(t)
    scored = []
    for pfx, ts in groups.items():
        n_el = len(set().union(*[set(tables[t]["elements"]) for t in ts]))
        has_p = any(tables[t]["has_params"] for t in ts)
        preferred = any(pfx.startswith(p) for p in (spec.get("prefer") or []))
        scored.append({"prefix": pfx, "tables": ts, "n_elements_union": n_el, "has_params": has_p,
                       "preferred": preferred,
                       "score": n_el * 10 + (20 if has_p else 0) + (5 if preferred else 0)})
    scored.sort(key=lambda r: -r["score"])
    return {"source": name, "keywords": spec.get("keywords"), "n_tables_found": len(found),
            "tables": tables, "groups": scored}


def _merge_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str | None]:
    """Join split tables on a shared identifier column (outer join)."""
    names = list(frames)
    if len(names) == 1:
        return frames[names[0]], None
    common = None
    for cand in _ID_CANDIDATES:
        hits = []
        for t in names:
            hit = next((c for c in frames[t].columns if _canon(c) == cand), None)
            hits.append(hit)
        if all(h is not None for h in hits):
            common = hits
            break
    if common is None:
        # no shared id: keep the richest table alone
        best = max(names, key=lambda t: frames[t].shape[1])
        return frames[best], None
    merged = None
    for t, idc in zip(names, common, strict=True):
        f = frames[t].copy()
        f["_join_id"] = f[idc].astype(str).str.strip()
        f = f.drop(columns=[c for c in f.columns if c != "_join_id" and merged is not None
                            and c in merged.columns])
        merged = f if merged is None else merged.merge(f, on="_join_id", how="outer")
    merged = merged.rename(columns={"_join_id": "Name"}) if "Name" not in merged.columns \
        else merged.drop(columns=["_join_id"])
    return merged, common[0]


def fetch_source(name: str, spec: dict, *, max_rows: int = 20000, query_fn=None) -> SourcePull:
    """Discover, pull and normalise one compilation; every outcome is a verdict."""
    query_fn = query_fn or (lambda q: A.tap_query(q))
    log: list[str] = []
    try:
        disc = discover_source(name, spec, query_fn=query_fn)
    except Exception as exc:  # noqa: BLE001
        return SourcePull(name, pd.DataFrame(), VERDICT_NO_DATA, degraded=True,
                          degradation=f"{VERDICT_NO_DATA}: discovery raised {exc!r}",
                          log=[f"{name}: discovery raised {exc!r}"])
    if not disc["groups"]:
        return SourcePull(name, pd.DataFrame(), VERDICT_NO_DATA, discovery=disc, degraded=True,
                          degradation=(f"{VERDICT_NO_DATA}: no VizieR table answered for "
                                       f"{spec.get('keywords')} ({disc['n_tables_found']} names found, "
                                       "none with resolvable columns)"),
                          log=[f"{name}: no usable table"])
    grp = disc["groups"][0]
    log.append(f"{name}: using catalogue group {grp['prefix']} ({grp['n_elements_union']} elements, "
               f"params={grp['has_params']}, preferred={grp['preferred']})")
    frames: dict[str, pd.DataFrame] = {}
    errors = []
    for t in grp["tables"]:
        info = disc["tables"][t]
        if info["n_elements"] == 0 and not info["has_params"]:
            continue
        try:
            df = query_fn(f'SELECT TOP {int(max_rows)} * FROM "{A.unquote_table(t)}"')
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{t}: {exc!r}")
            continue
        if df is None or len(df) == 0:
            errors.append(f"{t}: empty")
            continue
        frames[t] = df
        log.append(f"{name}: {t} -> {len(df)} rows, {info['n_elements']} elements")
    if not frames:
        if errors:
            return SourcePull(name, pd.DataFrame(), "QUERY_FAILED", discovery=disc, degraded=True,
                              degradation=f"QUERY_FAILED: every table in {grp['prefix']} errored or was "
                                          f"empty ({'; '.join(errors)})", log=log)
        return SourcePull(name, pd.DataFrame(), "QUERY_RETURNED_ZERO_ROWS", discovery=disc, degraded=True,
                          degradation=(f"QUERY_RETURNED_ZERO_ROWS: no table in {grp['prefix']} carried "
                                       "an n-capture element column or stellar parameters, so nothing "
                                       "was pulled"), log=log)
    raw, join_col = _merge_tables(frames)
    if join_col:
        log.append(f"{name}: joined {len(frames)} tables on {join_col} -> {len(raw)} rows")
    norm, nlog = normalise_hires(raw, source=name)
    n_el = len(nlog["elements"])
    notes = []
    if n_el == 0:
        return SourcePull(name, pd.DataFrame(), "QUERY_RETURNED_ZERO_ROWS", discovery=disc,
                          tables_used=list(frames), degraded=True,
                          degradation="the tables answered but no n-capture element column resolved",
                          log=log, columns_found=nlog)
    missing_p = [k for k in ("teff", "logg", "fe_h") if k not in norm.columns]
    if missing_p:
        notes.append(f"no {', '.join(missing_p)} column: the peer regression falls back to medians")
    if errors:
        notes.append("some tables failed: " + "; ".join(errors))
    if not grp["preferred"]:
        notes.append(f"the preferred catalogue id {spec.get('prefer')} was not what answered; "
                     f"used {grp['prefix']}")
    if len(norm) >= max_rows:
        notes.append(f"hit the {max_rows}-row cap: TRUNCATED")
    log.append(f"{name}: {len(norm)} rows normalised, {n_el} elements: "
               + ", ".join(f"{el}({d['n_values']}{'/' + str(d['n_upper_limits']) + 'ul' if d['n_upper_limits'] else ''})"
                           for el, d in sorted(nlog["elements"].items())))
    return SourcePull(name, norm, "OK", tables_used=list(frames), discovery=disc, log=log,
                      columns_found=nlog, degraded=bool(notes), degradation="; ".join(notes))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _hcfg(block: dict) -> dict:
    return dict(block.get("hires") or {})


def _pattern_config(block: dict) -> P.PatternConfig:
    from .run import _pattern_config as base_cfg

    base = dict(block.get("pattern") or {})
    base.update(_hcfg(block).get("pattern") or {})
    return base_cfg({"pattern": base})


def _sources(block: dict, requested: list[str] | None) -> dict[str, dict]:
    reg = dict(DEFAULT_SOURCES)
    for k, v in (_hcfg(block).get("sources") or {}).items():
        reg[k.upper()] = {**reg.get(k.upper(), {}), **(v or {})}
    if requested:
        return {k: reg[k] for k in requested if k in reg}
    return reg


def stage_hires_probe(block: dict, out_dir: Path, sources: list[str] | None, *, query_fn=None) -> dict:
    rep = {"generated_utc": _now_utc(), "sources": {}}
    for name, spec in _sources(block, sources).items():
        try:
            rep["sources"][name] = discover_source(name, spec, query_fn=query_fn)
        except Exception as exc:  # noqa: BLE001
            rep["sources"][name] = {"error": repr(exc)}
        d = rep["sources"][name]
        print(f"[fallout/hires] probe {name}: {d.get('n_tables_found', 0)} tables; groups: "
              + "; ".join(f"{g['prefix']} n_el={g['n_elements_union']} params={g['has_params']}"
                          for g in d.get("groups", [])[:4]))
    _json(out_dir / "hires_probe.json", rep)
    return rep


def stage_hires_acquire(block: dict, out_dir: Path, sources: list[str] | None, *, max_rows: int | None,
                        query_fn=None) -> dict:
    from ..tailings.acquire import write_checkpoint

    hb = _hcfg(block)
    cap = int(max_rows or (hb.get("pull") or {}).get("max_rows", 20000))
    rep = {"generated_utc": _now_utc(), "sources": []}
    for name, spec in _sources(block, sources).items():
        pull = fetch_source(name, spec, max_rows=cap, query_fn=query_fn)
        for line in pull.log:
            print(f"[fallout/hires] {line}")
        rep["sources"].append(pull.provenance())
        if pull.n_rows:
            write_checkpoint(pull.table, out_dir / f"stars_hires_{name.lower()}.parquet")
    _json(out_dir / "hires_acquisition.json", rep)
    return rep


def screen_source(stars: pd.DataFrame, *, source: str, block: dict, out_dir: Path) -> dict:
    """Peer residuals, duplicate collapse, error floors, censored fits, LOO, vetoes."""
    from ..tailings.acquire import write_checkpoint

    cfg = _pattern_config(block)
    hb = _hcfg(block)
    peer = {**(block.get("peer") or {}), **(hb.get("peer") or {})}
    min_rows = int(peer.get("min_rows", 40))
    loo_prefilter = float((block.get("report") or {}).get("loo_prefilter_lr", 4.0))

    elements = [e for e in cfg.elements if e in stars.columns
                and pd.to_numeric(stars[e], errors="coerce").notna().sum() >= int(hb.get("min_values_per_element", 5))]
    if len(elements) < cfg.min_elements or len(stars) < min_rows:
        return {"source": source, "sample": f"hires_{source.lower()}", "n_stars": int(len(stars)),
                "n_elements": len(elements), "elements": elements,
                "verdict": f"INSUFFICIENT: {len(elements)} usable pattern elements or {len(stars)} stars"}
    T = P.build_templates(elements, horizon_yr=cfg.horizon_yr)
    elements = list(T.elements)

    stars, dup_scatter, dup_notes = collapse_duplicates(stars, elements)
    for c in ("teff", "logg", "fe_h"):
        if c not in stars.columns:
            stars[c] = np.nan
    limits_by_el = {el: (pd.to_numeric(stars.get(f"lim_{el}"), errors="coerce").fillna(0).to_numpy(dtype=int) != 0)
                    if f"lim_{el}" in stars.columns else np.zeros(len(stars), dtype=bool) for el in elements}
    resid, scatter, notes = P.peer_residuals(
        stars, elements, degree=int(peer.get("degree", 2)), clip=float(peer.get("clip_sigma", 4.0)),
        n_iter=int(peer.get("clip_iterations", 4)), min_rows=min_rows, exclude_from_fit=limits_by_el)
    floors = P.error_floors(stars, elements, scatter, cfg=cfg, duplicate_scatter=dup_scatter)
    floor_map = {el: d["floor_dex"] for el, d in floors.items()}

    frame = stars.copy()
    for el in elements:
        frame[f"peer_{el}"] = resid[el].to_numpy()
    obs, sig, _, info = P.assemble_vectors(frame, elements, value_prefix="peer_", cfg=cfg,
                                           fallback_sigma=scatter, sigma_floor=floor_map,
                                           limit_prefix="lim_")
    limits = info["limits"]
    raw = P.raw_vectors(stars, elements)
    for el in elements:
        frame[f"raw_{el}"] = raw[el].to_numpy()
    obs_raw, sig_raw, _, _ = P.assemble_vectors(frame, elements, value_prefix="raw_", cfg=cfg,
                                                fallback_sigma=scatter, sigma_floor=floor_map,
                                                limit_prefix="lim_")

    fit = P.fit_patterns(obs, sig, T, cfg, limits)
    fit["fission_lr_raw"] = P.fission_lr_only(obs_raw, sig_raw, T, cfg, limits)
    fit["lr_noba"] = P.lr_without(obs, sig, T, cfg, drop=(cfg.ba_element,), limits=limits)
    testable = P.testable_mask(obs, T, cfg)
    fit["testable"] = testable
    pre = fit["fission_lr"].to_numpy() >= loo_prefilter
    loo_cols = [f"lr_without_{el}" for el in elements] + ["lr_loo_min", "lr_loo_driver"]
    for c in loo_cols:
        fit[c] = "" if c == "lr_loo_driver" else np.nan
    if pre.any():
        loo = P.leave_one_out(obs[pre], sig[pre], T, cfg, limits[pre])
        for c in loo_cols:
            fit.loc[pre, c] = loo[c].to_numpy()
    fit["lr_loo_driver"] = fit["lr_loo_driver"].astype(str)

    keep = [c for c in ("star_id", "teff", "logg", "fe_h", "n_entries", "hetero_max_dex", "source")
            if c in stars.columns]
    scores = pd.concat([stars[keep].reset_index(drop=True),
                        stars[elements].reset_index(drop=True),
                        fit.reset_index(drop=True)], axis=1)
    scores["sample"] = f"hires_{source.lower()}"
    for k, el in enumerate(elements):
        scores[f"peer_{el}"] = resid[el].to_numpy()
        scores[f"sig_{el}"] = sig[:, k]
        scores[f"lim_{el}"] = limits[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[f"z_{el}"] = np.where(np.isfinite(obs[:, k]), obs[:, k] / sig[:, k], np.nan)
    # the decisive ratios, per star, from the peer residuals with the limit sense
    dec = [P.decisive_ratios({el: float(obs[i, k]) for k, el in enumerate(elements)},
                             {el: int(limits[i, k]) for k, el in enumerate(elements)})
           for i in range(len(scores))]
    for key in dec[0] if dec else []:
        scores[f"dec_{key}"] = [d[key] for d in dec]

    tag = f"hires_{source.lower()}"
    write_checkpoint(scores, out_dir / f"scores_{tag}.parquet")
    vec = pd.DataFrame({**{f"v_{el}": obs[:, k] for k, el in enumerate(elements)},
                        **{f"s_{el}": sig[:, k] for k, el in enumerate(elements)},
                        **{f"l_{el}": limits[:, k] for k, el in enumerate(elements)},
                        "testable": testable})
    write_checkpoint(vec, out_dir / f"vectors_{tag}.parquet")
    counts = fit["classification"].value_counts().to_dict()
    n_lim = {el: int((limits[:, k] == P.UPPER_LIMIT).sum()) for k, el in enumerate(elements)}
    return {
        "source": source, "sample": tag, "n_stars": int(len(stars)), "n_elements": len(elements),
        "elements": elements, "class_counts": {k: int(v) for k, v in counts.items()},
        "n_lr_prefilter": int(pre.sum()), "n_above_lr_min": int((fit["fission_lr"] >= cfg.lr_min).sum()),
        "n_testable": int(testable.sum()), "testable_fraction": float(testable.mean()),
        "n_unexplained": int((fit["classification"] == P.UNEXPLAINED).sum()),
        "per_element_counts": {el: int(np.isfinite(obs[:, k]).sum()) for k, el in enumerate(elements)},
        "per_element_upper_limits": n_lim,
        "decisive_elements_present": [e for e in ("Pb", "Ag", "Pd", "Cd", "Sn", "Eu", "Th") if e in elements],
        "duplicates": dup_notes, "duplicate_scatter_dex": {k: (round(v, 4) if np.isfinite(v) else None)
                                                             for k, v in dup_scatter.items()},
        "peer_scatter_dex": {k: round(float(v), 4) for k, v in scatter.items()},
        "error_model": {"mode": cfg.error_floor_mode, "systematic_floor_dex": cfg.systematic_floor_dex,
                        "per_element": floors},
        "peer_notes": notes, "templates": T.to_dict(1.0), "verdict": None,
    }


def stage_hires_screen(block: dict, out_dir: Path, sources: list[str] | None) -> list[dict]:
    out = []
    for name in _sources(block, sources):
        p = out_dir / f"stars_hires_{name.lower()}.parquet"
        if not p.exists():
            out.append({"source": name, "sample": None, "n_stars": 0,
                        "verdict": f"{VERDICT_NO_DATA}: no checkpoint for {name}"})
            continue
        stars = pd.read_parquet(p)
        print(f"[fallout/hires] {name}: {len(stars)} rows to screen")
        out.append(screen_source(stars, source=name, block=block, out_dir=out_dir))
    _json(out_dir / "hires_screen.json", {"generated_utc": _now_utc(), "samples": out})
    return out


_SHOW = ["star_id", "teff", "logg", "fe_h", "n_entries", "n_measured", "n_limits", "fission_lr",
         "enrich_lr", "reduced_chi2_best", "lr_noba", "lr_loo_min", "lr_loo_driver", "n_heavy_coherent",
         "fission_lr_raw", "a_f", "classification", "natural_class", "first_veto", "veto_reasons",
         "dec_Pb/Nd", "dec_Pb/Nd_limit", "dec_Ag/Nd", "dec_Ag/Nd_limit", "dec_Pd/Nd", "dec_Pd/Nd_limit",
         "dec_Eu/Nd", "dec_Eu/Nd_limit"]


def _records(df: pd.DataFrame, elements: list[str], n: int) -> list[dict]:
    if df is None or not len(df):
        return []
    cols = [c for c in _SHOW + list(elements) + [f"peer_{e}" for e in elements] + [f"lim_{e}" for e in elements]
            if c in df.columns]
    return df[cols].head(n).to_dict(orient="records")


def assess_source(rec: dict, *, block: dict, out_dir: Path) -> dict:
    cfg = _pattern_config(block)
    hb = _hcfg(block)
    nb = {**(block.get("null") or {}), **(hb.get("null") or {})}
    sb = {**(block.get("sensitivity") or {}), **(hb.get("sensitivity") or {})}
    tag = rec["sample"]
    sp = out_dir / f"scores_{tag}.parquet"
    vp = out_dir / f"vectors_{tag}.parquet"
    if not sp.exists() or not vp.exists():
        return {**rec, "verdict": rec.get("verdict") or "INSUFFICIENT: no screen checkpoint"}
    scores = pd.read_parquet(sp)
    vec = pd.read_parquet(vp)
    elements = list(rec["elements"])
    T = P.build_templates(elements, horizon_yr=cfg.horizon_yr)
    obs = np.column_stack([vec[f"v_{el}"].to_numpy(dtype=float) for el in T.elements])
    sig = np.column_stack([vec[f"s_{el}"].to_numpy(dtype=float) for el in T.elements])
    lim = np.column_stack([vec[f"l_{el}"].to_numpy(dtype=int) for el in T.elements])
    testable = vec["testable"].to_numpy(dtype=bool)

    rng = np.random.default_rng(int(nb.get("seed", 20260906)))
    shuffled = P.shuffled_null(obs, sig, T, cfg, n_perm=int(nb.get("n_perm", 5)),
                               max_rows=int(nb.get("max_rows", 20000)), rng=rng, limits=lim)
    thr, why = P.derive_threshold(cfg, shuffled)
    lr = scores["fission_lr"].to_numpy(dtype=float)
    en = scores["enrich_lr"].to_numpy(dtype=float)
    pass_lr = np.isfinite(lr) & (lr >= thr) & np.isfinite(en) & (en >= cfg.enrich_min)
    cand = scores.loc[pass_lr].copy()
    if len(cand):
        cand = P.apply_vetoes(cand, cfg=cfg, lr_threshold=thr, la_cn_suspect=False)
        cand = cand.sort_values("fission_lr", ascending=False, ignore_index=True)
        cand.to_csv(out_dir / f"hires_candidates_{rec['source'].lower()}.csv", index=False)
        counters = P.veto_counters(cand)
    else:
        counters = {name: 0 for name in P.VETOES}
        counters["first_veto"] = {name: 0 for name in P.VETOES}
        counters["n_pass"] = 0
    sens = P.sensitivity_curve(obs, sig, T, cfg, lr_threshold=thr,
                               amplitudes=tuple(sb.get("amplitudes", (1, 2, 3, 5, 10, 20))),
                               n_inject=int(sb.get("n_inject", 1000)), rng=rng, testable=testable, limits=lim)
    survivors = cand[cand["vet_pass"]] if len(cand) else cand
    unexplained = cand[cand["veto_unexplained_by_all_templates"]] if len(cand) else cand
    n_show = int((block.get("report") or {}).get("max_candidates", 50))
    return {
        **rec,
        "threshold": {"lr_used": thr, "why": why, "lr_min_config": cfg.lr_min, "enrich_min": cfg.enrich_min,
                      "max_reduced_chi2": cfg.max_reduced_chi2},
        "null_sample": P.sample_null(lr),
        "null_shuffled": {k: v for k, v in shuffled.items() if k != "lr"},
        "n_pass_lr": int(pass_lr.sum()),
        "vetoes": counters,
        "n_survivors": int(len(survivors)),
        "n_unexplained_above_threshold": int(len(unexplained)),
        "sensitivity": sens,
        "survivors": _records(survivors, elements, n_show),
        "unexplained_top": _records(unexplained, elements, 10),
        "vetoed_top": _records(cand[~cand["vet_pass"]] if len(cand) else cand, elements, 10),
    }


def _verdict(acq: dict | None, per_source: list[dict]) -> tuple[str, str]:
    srcs = (acq or {}).get("sources") or []
    reached = [s for s in srcs if int(s.get("n_rows", 0)) > 0]
    if not reached:
        vs = sorted({str(s.get("verdict")) for s in srcs}) or ["no acquisition record"]
        return VERDICT_NO_DATA, (f"{VERDICT_NO_DATA}: no compilation delivered rows ({', '.join(vs)}). "
                                 "This is an archive-access statement, not a limit on the signature.")
    scored = [s for s in per_source if s.get("threshold")]
    degraded = [s for s in reached if s.get("degraded")]
    prefix = ""
    if degraded:
        prefix = f"{VERDICT_DEGRADED} (" + "; ".join(f"{s['source']}: {s.get('degradation')}" for s in degraded) + "); "
    if not scored:
        why = "; ".join(str(s.get("verdict")) for s in per_source) or "no source was screened"
        return VERDICT_DEGRADED, (f"{VERDICT_DEGRADED}: rows were retrieved but no source could be scored "
                                  f"({why}). This is a coverage statement, not a limit on the signature.")
    n_surv = sum(int(s.get("n_survivors", 0)) for s in scored)
    n_unex = sum(int(s.get("n_unexplained_above_threshold", 0)) for s in scored)
    dec = sorted(set().union(*[set(s.get("decisive_elements_present") or []) for s in scored]))
    if n_surv:
        code = VERDICT_CANDIDATES
        text = (prefix + f"{VERDICT_CANDIDATES}: {n_surv} high-resolution catalogue-level survivors "
                f"(decisive elements available: {', '.join(dec) or 'none'}); {n_unex} above-threshold stars "
                "are UNEXPLAINED_BY_ALL_TEMPLATES and listed separately. None is a detection until the "
                "source spectra are re-analysed homogeneously and the Pb/Ag/Pd limits are confirmed.")
    else:
        code = VERDICT_NO_PATTERN
        text = (prefix + f"{VERDICT_NO_PATTERN}: no compilation star kept the fission-only preference through "
                f"the vetoes (decisive elements available: {', '.join(dec) or 'none'}); {n_unex} above-threshold "
                "stars are UNEXPLAINED_BY_ALL_TEMPLATES. Per CLAUDE.md a null changes the question, not the "
                "write-up: the next question is a homogeneous Pb/Ag/Pd re-analysis of the unexplained stars.")
    if degraded:
        return VERDICT_DEGRADED, text
    return code, text


def stage_hires_assess(block: dict, out_dir: Path, sources: list[str] | None, screen: list[dict] | None,
                       acq: dict | None, stage: str) -> dict:
    if screen is None and (out_dir / "hires_screen.json").exists():
        screen = json.loads((out_dir / "hires_screen.json").read_text()).get("samples")
    screen = screen or []
    per_source = [assess_source(r, block=block, out_dir=out_dir) if r.get("sample") else r for r in screen]
    code, text = _verdict(acq, per_source)
    summary = {
        "channel": "fallout", "tier": "hires", "generated_utc": _now_utc(), "stage": stage,
        "verdict_code": code, "verdict": text,
        "funnel": {
            "n_rows_acquired": {s["source"]: int(s.get("n_rows", 0)) for s in ((acq or {}).get("sources") or [])},
            "per_source": [{"source": s.get("source"), "n_stars": int(s.get("n_stars", 0)),
                            "n_elements": int(s.get("n_elements", 0)),
                            "decisive_elements_present": s.get("decisive_elements_present"),
                            "n_testable": int(s.get("n_testable", 0)),
                            "testable_fraction": s.get("testable_fraction"),
                            "n_above_lr_min": int(s.get("n_above_lr_min", 0)),
                            "n_pass_lr": int(s.get("n_pass_lr", 0)),
                            "n_unexplained_above_threshold": int(s.get("n_unexplained_above_threshold", 0)),
                            "n_survivors": int(s.get("n_survivors", 0))} for s in per_source],
            "n_survivors": sum(int(s.get("n_survivors", 0)) for s in per_source),
        },
        "vetoes": {s.get("source"): s.get("vetoes") for s in per_source if s.get("vetoes")},
        "error_model": {s.get("source"): s.get("error_model") for s in per_source if s.get("error_model")},
        "acquisition": acq,
        "columns_found": {s["source"]: s.get("columns_found") for s in ((acq or {}).get("sources") or [])},
        "thresholds": {s.get("source"): s.get("threshold") for s in per_source if s.get("threshold")},
        "per_source": per_source,
    }
    _json(out_dir / "hires_summary.json", summary)
    (out_dir / "HIRES_REPORT.md").write_text(report_section(summary, heading_level=1))
    print("[fallout/hires] " + json.dumps({"verdict": code, "n_survivors": summary["funnel"]["n_survivors"]}))
    return summary


def hires_run(block: dict, out_dir: Path, sources: list[str] | None, stage: str, *, max_rows=None,
              inject: dict | None = None) -> dict:
    inject = inject or {}
    qf = inject.get("query_fn")
    if stage == "hires-probe":
        return stage_hires_probe(block, out_dir, sources, query_fn=qf)
    acq = None
    if stage in ("hires-acquire", "hires-all"):
        acq = stage_hires_acquire(block, out_dir, sources, max_rows=max_rows, query_fn=qf)
        if stage == "hires-acquire":
            return acq
    if acq is None and (out_dir / "hires_acquisition.json").exists():
        acq = json.loads((out_dir / "hires_acquisition.json").read_text())
    screen = None
    if stage in ("hires-screen", "hires-all"):
        screen = stage_hires_screen(block, out_dir, sources)
        if stage == "hires-screen":
            return {"stage": stage, "samples": screen}
    return stage_hires_assess(block, out_dir, sources, screen, acq, stage)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _ratio_cell(rec: dict, key: str) -> str:
    v = rec.get(f"dec_{key}", float("nan"))
    lm = rec.get(f"dec_{key}_limit", "") or ""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{lm}{fv:+.2f}" if np.isfinite(fv) else "-"


def report_section(summary: dict, *, heading_level: int = 2) -> str:
    h = "#" * heading_level
    L = [f"{h} FALLOUT high-resolution tier — Pb / Ag / Pd / Eu on literature compilations\n",
         f"Generated {summary.get('generated_utc')} (UTC).\n",
         f"**Verdict.** {summary.get('verdict')}\n",
         "The elements that decide the fission vector: Pb (made by the s-process, never by fission), "
         "Ag and Pd (the fission valley; the r-process fills it), Eu (r-process, not fission). Upper "
         "limits enter the likelihood as censored values. Errors are floored at the larger of the peer "
         "scatter and the star-to-star scatter of duplicate literature entries.\n"]
    acq = summary.get("acquisition") or {}
    L.append(f"{h}# Acquisition\n")
    for s in acq.get("sources") or []:
        L.append(f"- **{s.get('source')}**: {s.get('verdict')} — {s.get('n_rows', 0):,} rows from "
                 f"{s.get('tables_used')}")
        if s.get("degradation"):
            L.append(f"  - degradation: {s['degradation']}")
        for line in s.get("log") or []:
            L.append(f"  - {line}")
    if not (acq.get("sources") or []):
        L.append("- no acquisition record")
    L.append("")
    for s in summary.get("per_source") or []:
        L.append(f"{h}# {s.get('source')}\n")
        if s.get("verdict") and not s.get("threshold"):
            L.append(f"{s['verdict']}\n")
            continue
        L.append(f"- stars (after duplicate collapse): **{s.get('n_stars', 0):,}**; elements: "
                 f"**{s.get('n_elements', 0)}** ({', '.join(s.get('elements') or [])})")
        L.append(f"- decisive elements present: **{', '.join(s.get('decisive_elements_present') or []) or 'none'}**")
        pec = s.get("per_element_counts") or {}
        ul = s.get("per_element_upper_limits") or {}
        if pec:
            L.append("- per-element counts (upper limits): " + ", ".join(
                f"{el} {n}" + (f" ({ul[el]} ul)" if ul.get(el) else "") for el, n in pec.items()))
        d = s.get("duplicates") or {}
        L.append(f"- duplicates: {d.get('n_rows_in')} entries -> {d.get('n_stars')} stars, "
                 f"{d.get('n_multi_entry')} multi-entry; duplicate scatter measured for "
                 f"{', '.join(d.get('elements_with_duplicate_scatter') or []) or 'no element'}")
        em = (s.get("error_model") or {}).get("per_element") or {}
        if em:
            L.append("- error floors: " + ", ".join(f"{el} {v['floor_dex']:.2f} ({v.get('source')})"
                                                     for el, v in em.items()))
        tf = s.get("testable_fraction")
        if tf is not None:
            L.append(f"- testable: {s.get('n_testable', 0):,} = {float(tf):.1%}")
        cc = s.get("class_counts") or {}
        if cc:
            L.append("- classification: " + ", ".join(f"{k} {v:,}" for k, v in cc.items()))
        th = s.get("threshold") or {}
        L.append(f"- threshold: ln LR ≥ **{th.get('lr_used', float('nan')):.2f}** ({th.get('why')})")
        ns = (s.get("null_shuffled") or {}).get("quantiles") or {}
        if ns:
            L.append("- shuffled null (ln LR): " + ", ".join(f"{k} {v:.2f}" for k, v in ns.items()))
        L.append(f"- above threshold: **{s.get('n_pass_lr', 0)}**; unexplained: "
                 f"**{s.get('n_unexplained_above_threshold', 0)}**; survivors: **{s.get('n_survivors', 0)}**")
        vt = s.get("vetoes") or {}
        if vt:
            L.append("- vetoes: " + ", ".join(f"{k} {v}" for k, v in vt.items() if k not in ("first_veto", "n_pass")))
        sens = s.get("sensitivity") or []
        if sens:
            L.append(f"\n{h}## Sensitivity\n")
            L.append("| a_f | Δ[Nd/H] | LR pass (testable) | LR+LOO (testable) | LR pass (all) |")
            L.append("|---|---|---|---|---|")
            for r in sens:
                L.append(f"| {r['a_f']:g} | {r['nd_dex']:+.2f} | {r.get('frac_lr_pass_testable', float('nan')):.2f} | "
                         f"{r.get('frac_lr_and_loo_pass_testable', float('nan')):.2f} | "
                         f"{r.get('frac_lr_pass_all', float('nan')):.2f} |")
        for title, key in (("Survivors (pending homogeneous re-analysis)", "survivors"),
                           ("Unexplained by all templates", "unexplained_top")):
            rows = s.get(key) or []
            if rows:
                L.append(f"\n{h}## {title}\n")
                L.append("| star | [Fe/H] | n_el (lim) | ln LR | red. chi2 | LOO min (driver) | heavy≥2σ | [Pb/Nd] | [Ag/Nd] | [Pd/Nd] | [Eu/Nd] |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|")
                for c in rows[:25]:
                    L.append(f"| {c.get('star_id')} | {float(c.get('fe_h', np.nan)):+.2f} | "
                             f"{int(c.get('n_measured', 0))} ({int(c.get('n_limits', 0))}) | "
                             f"{float(c.get('fission_lr', np.nan)):.1f} | {float(c.get('reduced_chi2_best', np.nan)):.1f} | "
                             f"{float(c.get('lr_loo_min', np.nan)):.1f} ({c.get('lr_loo_driver')}) | "
                             f"{c.get('n_heavy_coherent', '-')} | {_ratio_cell(c, 'Pb/Nd')} | "
                             f"{_ratio_cell(c, 'Ag/Nd')} | {_ratio_cell(c, 'Pd/Nd')} | {_ratio_cell(c, 'Eu/Nd')} |")
        L.append("")
    return "\n".join(L) + "\n"


__all__ = ["DEFAULT_SOURCES", "HIRES_STAGES", "SourcePull", "assess_source", "collapse_duplicates",
           "discover_source", "fetch_source", "hires_run", "normalise_hires", "report_section",
           "resolve_hires_columns", "screen_source", "stage_hires_acquire", "stage_hires_assess",
           "stage_hires_probe", "stage_hires_screen"]
