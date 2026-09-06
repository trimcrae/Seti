"""FALLOUT acquisition: GALAH DR4 (and optionally APOGEE DR17) via TAILINGS.

Nothing here talks to an archive directly. The transport, the route prober,
the memmapped FITS reader, the runtime schema discovery and the canonical
renaming all live in :mod:`seti.tailings.acquire` and are reused as-is; this
module only (a) orders the bulk-download routes so the URL that **actually
answered on the runner** (``cloud.datacentral.org.au`` -- see
``results/tailings/provenance.json``) is tried first, (b) widens the selection
box so that dwarfs *and* giants come down in one pull and are split at screen
time, and (c) discovers and carries the extra columns this channel needs that
TAILINGS never asked for: ``flag_sp``, an age, a binary flag.

Every column that was looked for is reported as found-or-not in
``columns_found``; nothing is assumed about the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..tailings import acquire as A

#: Extra columns the vetoes use, discovered by pattern at runtime. The first
#: pattern that matches a column wins; a canonical name that matches nothing is
#: reported as absent and the veto that needs it is disabled *and says so*.
EXTRA_COLUMN_PATTERNS: dict[str, tuple[str, ...]] = {
    "flag_sp": (r"^flag_sp$", r"^starflag$", r"^aspcapflag$", r"^flag_cannon$", r"^sp_flag$"),
    "flag_fe_h": (r"^flag_fe_h$", r"^fe_h_flag$", r"^flag_feh$"),
    "e_teff": (r"^e_teff$", r"^teff_err$"),
    "e_logg": (r"^e_logg$", r"^logg_err$"),
    "e_fe_h": (r"^e_fe_h$", r"^fe_h_err$"),
    "age": (r"^age$", r"^age_bstep$", r"^age_50$", r"^age_gyr$"),
    "mass": (r"^mass$", r"^mass_bstep$", r"^m_act$"),
    "log_lum": (r"^log_lum$", r"^log_lum_bstep$"),
    "binary_flag": (r"^flag_binary$", r"^is_binary$", r"^binary$", r"^binary_flag$",
                    r"^sb2_flag$", r"^flag_sb2$", r"^n_comp$"),
    "vsini": (r"^vsini$", r"^vbroad$"),
    "rv_err": (r"^e_rv_comp_1$", r"^e_rv$", r"^rv_err$", r"^e_rv_galah$"),
    "ebv": (r"^ebv$", r"^e_b-v$"),
    "chi2_sp": (r"^chi2_sp$", r"^chi2$"),
}


def _canon(name: str) -> str:
    s = str(name).strip().lower()
    return re.sub(r"_+", "_", s).strip("_")


def resolve_extra_columns(columns, patterns: dict[str, tuple[str, ...]] | None = None
                          ) -> dict[str, str | None]:
    """{canonical: original-or-None} for every extra column this channel wants."""
    patterns = patterns or EXTRA_COLUMN_PATTERNS
    canon = {_canon(c): str(c) for c in columns}
    out: dict[str, str | None] = {}
    for key, pats in patterns.items():
        hit = None
        for pat in pats:
            rx = re.compile(pat)
            hit = next((orig for cc, orig in canon.items() if rx.match(cc)), None)
            if hit is not None:
                break
        out[key] = hit
    return out


def routes_from_config(block: dict, survey: str) -> tuple[A.DownloadRoute, ...]:
    """Ordered bulk-download routes: the config's list first, TAILINGS' registry after.

    The config puts the runner-proven URL first. Anything TAILINGS knows that
    the config does not mention is appended, so a renumbered host still has a
    fallback path.
    """
    seen: set[str] = set()
    routes: list[A.DownloadRoute] = []
    for r in ((block.get("sources") or {}).get(survey) or []):
        url = str(r.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        routes.append(A.DownloadRoute(str(r.get("name", url)), url, str(r.get("note", "")),
                                      bool(r.get("abundances", True))))
    for r in A.FILE_ROUTES.get(survey, ()):
        if r.url not in seen:
            seen.add(r.url)
            routes.append(r)
    return tuple(routes)


def _column_picker(patterns):
    def pick(colnames) -> list[str]:
        want = list(A._default_column_picker(colnames))
        extras = resolve_extra_columns(colnames, patterns)
        want += [v for v in extras.values() if v is not None]
        return list(dict.fromkeys(want))
    return pick


@dataclass
class SurveyPull:
    """One survey's pull: the canonical table plus what was found and how."""

    survey: str
    table: pd.DataFrame
    acquisition: A.Acquisition
    columns_found: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(len(self.table))

    def provenance(self) -> dict:
        p = self.acquisition.provenance()
        p["columns_found"] = self.columns_found
        p["log"] = list(self.log)
        return p


def fetch_survey(
    survey: str,
    *,
    block: dict,
    max_rows: int | None = None,
    cache_dir: str | Path | None = None,
    route_probe_fn=None,
    read_fn=None,
    probe_fn=None,
    query_fn=None,
    extra_patterns: dict | None = None,
) -> SurveyPull:
    """Pull one survey through TAILINGS' acquisition with FALLOUT's box and extras.

    ``route_probe_fn`` / ``read_fn`` / ``probe_fn`` / ``query_fn`` are the same
    injection points TAILINGS exposes, so the offline suite can drive the whole
    path without a socket. The raw frame the reader returns is captured so the
    extra columns (which ``normalize`` does not keep) can be re-attached to the
    canonical table by row.
    """
    pull = dict(block.get("pull") or {})
    patterns = dict(EXTRA_COLUMN_PATTERNS)
    for k, v in (block.get("extra_columns") or {}).items():
        patterns[k] = tuple(v)
    if extra_patterns:
        patterns.update({k: tuple(v) for k, v in extra_patterns.items()})

    store: dict = {}
    inner = read_fn or (lambda url, sel: A.download_and_read(
        url, selection=sel, cache_dir=cache_dir, column_picker=_column_picker(patterns)))

    def capture(url, sel):
        raw = inner(url, sel)
        store["raw"] = raw
        store["url"] = url
        return raw

    max_rows = int(max_rows if max_rows is not None else pull.get("max_rows", 1_200_000))
    kwargs = dict(
        teff_max=float(pull.get("teff_max", 6500.0)),
        teff_min=float(pull.get("teff_min", 4000.0)),
        logg_min=float(pull.get("logg_min", 0.5)),
        snr_min=float(pull.get("snr_min", 30.0)),
        feh_min=float(pull.get("feh_min", -1.5)),
        max_rows=max_rows,
        routes=routes_from_config(block, survey),
        route_probe_fn=route_probe_fn,
        read_fn=capture,
        cache_dir=cache_dir,
    )
    if probe_fn is not None or query_fn is not None:
        kwargs["probe_fn"] = probe_fn
        kwargs["query_fn"] = query_fn
        kwargs["use_files"] = True
    acq = A.fetch_survey(survey, **kwargs)

    log: list[str] = [f"{survey}: acquisition verdict {acq.verdict} via {acq.route} "
                      f"from {acq.source_used} ({acq.n_rows} rows, {len(acq.elements)} elements)"]
    if acq.degradation:
        log.append(f"{survey}: degradation: {acq.degradation}")

    table = acq.table
    found: dict = {
        "elements": list(acq.elements),
        "params": dict(acq.param_columns or {}),
        "extras": {},
        "extras_absent": [],
        "flag_columns": {},
        "error_columns": {},
    }
    if acq.n_rows and isinstance(table, pd.DataFrame):
        for el in acq.elements:
            if f"f_{el}" in table.columns:
                found["flag_columns"][el] = f"f_{el}"
            if f"e_{el}" in table.columns:
                found["error_columns"][el] = f"e_{el}"
        raw = store.get("raw")
        if raw is not None and acq.route == "file":
            extras = resolve_extra_columns(raw.columns, patterns)
            found["extras_absent"] = sorted(k for k, v in extras.items() if v is None)
            present = {k: v for k, v in extras.items() if v is not None}
            selection = A.Selection(teff_min=kwargs["teff_min"], teff_max=kwargs["teff_max"],
                                    logg_min=kwargs["logg_min"], snr_min=kwargs["snr_min"],
                                    feh_min=kwargs["feh_min"], max_rows=max_rows)
            sel_df, _ = A.apply_selection(raw, selection)
            if len(sel_df) == len(table):
                table = table.copy()
                for canon, orig in present.items():
                    if canon in table.columns:
                        continue
                    v = sel_df[orig].to_numpy()
                    if v.dtype.kind in "SUO":
                        table[canon] = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy()
                    else:
                        table[canon] = v
                    found["extras"][canon] = orig
                log.append(f"{survey}: extra columns attached: "
                           + (", ".join(f"{k}<-{v}" for k, v in found['extras'].items()) or "none"))
            else:
                log.append(f"{survey}: extra columns NOT attached -- selection re-application "
                           f"gave {len(sel_df)} rows against {len(table)} in the canonical table")
        else:
            log.append(f"{survey}: extra columns not available on route {acq.route}")
        if found["extras_absent"]:
            log.append(f"{survey}: extra columns absent from the catalogue: "
                       + ", ".join(found["extras_absent"]))
        missing_flags = [el for el in acq.elements if el not in found["flag_columns"]]
        if missing_flags:
            log.append(f"{survey}: no per-element flag column for {', '.join(missing_flags)}; "
                       "those elements are used unflagged")
    return SurveyPull(survey=survey, table=table if isinstance(table, pd.DataFrame) else pd.DataFrame(),
                      acquisition=acq, columns_found=found, log=log)


def probe_routes(block: dict, surveys: list[str], *, route_probe_fn=None,
                 min_bytes: int = 1_000_000) -> dict:
    """HEAD/GET every registered route and report; no download."""
    out = {}
    for sv in surveys:
        routes = routes_from_config(block, sv)
        report = A.probe_download_routes(routes, probe_fn=route_probe_fn, min_bytes=min_bytes)
        out[sv] = {
            "n_routes": len(report),
            "n_live": int(sum(1 for r in report if int(r.get("status") or 0) in (200, 206))),
            "n_eligible": int(sum(1 for r in report if r.get("eligible"))),
            "routes": report,
        }
    return out


def split_samples(table: pd.DataFrame, samples: dict) -> dict[str, pd.DataFrame]:
    """Apply each named sample box (dwarf / giant) to the pulled table."""
    out = {}
    teff = pd.to_numeric(table.get("teff"), errors="coerce").to_numpy(dtype=float) \
        if "teff" in table.columns else np.full(len(table), np.nan)
    logg = pd.to_numeric(table.get("logg"), errors="coerce").to_numpy(dtype=float) \
        if "logg" in table.columns else np.full(len(table), np.nan)
    feh = pd.to_numeric(table.get("fe_h"), errors="coerce").to_numpy(dtype=float) \
        if "fe_h" in table.columns else np.full(len(table), np.nan)
    snr = pd.to_numeric(table.get("snr"), errors="coerce").to_numpy(dtype=float) \
        if "snr" in table.columns else np.full(len(table), np.inf)
    for name, box in (samples or {}).items():
        m = np.ones(len(table), dtype=bool)
        m &= np.isfinite(teff) & (teff > float(box.get("teff_min", -np.inf))) \
            & (teff < float(box.get("teff_max", np.inf)))
        m &= np.isfinite(logg) & (logg > float(box.get("logg_min", -np.inf))) \
            & (logg < float(box.get("logg_max", np.inf)))
        if "feh_min" in box:
            m &= np.isfinite(feh) & (feh >= float(box["feh_min"]))
        if "snr_min" in box:
            m &= ~np.isfinite(snr) | (snr >= float(box["snr_min"]))
        out[name] = table.loc[m].reset_index(drop=True)
    return out


__all__ = [
    "EXTRA_COLUMN_PATTERNS",
    "SurveyPull",
    "fetch_survey",
    "probe_routes",
    "resolve_extra_columns",
    "routes_from_config",
    "split_samples",
]
