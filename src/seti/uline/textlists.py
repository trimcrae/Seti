"""U-line tables that VizieR does not serve: AAS machine-readable and CDS files.

The Orion KL survey of Crockett et al. 2014 (ApJ 787, 112) is the one U-line
list in this channel with real statistics --- ~1,730 unidentified features out
of ~13,000, **with intensities** --- and it is *measured absent* from VizieR:
``J/ApJ/787/112`` returns no table over TAP, empty metadata on three ASU
mirrors, and a 404 on the CDS ReadMe (``config/uline.yaml``, run 35038662696).
A catalogue that was never deposited is not reached by trying VizieR harder.

So this module reads the two formats the same tables are published in
elsewhere, from a plain URL:

**AAS machine-readable tables** (``*_mrt.txt`` on IOPscience) and **CDS
``ReadMe`` + ``.dat`` pairs** share one format --- a *Byte-by-byte Description*
block giving each column's byte range, Fortran format, unit and label, followed
(MRT) or accompanied (CDS) by fixed-width data.  :func:`parse_byte_by_byte`
reads the block, :func:`read_fixed_width` applies it, and
:func:`mrt_table` does both for a single file.

Everything is a ladder with a ledger.  :func:`fetch_first` tries each URL in
order and records the status, the byte count and the error of **every** attempt,
because "which door answered" is a result in itself: the whole point of this
module is that the front door was shut.  No row is ever synthesised; a file that
does not parse yields an empty table and says why.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .acquire import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    AcquisitionLog,
    fetch_text,
    frequency_scale,
    resolve_line_columns,
    unidentified_mask,
)

_SEP_RE = re.compile(r"^-{10,}\s*$")
_BYTES_RE = re.compile(
    r"^\s*(\d+)\s*(?:-\s*(\d+))?\s+"            # byte range (1-based, inclusive)
    r"([AIFEG]\d+(?:\.\d+)?|\d*[AIFEG]\d+(?:\.\d+)?)\s+"   # Fortran format
    r"(\S+)\s+"                                  # units ("---" when none)
    r"(\S+)\s*"                                  # label
    r"(.*)$")                                    # explanation
_FILE_RE = re.compile(r"Byte-by-byte Description of file:\s*(\S+)", re.IGNORECASE)


def parse_byte_by_byte(text: str, filename: str | None = None) -> list[dict]:
    """The column specification of an MRT / CDS ReadMe byte-by-byte block.

    Returns ``[{"start": 0-based, "stop": exclusive, "format", "unit", "label",
    "description"}]``.  ``filename`` selects one block when the text describes
    several files (a CDS ReadMe describes them all); without it the FIRST block
    is taken.  A ``---`` unit becomes ``""``.
    """
    lines = (text or "").splitlines()
    blocks: list[tuple[str, int, int]] = []
    cur_name, cur_start = None, None
    for i, ln in enumerate(lines):
        m = _FILE_RE.search(ln)
        if m:
            if cur_name is not None and cur_start is not None:
                blocks.append((cur_name, cur_start, i))
            cur_name, cur_start = m.group(1), i
    if cur_name is not None and cur_start is not None:
        blocks.append((cur_name, cur_start, len(lines)))
    if not blocks:
        blocks = [("", 0, len(lines))]
    chosen = None
    if filename:
        for name, a, b in blocks:
            if str(filename).lower() in str(name).lower() or str(name).lower() in str(filename).lower():
                chosen = (a, b)
                break
    if chosen is None:
        chosen = (blocks[0][1], blocks[0][2])
    out: list[dict] = []
    for ln in lines[chosen[0]:chosen[1]]:
        if _SEP_RE.match(ln) and out:
            break                                   # the block ends at the next rule
        m = _BYTES_RE.match(ln)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2) or m.group(1))
        if hi < lo or hi > 10000:
            continue
        unit = m.group(4)
        out.append({"start": lo - 1, "stop": hi, "format": m.group(3),
                    "unit": "" if unit in ("---", "--") else unit,
                    "label": m.group(5), "description": m.group(6).strip()})
    return out


def _is_numeric_format(fmt: str) -> bool:
    return bool(re.search(r"[IFEG]", str(fmt).upper()))


def read_fixed_width(spec: list[dict], text: str, *, min_line_len: int | None = None
                     ) -> pd.DataFrame:
    """Apply a byte-by-byte ``spec`` to the data part of ``text``.

    The data are the lines **after the last rule** (``-----``) that are at least
    as long as the first column's byte range and are not comments.  Numeric
    columns are coerced with ``errors="coerce"`` so a blank or an upper-limit
    marker becomes NaN rather than killing the row.
    """
    if not spec:
        return pd.DataFrame()
    lines = (text or "").splitlines()
    last_rule = max((i for i, ln in enumerate(lines) if _SEP_RE.match(ln)), default=-1)
    data = lines[last_rule + 1:]
    need = min_line_len if min_line_len is not None else max(c["start"] + 1 for c in spec)
    rows = [ln for ln in data if len(ln.rstrip()) >= need and not ln.lstrip().startswith("#")]
    if not rows:
        return pd.DataFrame()
    out = {}
    for c in spec:
        vals = [ln[c["start"]:c["stop"]].strip() if len(ln) > c["start"] else "" for ln in rows]
        if _is_numeric_format(c["format"]):
            # No `.replace("", np.nan)` first: `to_numeric(errors="coerce")`
            # already sends an empty field to NaN, and replacing into a
            # pandas-3 `str` column (which holds pd.NA, not np.nan) is exactly
            # the kind of dtype-dependent step that works on the sandbox's
            # pandas 2 and fails on the runner's pandas 3.
            out[c["label"]] = pd.to_numeric(pd.Series(vals), errors="coerce")
        else:
            out[c["label"]] = pd.Series(vals)
    df = pd.DataFrame(out)
    df.attrs["units"] = {c["label"]: c["unit"] for c in spec}
    df.attrs["descriptions"] = {c["label"]: c["description"] for c in spec}
    return df


def mrt_table(text: str, filename: str | None = None) -> pd.DataFrame:
    """An AAS machine-readable table (header and data in one file) as a frame."""
    spec = parse_byte_by_byte(text, filename)
    return read_fixed_width(spec, text)


# ---------------------------------------------------------------------------
# the fetch ladder
# ---------------------------------------------------------------------------
def fetch_first(urls, *, fetch_fn=None, timeout: float = 120.0, retries: int = 2,
                log: AcquisitionLog | None = None, stage: str = "textlist",
                min_bytes: int = 200, must_match: str | None = None
                ) -> tuple[str | None, str | None, list[dict]]:
    """Try each URL in order; return ``(text, url, attempts)``.

    ``attempts`` records every URL with its status, bytes and error text — the
    ledger this channel owes for an absence claim.  ``must_match`` (a regex)
    rejects a body that came back 200 but is a login page or an error document:
    a route that answers with the wrong thing is a failure, and saying so is the
    difference between "the file is not there" and "we were served a banner".
    """
    attempts: list[dict] = []
    rx = re.compile(must_match, re.IGNORECASE) if must_match else None
    for url in list(urls or []):
        rec = {"url": str(url), "status": STATUS_FAILED}
        try:
            body = fetch_text(str(url), fetch_fn=fetch_fn, retries=int(retries),
                              timeout=float(timeout))
        except Exception as exc:                               # noqa: BLE001
            rec["error"] = repr(exc)[:1200]
            attempts.append(rec)
            if log:
                log.record(stage, str(url), error=repr(exc))
            continue
        n = len(body or "")
        rec["n_bytes"] = n
        if body is None:
            # fetch_text RETURNS None once its retries are spent; it does not
            # raise.  A route that came back empty is a failed route.
            rec["error"] = "no body after retries (fetch_text returned None)"
            attempts.append(rec)
            if log:
                log.record(stage, str(url), error="empty body")
            continue
        if n < int(min_bytes):
            rec.update({"status": STATUS_ZERO, "error": f"only {n} bytes"})
            attempts.append(rec)
            continue
        if rx is not None and not rx.search(body):
            rec.update({"status": STATUS_ZERO,
                        "error": f"body does not match {must_match!r}",
                        "head": body[:300]})
            attempts.append(rec)
            continue
        rec["status"] = STATUS_OK
        attempts.append(rec)
        if log:
            log.record(stage, str(url), rows=n)
        return body, str(url), attempts
    return None, None, attempts


_LINK_RE = re.compile(r"""(?:href|src|content)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def find_links(html: str, pattern: str, base: str = "") -> list[str]:
    """Every link in ``html`` whose URL matches ``pattern``, absolutised on ``base``.

    How the MRT files of an ApJ paper are found without guessing their names:
    the article's data page names them (``apj493713t1_mrt.txt``), and that name
    carries an internal manuscript id no caller could have constructed.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    out: list[str] = []
    for m in _LINK_RE.finditer(str(html or "")):
        u = m.group(1).strip()
        if not rx.search(u):
            continue
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/") and base:
            root = re.match(r"^(https?://[^/]+)", base)
            u = (root.group(1) if root else "") + u
        elif not u.startswith("http") and base:
            u = base.rstrip("/") + "/" + u.lstrip("./")
        if u not in out:
            out.append(u)
    return out


def canonical_line_table(df: pd.DataFrame, *, column_patterns=None, uline_patterns=None,
                         all_unidentified: bool = False) -> pd.DataFrame:
    """A parsed table in the channel's canonical column set.

    Same output as :func:`seti.uline.acquire.fetch_line_table`, so a source read
    from a text file and one read from VizieR are indistinguishable downstream:
    ``row, freq_mhz, freq_err_mhz, ident, intensity, transition, unidentified``.
    The frequency unit comes from the byte-by-byte block's own unit, falling
    back to the magnitude of the column.
    """
    if df is None or not len(df):
        return pd.DataFrame()
    roles = resolve_line_columns(list(df.columns), column_patterns)
    if not roles.get("freq"):
        return pd.DataFrame()
    units = df.attrs.get("units") or {}
    out = pd.DataFrame({"row": np.arange(len(df))})
    scale, how = frequency_scale(units.get(roles["freq"]), df[roles["freq"]])
    out["freq_mhz"] = pd.to_numeric(df[roles["freq"]], errors="coerce") * scale
    if roles.get("freq_err"):
        escale, _ = frequency_scale(units.get(roles["freq_err"]) or units.get(roles["freq"]))
        out["freq_err_mhz"] = pd.to_numeric(df[roles["freq_err"]], errors="coerce") * escale
    else:
        out["freq_err_mhz"] = np.nan
    out["ident"] = df[roles["ident"]].astype(str) if roles.get("ident") else ""
    out["intensity"] = (pd.to_numeric(df[roles["intensity"]], errors="coerce")
                        if roles.get("intensity") else np.nan)
    out["transition"] = (df[roles["transition"]].astype(str)
                         if roles.get("transition") else "")
    out["unidentified"] = (np.ones(len(out), dtype=bool) if all_unidentified
                           else unidentified_mask(out["ident"].tolist(), uline_patterns))
    out = out[np.isfinite(out["freq_mhz"])].reset_index(drop=True)
    out.attrs.update({"freq_scale": how, "roles": roles,
                      "units": {r: units.get(c, "") for r, c in roles.items() if c}})
    return out


def fetch_text_line_table(spec: dict, *, fetch_fn=None, column_patterns=None,
                          uline_patterns=None, log: AcquisitionLog | None = None,
                          timeout: float = 120.0, retries: int = 2) -> dict:
    """Run one source's non-VizieR route ladder and return the table + the ledger.

    ``spec`` (from ``config/uline.yaml``) understands:

    ``urls``            direct URLs to an MRT or CDS ``.dat`` file, tried in order
    ``index_urls``      pages to scrape for data-file links first
    ``link_pattern``    which links on those pages are data files
    ``readme_urls``     a CDS ReadMe to take the byte-by-byte block from, when
                        the data file carries none of its own
    ``readme_file``     which block of that ReadMe (a filename)
    ``must_match``      a regex a fetched body must contain to count as data
    ``all_unidentified``  every row of this file is a U-line

    The return is always a record: ``status``, every ``attempts`` entry with its
    error, the columns found, and the table (possibly empty).  A ladder that
    ends with nothing is an honest ``QUERY_FAILED`` naming every door tried.
    """
    rec: dict = {"route": "text", "attempts": [], "status": STATUS_FAILED,
                 "urls_tried": [], "n_rows": 0}
    urls = [str(u) for u in (spec.get("urls") or [])]
    for idx in (spec.get("index_urls") or []):
        body, url, att = fetch_first([idx], fetch_fn=fetch_fn, timeout=timeout, retries=retries,
                                     log=log, stage="textlist_index")
        rec["attempts"].extend(att)
        if body:
            found = find_links(body, str(spec.get("link_pattern") or r"_mrt\.txt$"), base=url or "")
            rec.setdefault("links_found", []).extend(found)
            urls.extend(u for u in found if u not in urls)
    rec["urls_tried"] = urls
    if not urls:
        rec["error"] = "no candidate data URL (none configured and none found on the index pages)"
        return rec
    body, url, att = fetch_first(urls, fetch_fn=fetch_fn, timeout=timeout, retries=retries,
                                 log=log, stage="textlist",
                                 must_match=spec.get("must_match"))
    rec["attempts"].extend(att)
    if not body:
        rec["error"] = "every data URL failed"
        return rec
    rec["url"] = url
    rec["n_bytes"] = len(body)
    spec_cols = parse_byte_by_byte(body, spec.get("readme_file"))
    if not spec_cols and (spec.get("readme_urls")):
        rm, rurl, ratt = fetch_first(spec["readme_urls"], fetch_fn=fetch_fn, timeout=timeout,
                                     retries=retries, log=log, stage="textlist_readme")
        rec["attempts"].extend(ratt)
        if rm:
            rec["readme_url"] = rurl
            spec_cols = parse_byte_by_byte(rm, spec.get("readme_file"))
    if not spec_cols:
        rec.update({"status": STATUS_ZERO,
                    "error": "no byte-by-byte description found in the file or any ReadMe",
                    "head": body[:500]})
        return rec
    rec["columns"] = [{k: c[k] for k in ("label", "unit", "format", "description")}
                      for c in spec_cols]
    raw = read_fixed_width(spec_cols, body)
    if not len(raw):
        rec.update({"status": STATUS_ZERO, "error": "byte-by-byte block parsed but no data rows"})
        return rec
    df = canonical_line_table(raw, column_patterns=column_patterns,
                              uline_patterns=uline_patterns,
                              all_unidentified=bool(spec.get("all_unidentified")))
    if not len(df):
        rec.update({"status": STATUS_ZERO,
                    "error": "no frequency column resolved among "
                             f"{[c['label'] for c in spec_cols][:30]}"})
        return rec
    rec.update({"status": STATUS_OK, "n_rows": int(len(df)),
                "n_unidentified": int(df["unidentified"].sum()),
                "has_intensity": bool(np.isfinite(df["intensity"]).any()),
                "freq_scale": df.attrs.get("freq_scale"), "roles": df.attrs.get("roles"),
                "fmin_mhz": float(df["freq_mhz"].min()),
                "fmax_mhz": float(df["freq_mhz"].max())})
    rec["table"] = df
    return rec


__all__ = ["canonical_line_table", "fetch_first", "fetch_text_line_table", "find_links",
           "mrt_table", "parse_byte_by_byte", "read_fixed_width"]
