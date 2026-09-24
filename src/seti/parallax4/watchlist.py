"""The DR4 release-day watchlist: every Gaia DR3 source_id any channel in this
repository has shortlisted, flagged, vetted or named as a candidate.

On 2026-12-02 the photocentre test (``photocentre.fit_photocentre``) runs on
every row here: for each, does the photocentre move with the brightness (a
blend) or not (the anomaly is on the target)?

Sources, in priority order:

1. Gaia DR3 ids NAMED in STATUS.md or docs/*.md (``Gaia DR3 <id>``) -- the
   curated survivors, each of which a human-readable section argued about;
2. rows of candidate / shortlist / vetted / survivor / final files under
   ``results/`` with a source_id-like column (<= ``big_file_bytes``);
3. the same from larger screen-level files.

A row per (source_id, channel, file); ``watchlist_unique.csv`` collapses to
one row per source_id with the channels that named it and its best priority.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

NAME_PAT = re.compile(r"(candidate|shortlist|survivor|vetted|interest|watch|final|triage|flagged|"
                      r"anomal|outlier|followup|dossier|revet|deepvet)", re.I)
ID_COLS = ("source_id", "gaia_source_id", "gaia_dr3_source_id", "dr3_source_id", "gaia_id",
           "source_id_dr3", "gaia_dr3_id", "gaiadr3_source_id")
CLASS_COLS = ("class", "tier", "verdict", "status", "cradle_class", "classification", "fate",
              "vet_verdict", "final_class", "label")
#: a free-text Gaia id: any standalone 15-19 digit integer in the source_id
#: range (GitHub run ids are 11 digits, TIC/KIC <= 10; commit shas are hex).
#: Written as "Gaia DR3 <id>", "`<id>`", "gaia<id>" or bare in the docs.
MENTION = re.compile(r"(?<![0-9A-Za-z])(?:gaia|Gaia|GAIA)?(?:\s*DR[23]\s*)?(\d{15,19})(?![0-9])")
#: every Gaia DR3 source_id is below 2**63 and above the smallest HEALPix-0 id
SID_MIN, SID_MAX = 4_295_806_720, 6_917_528_997_577_384_320


def _gaia_context(txt: str, m: re.Match) -> bool:
    """A 17-19 digit number is taken as a Gaia id on its own; a 15-16 digit
    one (GALAH sobject_ids, SBDB/Rubin ids live there too) only with "gaia",
    "DR3" or "source_id" within 25 characters before it."""
    if len(m.group(1)) >= 17:
        return True
    before = txt[max(0, m.start() - 25): m.start()].lower()
    return any(k in before for k in ("gaia", "dr3", "source_id", "source id"))


def _valid(x) -> bool:
    try:
        v = int(x)
    except (TypeError, ValueError):
        return False
    return SID_MIN <= v <= SID_MAX


def _channel_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[1] if len(parts) > 2 and parts[0] == "results" else parts[0]


def _from_csv(path: Path) -> list[dict]:
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:  # noqa: BLE001
        return []
    low = {c.lower(): c for c in head.columns}
    idc = next((low[c] for c in ID_COLS if c in low), None)
    if idc is None:
        return []
    cc = next((low[c] for c in CLASS_COLS if c in low), None)
    use = [idc] + ([cc] if cc else [])
    try:
        df = pd.read_csv(path, usecols=use, dtype=str)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for _, r in df.iterrows():
        v = str(r[idc]).split(".")[0].strip()
        if _valid(v):
            out.append({"source_id": int(v), "label": (str(r[cc]) if cc else "")[:80]})
    return out


def _walk_json(obj, out: list[dict], label: str = ""):
    if isinstance(obj, dict):
        lab = next((str(obj[k]) for k in CLASS_COLS if k in obj and isinstance(obj[k], str)), label)
        for k, v in obj.items():
            if str(k).lower() in ID_COLS and _valid(v):
                out.append({"source_id": int(v), "label": lab[:80]})
            elif isinstance(v, str):
                for m in MENTION.finditer(v):
                    if _valid(m.group(1)):
                        out.append({"source_id": int(m.group(1)), "label": lab[:80]})
            else:
                _walk_json(v, out, lab)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json(v, out, label)


def _from_json(path: Path) -> list[dict]:
    try:
        obj = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    _walk_json(obj, out)
    return out


def build_watchlist(root: Path | str = ".", *, big_file_bytes: int = 2_000_000,
                    exclude_channels: tuple[str, ...] = ("parallax4",)) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    rows: list[dict] = []
    stats = {"files_scanned": 0, "files_with_ids": 0, "status_mentions": 0}
    # 1. curated mentions
    for doc in [root / "STATUS.md", *sorted((root / "docs").glob("*.md"))]:
        if not doc.exists():
            continue
        txt = doc.read_text(errors="ignore")
        for m in MENTION.finditer(txt):
            if _valid(m.group(1)) and _gaia_context(txt, m):
                line_start = txt.rfind("\n", 0, m.start()) + 1
                heading = ""
                hpos = txt.rfind("\n#", 0, m.start())
                if hpos >= 0:
                    heading = txt[hpos + 1: txt.find("\n", hpos + 1)].lstrip("# ").strip()[:80]
                rows.append({"source_id": int(m.group(1)), "channel": doc.stem.lower(),
                             "file": str(doc.relative_to(root)), "priority": 1,
                             "label": heading, "context": txt[line_start:line_start + 160].strip()})
                stats["status_mentions"] += 1
    # 2/3. result files
    res = root / "results"
    if res.exists():
        for p in sorted(res.rglob("*")):
            if not p.is_file() or p.suffix not in (".csv", ".json") or not NAME_PAT.search(p.name):
                continue
            ch = _channel_of(p, root)
            if ch in exclude_channels:
                continue
            stats["files_scanned"] += 1
            got = _from_csv(p) if p.suffix == ".csv" else _from_json(p)
            if not got:
                continue
            stats["files_with_ids"] += 1
            pr = 2 if p.stat().st_size <= big_file_bytes else 3
            for g in got:
                rows.append({"source_id": g["source_id"], "channel": ch,
                             "file": str(p.relative_to(root)), "priority": pr,
                             "label": g.get("label", ""), "context": ""})
    df = pd.DataFrame(rows, columns=["source_id", "channel", "file", "priority", "label", "context"])
    df = df.drop_duplicates(["source_id", "channel", "file"]).reset_index(drop=True)
    stats["n_rows"] = int(len(df))
    stats["n_unique"] = int(df["source_id"].nunique()) if len(df) else 0
    stats["by_priority_unique"] = ({int(k): int(v) for k, v in
                                    df.groupby("source_id")["priority"].min().value_counts().items()}
                                   if len(df) else {})
    stats["by_channel_unique"] = ({str(k): int(v) for k, v in
                                   df.groupby("channel")["source_id"].nunique().items()} if len(df) else {})
    return df, stats


def unique_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame(columns=["source_id", "priority", "channels", "n_channels", "labels"])
    g = df.groupby("source_id")
    u = pd.DataFrame({
        "priority": g["priority"].min(),
        "channels": g["channel"].apply(lambda s: ";".join(sorted(set(s)))),
        "n_channels": g["channel"].nunique(),
        "labels": g["label"].apply(lambda s: ";".join(sorted({x for x in s if x}))[:200]),
    }).reset_index()
    return u.sort_values(["priority", "n_channels", "source_id"], ascending=[True, False, True])
