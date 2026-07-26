#!/usr/bin/env python3
"""Pull the quantitative skeleton out of the fetched Project Hephaistos texts.

Reads results/hephlit/txt_<arxivid>.txt (produced by hephlit_fetch.py on the
Actions runner) and prints, per paper, the passages that carry the numbers we
actually need: parent-sample sizes, the pipeline cascade, the model grid
(temperature / covering fraction), the atmosphere models, and the candidate
table. Pure reporting -- it does not modify anything.

Usage:  python scripts/hephlit_extract.py [arxiv_id ...]
"""
from __future__ import annotations

import pathlib
import re
import sys

OUT = pathlib.Path("results/hephlit")

# Each theme is a set of regexes; we print every matching line with context.
THEMES: dict[str, list[str]] = {
    "SAMPLE": [
        r"\b\d[\d,\.]*\s*(?:million|,\d{3},\d{3})\b",
        r"\bsample of\b", r"\bparent sample\b", r"\bwe start\b", r"\bstarting\b",
        r"\bpc\b.*\bsources\b", r"\bwithin\s+\d+\s*pc\b",
    ],
    "CASCADE": [
        r"\bsurviv\w+", r"\bwe are left with\b", r"\breduces? the sample\b",
        r"\bafter (?:this|the) (?:cut|filter|step)\b", r"\bremain(?:ing|s)?\b",
        r"\bfilter\b", r"\brejected\b", r"\bdiscard\w+",
    ],
    "QUALITY": [
        r"\bRUWE\b", r"\bruwe\b", r"\bastrometric_excess_noise\b",
        r"\bparallax_over_error\b", r"\bph_qual\b", r"\bcc_flags?\b",
        r"\bext_flag\b", r"\bvar_flg\b", r"\bnb_?blend\b", r"\bcontamination\b",
        r"\bsignal-to-noise\b", r"\bS/N\b",
    ],
    "GRID": [
        r"\bT_?d\b", r"\btemperature\b.*\bK\b", r"\bcovering (?:fraction|factor)\b",
        r"\bfilling factor\b", r"\bgrid\b", r"\bblackbody\b", r"\bblack body\b",
        r"\bemissivity\b", r"\bgray\b", r"\bgrey\b",
    ],
    "ATMOS": [
        r"\bBT-?Settl\b", r"\bATLAS9?\b", r"\bKurucz\b", r"\bPHOENIX\b",
        r"\bCoelho\b", r"\bMIST\b", r"\bPARSEC\b", r"\bisochrone\b",
        r"\bmodel atmosphere\b", r"\bsynthetic spectra\b", r"\bstellar model\b",
    ],
    "FIT": [
        r"\bRMSE\b", r"\bchi\b", r"\bchi2\b", r"\bgoodness[- ]of[- ]fit\b",
        r"\bresidual\b", r"\bAIC\b", r"\bBIC\b",
    ],
    "CANDIDATES": [
        r"\bcandidate [A-J]\b", r"\bGaia DR3 \d{10,}\b", r"\bsource_?id\b",
        r"\bJ\d{6}[+-]\d{6}\b",
    ],
    "BANDS": [
        r"\bW1\b", r"\bW2\b", r"\bW3\b", r"\bW4\b", r"\bJ, ?H, ?K", r"\bKs\b",
        r"\bG_?BP\b", r"\bG_?RP\b", r"\bAllWISE\b", r"\bunWISE\b", r"\bCatWISE\b",
    ],
}


def scan(path: pathlib.Path, themes: list[str] | None = None) -> None:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    print(f"\n{'='*78}\n### {path.name}   ({len(lines)} lines)\n{'='*78}")
    for theme, pats in THEMES.items():
        if themes and theme not in themes:
            continue
        rx = re.compile("|".join(pats), re.IGNORECASE)
        hits = [(i, ln) for i, ln in enumerate(lines) if rx.search(ln)]
        if not hits:
            continue
        print(f"\n--- {theme}  ({len(hits)} matching lines) ---")
        shown: set[int] = set()
        for i, ln in hits:
            if i in shown:
                continue
            shown.add(i)
            print(f"{i:5d}| {ln.rstrip()}")


def main() -> None:
    ids = sys.argv[1:]
    files = sorted(OUT.glob("txt_*.txt"))
    if ids:
        files = [f for f in files if any(i in f.name for i in ids)]
    if not files:
        print(f"no txt_*.txt under {OUT} -- run the hephlit workflow first")
        return
    for f in files:
        scan(f)


if __name__ == "__main__":
    main()
