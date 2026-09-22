#!/usr/bin/env bash
# A verdict must never read as a science null when no data was analysed.
set -uo pipefail
python - <<'PY'
import json
import pathlib
import sys

p = pathlib.Path("results/ossuary/summary.json")
if not p.exists():
    print("::error::no summary.json -- the analyze stage produced nothing")
    sys.exit(1)
s = json.loads(p.read_text())
v = s.get("verdict", "")
n = int(s.get("n_input", 0) or 0)
if n == 0 and "NO_DATA_REACHED" not in v:
    print(f"::error::verdict {v!r} with 0 input rows is a disguised no-data run")
    sys.exit(1)
if v == "OK":
    print(f"::notice::{n:,} stars analysed on the {s.get('analysis_path')} path: "
          f"{s.get('n_excess_flagged', 0):,} flagged, {s.get('n_candidates', 0):,} "
          f"survive the gauntlet"
          + (f", {s['n_candidates_after_followup']:,} survive follow-up"
             if "n_candidates_after_followup" in s else "")
          + f"; {s.get('n_from_row_limited_bands', 0):,} rows from row-limited bands")
else:
    print(f"::warning::verdict {v}: {s.get('note', '')}")
PY
