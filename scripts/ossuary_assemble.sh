#!/usr/bin/env bash
# Assemble results/ossuary/sample.parquet from the checkpointed declination
# bands.  Writes an honest NO_DATA_REACHED summary when no band reached us.
set -euo pipefail
python -u - <<'PY'
import glob
import json
import time

import pandas as pd

from seti.config import load_config
from seti.ossuary import acquire as acq
from seti.ossuary.run import out_dir

cfg = load_config()
d = out_dir(cfg)
files = sorted(glob.glob(str(d / "chunks" / "*.parquet")))
print(f"{len(files)} checkpointed declination bands", flush=True)
if not files:
    (d / "summary.json").write_text(json.dumps(
        {"verdict": "NO_DATA_REACHED", "n_input": 0,
         "note": "no declination band returned rows"}, indent=2))
    raise SystemExit(0)
t0 = time.monotonic()
frames = []
for f in files:
    df = pd.read_parquet(f)
    print(f"  {f.split('/')[-1]}: {len(df):,} rows", flush=True)
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
del frames
df = df.drop_duplicates("source_id")
df = acq.harmonise(df)
# String columns cost ~60 bytes per cell in pandas; the flag strings repeat a
# handful of values, so store them as categories before the analysis copies.
for col in ("ph_qual", "cc_flags", "track", "feh_provenance", "flags_gspspec",
            "tmass_ph_qual", "var_flag"):
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].astype("category")
df.to_parquet(d / "sample.parquet", index=False)
# Chunks from before 2026-08 carry no row_limit_hit column (DataFrame.get of a
# missing column is None, which pd.to_numeric collapses to a scalar).
n_capped = (int(pd.to_numeric(df["row_limit_hit"], errors="coerce").fillna(0).sum())
            if "row_limit_hit" in df.columns else -1)
print(f"sample: {len(df):,} stars; {int(df['feh'].notna().sum()):,} with a metallicity; "
      f"rows from row-limited bands: {n_capped if n_capped >= 0 else 'unknown (old chunks)'} "
      f"({time.monotonic() - t0:.0f} s)", flush=True)
PY
