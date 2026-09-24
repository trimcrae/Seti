"""growth_direct.yml: each measure shard uploads ONLY its own files.

Run 35859780689's assess merged every shard artifact with merge-multiple.
Each artifact had been globbed as ``shard_*.csv`` and so also carried stale
copies of the other shards' files as checked out at job start; a shard that
finished early overwrote the fresh results of shards still running.  The
committed summary then said 4,358 rows and 368 NOT_REACHED while the committed
shard files held all 4,725 --- a summary that did not describe its own run.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github/workflows/growth_direct.yml"


def test_measure_artifact_carries_only_this_shards_files():
    wf = yaml.safe_load(WF.read_text())
    steps = wf["jobs"]["measure"]["steps"]
    uploads = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert uploads, "measure must upload its shard"
    for s in uploads:
        paths = [p.strip() for p in str(s["with"]["path"]).splitlines() if p.strip()]
        assert paths
        for p in paths:
            assert "shard_*" not in p, p
            assert "steps.tag.outputs.s" in p, p
    tag = [s for s in steps if s.get("id") == "tag"]
    assert tag and "printf '%02d'" in tag[0]["run"]
