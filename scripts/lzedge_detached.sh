#!/usr/bin/env bash
# Run one lzedge stage detached from the calling shell (survives tool timeouts),
# logging to $2/<stage>.log and touching $2/<stage>.done (or .failed) at the end.
stage="$1"; logdir="$2"; shift 2
mkdir -p "$logdir"
rm -f "$logdir/$stage.done" "$logdir/$stage.failed"
setsid nohup bash -c "python -m seti.lzedge.run --stage $stage --out results/lzedge $* > '$logdir/$stage.log' 2>&1 && touch '$logdir/$stage.done' || touch '$logdir/$stage.failed'" > /dev/null 2>&1 &
echo "launched $stage (pid $!)"
