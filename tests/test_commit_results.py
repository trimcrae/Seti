"""`scripts/commit_results.sh`, against real git repositories.

THE INCIDENT IT ENDS.  On 2026-08-27 `tocsin-altfeeds` run 33029076779 spent
169 minutes walking the ATLAS queue, wrote its ledger, committed it, and landed
NOTHING on `main` -- in a run that went green.  `main` had moved during those
169 minutes; `git pull --rebase --autostash || true` hit a conflict in a
generated JSON file and the `|| true` swallowed it, leaving a rebase in progress
with the commit unapplied; `git push` then had nothing to push, said "Everything
up-to-date", and exited 0.

The first test below is that sequence, reproduced against real repositories with
a real conflicting change on the remote.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("scripts/commit_results.sh").resolve()


def git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


@pytest.fixture
def remote_and_clone(tmp_path):
    """A bare 'origin' with one commit, and a working clone of it."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "--initial-branch=main", "-q")

    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "--initial-branch=main", "-q")
    git(seed, "config", "user.email", "t@example.com")
    git(seed, "config", "user.name", "t")
    (seed / "results").mkdir()
    (seed / "results" / "census.json").write_text('{"n": 1}\n')
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "seed")
    git(seed, "remote", "add", "origin", str(origin))
    git(seed, "push", "-q", "origin", "main")

    work = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(origin), str(work))
    git(work, "config", "user.email", "t@example.com")
    git(work, "config", "user.name", "t")
    return origin, work, seed


def run_script(work, message, *paths, expect=0):
    r = subprocess.run(["bash", str(SCRIPT), message, *paths], cwd=str(work),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(work),
                            "RESULTS_PUSH_ATTEMPTS": "2"})
    assert r.returncode == expect, (
        f"exit {r.returncode}, expected {expect}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r


def remote_file(origin, path):
    return git(origin, "show", f"main:{path}").stdout


# --------------------------------------------------------------------------
# The incident.
# --------------------------------------------------------------------------

def test_a_result_lands_even_when_the_branch_moved_under_a_long_run(remote_and_clone):
    """THE REGRESSION: 169 minutes of work, a moved branch, a conflicting file.

    The old pattern rebased, conflicted, swallowed it, and pushed nothing while
    exiting 0.  This must land the result and say so.
    """
    origin, work, seed = remote_and_clone

    # The branch moves while our long run is walking -- and it touches the very
    # file our run also rewrites, which is what made the rebase conflict.
    (seed / "results" / "census.json").write_text('{"n": 999, "from": "another run"}\n')
    git(seed, "commit", "-qam", "someone else's run")
    git(seed, "push", "-q", "origin", "main")

    # Our run's output, written against the OLD checkout.
    (work / "results" / "census.json").write_text('{"n": 40, "from": "the atlas walk"}\n')
    (work / "results" / "atlas").mkdir(parents=True)
    (work / "results" / "atlas" / "summary.json").write_text('{"targets_walked": 40}\n')

    run_script(work, "atlas ledger", "results/census.json", "results/atlas/summary.json")

    assert "targets_walked" in remote_file(origin, "results/atlas/summary.json")
    assert "the atlas walk" in remote_file(origin, "results/census.json")
    # And the other run's commit is still in the history -- nothing was rewritten.
    assert "someone else's run" in git(origin, "log", "--format=%s", "main").stdout


def test_the_old_pattern_really_did_fail_this_way(remote_and_clone):
    """Not a hypothetical.  The superseded sequence, run verbatim.

    If this ever starts passing, the premise of the rewrite is wrong and the
    argument in the script's header needs revisiting.
    """
    origin, work, seed = remote_and_clone

    (seed / "results" / "census.json").write_text('{"n": 999}\n')
    git(seed, "commit", "-qam", "concurrent")
    git(seed, "push", "-q", "origin", "main")

    (work / "results" / "census.json").write_text('{"n": 40}\n')
    old = """
      git add -f -- results/census.json
      git commit -q -m "ours"
      git pull --rebase --autostash origin main || true
      git push origin HEAD:main || true
    """
    r = subprocess.run(["bash", "-c", old], cwd=str(work), capture_output=True, text=True)
    assert r.returncode == 0, "the old pattern exited GREEN -- that was the problem"
    assert '{"n": 40}' not in remote_file(origin, "results/census.json"), (
        "the old pattern was expected to drop the result silently")


# --------------------------------------------------------------------------
# The two rules.
# --------------------------------------------------------------------------

def test_a_vacuous_push_is_caught_by_reading_the_remote_back(remote_and_clone, tmp_path):
    """Rule 2: a zero exit from `git push` is not evidence anything was pushed."""
    origin, work, _seed = remote_and_clone
    (work / "results" / "census.json").write_text('{"n": 7}\n')

    # A `git` that reports a successful push and does nothing.
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "git").write_text(
        '#!/usr/bin/env bash\n'
        'for a in "$@"; do if [ "$a" = "push" ]; then\n'
        '  echo "Everything up-to-date"; exit 0; fi; done\n'
        'exec /usr/bin/git "$@"\n')
    (shim / "git").chmod(0o755)

    r = subprocess.run(["bash", str(SCRIPT), "m", "results/census.json"], cwd=str(work),
                       capture_output=True, text=True,
                       env={"PATH": f"{shim}:/usr/bin:/bin", "HOME": str(work),
                            "RESULTS_PUSH_ATTEMPTS": "2"})
    assert r.returncode == 1, "a push that pushed nothing must not read as success"
    assert "does not contain" in r.stdout
    assert "artifact" in r.stderr, "it must say where the results still are"


def test_a_rebase_left_in_progress_by_an_earlier_step_is_cleared(remote_and_clone):
    """The state the old pattern left behind must not poison this one."""
    origin, work, seed = remote_and_clone
    (seed / "results" / "census.json").write_text('{"n": 999}\n')
    git(seed, "commit", "-qam", "concurrent")
    git(seed, "push", "-q", "origin", "main")

    (work / "results" / "census.json").write_text('{"n": 40}\n')
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "ours")
    git(work, "fetch", "-q", "origin", "main")
    git(work, "rebase", "origin/main", check=False)      # conflicts, stays in progress
    assert (Path(work) / ".git" / "rebase-merge").exists() or \
           (Path(work) / ".git" / "rebase-apply").exists()

    (work / "results" / "census.json").write_text('{"n": 40}\n')
    run_script(work, "recovered", "results/census.json")
    assert '{"n": 40}' in remote_file(origin, "results/census.json")


def test_untracked_files_this_run_generated_survive_for_later_steps(remote_and_clone):
    """`reset --hard` must not eat the artifact upload's inputs."""
    origin, work, _seed = remote_and_clone
    (work / "results" / "census.json").write_text('{"n": 3}\n')
    (work / "results" / "not_committed.json").write_text('{"kept": true}\n')
    run_script(work, "m", "results/census.json")
    assert (work / "results" / "not_committed.json").exists()


# --------------------------------------------------------------------------
# The quiet paths.
# --------------------------------------------------------------------------

def test_nothing_to_commit_is_success_and_says_so(remote_and_clone):
    origin, work, _ = remote_and_clone
    r = run_script(work, "m", "results/census.json")       # unchanged from origin
    assert "identical" in r.stdout


def test_a_path_this_run_did_not_produce_is_skipped_not_fatal(remote_and_clone):
    origin, work, _ = remote_and_clone
    (work / "results" / "census.json").write_text('{"n": 5}\n')
    r = run_script(work, "m", "results/census.json", "results/never_written.json")
    assert "not produced by this run" in r.stdout
    assert '{"n": 5}' in remote_file(origin, "results/census.json")


def test_producing_nothing_at_all_is_success_and_commits_nothing(remote_and_clone):
    origin, work, _ = remote_and_clone
    before = git(origin, "rev-parse", "main").stdout
    r = run_script(work, "m", "results/a.json", "results/b.json")
    assert "none of the requested paths" in r.stdout
    assert git(origin, "rev-parse", "main").stdout == before


def test_no_message_or_no_paths_is_a_usage_error():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 2


# --------------------------------------------------------------------------
# The wiring: a scheduled workflow may never swallow a failed push.
# --------------------------------------------------------------------------

def test_no_scheduled_workflow_swallows_the_result_of_its_push():
    """Scheduled workflows run unattended, which is where silence is fatal.

    A dispatch-only workflow is started by someone who watches it finish; a
    scheduled one has only its exit code to speak with, and `|| echo` takes even
    that away.
    """
    import yaml

    offenders = []
    for path in sorted(Path(".github/workflows").glob("*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        triggers = doc.get(True, doc.get("on")) or {}
        if "schedule" not in triggers:
            continue
        for job in (doc.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                body = step.get("run") or ""
                if "commit_results.sh" in body:
                    continue
                for line in body.splitlines():
                    s = line.strip()
                    if s.startswith("#"):
                        continue
                    if "git push" in s and ("|| echo" in s or "|| true" in s):
                        offenders.append(f"{path.name}: {s}")
                    if "git pull" in s and "|| true" in s:
                        offenders.append(f"{path.name}: {s}")
    assert not offenders, (
        "a scheduled workflow can drop its results in a green run:\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------------------------
# `--prune`: a path whose ABSENCE is the result.
# --------------------------------------------------------------------------

def test_a_finished_runs_deleted_checkpoint_is_committed_as_a_deletion(remote_and_clone):
    """loom-catalogue's case.

    A run that FINISHED deletes `catalogue.inprogress.json`.  If that deletion
    is not committed, the stale checkpoint stays in git as a resume point
    offered for a run that already ended -- which is how the catalogue got
    overwritten by an empty record in the first place.
    """
    origin, work, seed = remote_and_clone
    (seed / "results" / "catalogue.inprogress.json").write_text('{"partial": true}\n')
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "checkpoint")
    git(seed, "push", "-q", "origin", "main")

    # Our run finished: it wrote the catalogue and removed the checkpoint.
    (work / "results" / "catalogue.json").write_text('{"rows": 1371}\n')
    run_script(work, "finished", "results/catalogue.json",
               "--prune", "results/catalogue.inprogress.json")

    assert "rows" in remote_file(origin, "results/catalogue.json")
    listing = git(origin, "ls-tree", "-r", "--name-only", "main").stdout
    assert "catalogue.inprogress.json" not in listing


def test_an_unfinished_runs_checkpoint_is_committed_normally(remote_and_clone):
    origin, work, _ = remote_and_clone
    (work / "results" / "catalogue.inprogress.json").write_text('{"partial": true}\n')
    run_script(work, "unfinished", "--prune", "results/catalogue.inprogress.json")
    assert "partial" in remote_file(origin, "results/catalogue.inprogress.json")


def test_an_ordinary_missing_path_is_never_treated_as_a_deletion(remote_and_clone):
    """Absence only means 'delete' where the caller said so."""
    origin, work, _ = remote_and_clone
    (work / "results" / "new.json").write_text('{"a": 1}\n')
    # Genuinely absent from this run -- not merely left over from the checkout.
    (work / "results" / "census.json").unlink()
    r = run_script(work, "m", "results/new.json", "results/census.json")
    assert "not produced by this run" in r.stdout
    # It existed on the branch and this run made no claim about it: it stays.
    assert '{"n": 1}' in remote_file(origin, "results/census.json")
    assert '{"a": 1}' in remote_file(origin, "results/new.json")


def test_the_helper_is_executable():
    """Every scheduled workflow invokes it as a program.

    A committed-without-the-exec-bit script fails at runtime in all ten of them
    at once, and the failure is a bare `Permission denied` after the science has
    already been computed.
    """
    import os
    assert os.access(SCRIPT, os.X_OK), "scripts/commit_results.sh is not executable"
