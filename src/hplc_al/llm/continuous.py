"""User-started, bounded six-round execution with local Git seals and resume."""

import fcntl
import hashlib
import subprocess

from ..common import ROOT, atomic_json, sha
from . import runner
from .full_pool import BUDGETS, ROUNDS
from .responses_transport import preflight, require_key, settings


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def check_index():
    prefix = str(runner.STUDY.relative_to(ROOT)) + "/"
    paths = git("diff", "--cached", "--name-only", "-z").decode().split("\0")
    if any(path and not path.startswith(prefix) for path in paths):
        raise RuntimeError(
            "Git index contains unrelated staged changes; commit or unstage them before starting"
        )


def check_checkout():
    check_index()
    git("var", "GIT_AUTHOR_IDENT")
    git("var", "GIT_COMMITTER_IDENT")
    for path, expected in runner.source_hashes().items():
        if hashlib.sha256(git("show", f"HEAD:{path}")).hexdigest() != expected:
            raise RuntimeError(
                "Commit the verified implementation before scientific execution"
            )
    for name in ("protocol.json", "protocol_freeze.json"):
        path = runner.STUDY / name
        if hashlib.sha256(
            git("show", f"HEAD:{path.relative_to(ROOT)}")
        ).hexdigest() != sha(path):
            raise RuntimeError(
                "Commit the registered protocol before scientific execution"
            )


def commit_artifacts(message):
    """Only this study's nonignored artifacts; never push or include other changes."""
    with runner.exclusive():
        check_index()
        relative = str(runner.STUDY.relative_to(ROOT))
        git("add", "--", relative)
        if git("diff", "--cached", "--name-only", "--", relative).strip():
            git("commit", "--only", "-m", message, "--", relative)


def record_status(status, completed=None):
    record = {"status": status}
    if completed is not None:
        record.update(completed_rounds=completed, budget=BUDGETS[completed])
    atomic_json(runner.STUDY / "execution_status.json", record)
    detail = (
        f"Verified completed rounds: {completed}/{ROUNDS}; L{BUDGETS[completed]}."
        if completed is not None
        else "Execution started; inspect per-round complete.json for verified checkpoints."
    )
    (runner.STUDY / "STATUS.md").write_text(
        f"# {status}\n\n{detail}\n\n"
        "Registered: token4research / gpt-6-astra / high; six batches of32, stop at L525.\n"
        "This is the last recorded checkpoint, not a claim that a process is still alive.\n"
        "Resume with `.conda-hplc-al/bin/python scripts/run_hplc_fullpool_v2.py run`.\n"
        "Completed responses/fits are verified and reused; ambiguous API calls require review.\n"
        "See `execution_status.json` and `results/summary.json` when complete.\n"
    )


def run(progress=print):
    runner.STUDY.mkdir(parents=True, exist_ok=True)
    with (runner.STUDY / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another continuous V2 process is running") from None
        with runner.exclusive():
            runner.prepare()
            check_checkout()
        complete = all(
            (runner.runtime() / f"round_{r}/complete.json").exists()
            for r in range(ROUNDS)
        )
        if complete:
            result = runner.report()  # Verify all six rounds and their Git seals.
            record_status(result["status"], ROUNDS)
            commit_artifacts(f"Complete FullPool V2 six-round report at L{BUDGETS[-1]}")
            progress(result["status"], flush=True)
            return result
        if not complete:
            config = settings()
            if config != runner.EXPECTED_CONFIG:
                raise RuntimeError(
                    "configured provider/model differs from registered V2"
                )
            require_key(config)
            progress(
                "Running content-free token4research Responses preflight...", flush=True
            )
            preflight(runner.STUDY / "transport_preflight.json", config)
            record_status("IN_PROGRESS")
        for r in range(ROUNDS):
            directory = runner.runtime() / f"round_{r}"
            progress(
                f"Round {r + 1}/{ROUNDS}: L{BUDGETS[r]} -> L{BUDGETS[r + 1]}",
                flush=True,
            )
            if not (directory / "complete.json").exists():
                runner.run_selection(r)
                commit_artifacts(
                    f"Seal FullPool V2 round {r} selection before label reveal"
                )
                progress(
                    "Selection committed; revealing 32 labels and scratch training...",
                    flush=True,
                )
            runner.advance(r)  # Verifies and reuses completed fits.
            commit_artifacts(f"Record FullPool V2 L{BUDGETS[r + 1]} feedback and fit")
            record_status("IN_PROGRESS", r + 1)
        result = runner.report()
        record_status(result["status"], ROUNDS)
        commit_artifacts(f"Complete FullPool V2 six-round report at L{BUDGETS[-1]}")
        progress(result["status"], flush=True)
        return result
