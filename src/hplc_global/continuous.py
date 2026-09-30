"""User-started, bounded six-round execution with local Git seals and resume."""

import fcntl
import hashlib
import subprocess
from contextlib import contextmanager

from hplc_al.common import ROOT, atomic_json, sha
from hplc_al.llm.execution import log, preflight_with_retries, safe_error
from hplc_al.llm.full_pool import BUDGETS, ROUNDS
from hplc_al.llm.responses_transport import preflight, require_key

from . import runner


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
    path = runner.STUDY / "test_gate.json"
    if hashlib.sha256(git("show", f"HEAD:{path.relative_to(ROOT)}")).hexdigest() != sha(
        path
    ):
        raise RuntimeError(
            "Commit the V3 verification gate before scientific execution"
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
        "Resume with `.conda-hplc-al/bin/python scripts/run_hplc_global_v3.py run`.\n"
        "Completed responses/fits are verified and reused; transient requests retry at most five times with bounded backoff.\n"
        "See `execution_status.json` and `results/summary.json` when complete.\n"
    )


@contextmanager
def failure_status():
    """Record failure while the pipeline lock is still held."""
    try:
        yield
    except BaseException as error:
        status = (
            "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "STOPPED_FAILURE"
        )
        completed = 0
        for r in range(ROUNDS):
            if not (runner.runtime() / f"round_{r}/complete.json").exists():
                break
            completed += 1
        atomic_json(
            runner.STUDY / "last_failure.json",
            {
                "status": status,
                "error": safe_error(error),
                "completed_rounds": completed,
            },
        )
        record_status(status, completed)
        log(
            f"{status}: last recorded checkpoint L{BUDGETS[completed]}, {completed}/{ROUNDS} rounds; saved responses retained"
        )
        raise


def run(progress=print):
    runner.STUDY.mkdir(parents=True, exist_ok=True)
    with (runner.STUDY / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another continuous V3 process is running") from None
        with failure_status():
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
                commit_artifacts(
                    f"Complete Global V3 six-round report at L{BUDGETS[-1]}"
                )
                progress(result["status"], flush=True)
                return result
            if not complete:
                from .planner import build_request, require_capacity
                from .verification import verify_gate

                next_round = next(
                    r
                    for r in range(ROUNDS)
                    if not (runner.runtime() / f"round_{r}/complete.json").exists()
                )
                protocol, _, _, packet, _, _ = runner.make_round(next_round)
                require_capacity(build_request(**packet, limits=protocol["limits"])[3])
                verify_gate(runner.STUDY, protocol, runner.source_hashes())
                # The V3 protocol is bound to this exact provider/model. Do not
                # inherit an unrelated mutable Codex desktop configuration.
                config = dict(runner.EXPECTED_CONFIG)
                if config != runner.EXPECTED_CONFIG:
                    raise RuntimeError(
                        "configured provider/model differs from registered V3"
                    )
                require_key(config)
                progress(
                    "Running content-free token4research Responses preflight...",
                    flush=True,
                )
                preflight_with_retries(
                    lambda: preflight(runner.STUDY / "transport_preflight.json", config)
                )
                log(
                    "Preflight PASS; provider=token4research model=gpt-6-astra reasoning=high; six rounds ending at L525"
                )
                record_status("IN_PROGRESS")
            for r in range(ROUNDS):
                directory = runner.runtime() / f"round_{r}"
                progress(
                    f"Round {r + 1}/{ROUNDS}: L{BUDGETS[r]} -> L{BUDGETS[r + 1]}",
                    flush=True,
                )
                if not (directory / "complete.json").exists():
                    runner.run_selection(r)
                    log(
                        f"Round {r + 1}: committing selection and request receipts before revealing labels"
                    )
                    commit_artifacts(
                        f"Seal Global V3 round {r} selection before label reveal"
                    )
                    progress(
                        "Selection committed; revealing 32 labels and scratch training...",
                        flush=True,
                    )
                runner.advance(r)  # Verifies and reuses completed fits.
                commit_artifacts(f"Record Global V3 L{BUDGETS[r + 1]} feedback and fit")
                record_status("IN_PROGRESS", r + 1)
                log(f"Round {r + 1}/{ROUNDS} complete and committed; L{BUDGETS[r + 1]}")
            result = runner.report()
            record_status(result["status"], ROUNDS)
            commit_artifacts(f"Complete Global V3 six-round report at L{BUDGETS[-1]}")
            progress(result["status"], flush=True)
            return result
