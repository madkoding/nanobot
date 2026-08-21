"""Replay RLAIF preferences saved before the auto-apply feature existed.

For every preference in ``preferences.jsonl`` whose ``metadata.auto_apply`` is
not truthy, re-evaluate the chosen patch in a fresh git worktree, run tests
and lint, and if everything passes, commit the change to a dedicated branch
(``rlaif/replay`` by default).

Why a separate branch: never auto-merge into develop. The script just stacks
one commit per successful replay so the human can review/cherry-pick.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from nanobot.agent.rlaif.harness import PatchHarness
from nanobot.config.paths import get_runtime_subdir


DEFAULT_BRANCH = "rlaif/replay"
COMMIT_AUTHOR = "nanobot-replay"
COMMIT_MESSAGE_PREFIX = "rlaif(replay):"


def load_preferences(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def filter_unapplied(prefs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in prefs:
        md = p.get("metadata", {}) or {}
        applied = md.get("auto_apply")
        if applied is True or applied == "true":
            continue
        patch = (p.get("chosen") or {}).get("patch")
        if not patch:
            continue
        out.append(p)
    return out


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def ensure_branch(repo: Path, branch: str) -> str:
    """Make sure `branch` exists locally, based on the current HEAD of `repo`.
    Returns the base commit hash. Does NOT change the active branch of `repo`.
    """
    head = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    current = head.stdout.strip()
    if current == branch:
        return run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    exists = run(["git", "rev-parse", "--verify", branch], repo)
    if exists.returncode != 0:
        # Create branch from current HEAD without switching the main worktree.
        create = run(["git", "branch", branch], repo)
        if create.returncode != 0:
            raise RuntimeError(f"git branch {branch} failed: {create.stderr.strip()}")
    return run(["git", "rev-parse", branch], repo).stdout.strip()


def commit_patch(repo: Path, pref: dict, branch_msg: str) -> bool:
    rel_files = run(["git", "diff", "--name-only"], repo).stdout.strip()
    if not rel_files:
        return False
    add = run(["git", "add", "--"] + rel_files.splitlines(), repo)
    if add.returncode != 0:
        print(f"  git add failed: {add.stderr.strip()}")
        return False
    msg = f"{COMMIT_MESSAGE_PREFIX} {branch_msg}"
    body = (
        f"\n\nReplay of preference pair (index unknown after refactor).\n"
        f"Original task: {pref.get('task', '')[:200]}\n"
        f"Original reason: {pref.get('reason', '')[:200]}\n"
    )
    commit = run(
        [
            "git",
            "-c",
            f"user.name={COMMIT_AUTHOR}",
            "-c",
            f"user.email={COMMIT_AUTHOR}@localhost",
            "commit",
            "-m",
            msg,
            "-m",
            body,
        ],
        repo,
    )
    if commit.returncode != 0:
        print(f"  git commit failed: {commit.stderr.strip()}")
        return False
    return True


def summarize(s: str, limit: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


async def replay_one(
    pref: dict,
    *,
    repo_root: Path,
    index: int,
    total: int,
    test_command: list[str] | None,
    lint_command: list[str] | None,
) -> tuple[str, str]:
    """Returns (status, detail). status in {applied, skip, fail}."""
    patch = (pref.get("chosen") or {}).get("patch") or ""
    task = pref.get("task", "")
    label = summarize(task, 60)
    print(f"[{index}/{total}] {label}")
    if not patch.strip():
        print("  no patch text, skipping")
        return "skip", "no patch"

    # Reuse PatchHarness so we get the same git-worktree + tests+lint logic
    # the live observer uses. PatchHarness cleans up its temp worktree on exit.
    harness = PatchHarness(
        repo_root=repo_root,
        test_command=test_command,
        lint_command=lint_command,
    )
    result = await harness.evaluate(patch, patch_summary=label)

    if not result.passed:
        why = []
        if not result.test_passed:
            why.append("tests")
        if not result.lint_passed:
            why.append("lint")
        print(f"  skip ({', '.join(why) or 'checks'} failed)")
        return "skip", f"{', '.join(why) or 'checks'} failed"

    # PatchHarness already applied the patch in its temp worktree, but
    # discarded it. Apply again to a persistent worktree on the replay
    # branch so the change gets committed.
    return "needs_commit", "tests+lint passed"


def apply_and_commit(
    pref: dict,
    *,
    repo_root: Path,
    branch: str,
    test_command: list[str] | None,
    lint_command: list[str] | None,
) -> str:
    """Apply the patch in a long-lived worktree on `branch`, run checks, commit if pass."""
    patch = (pref.get("chosen") or {}).get("patch") or ""
    task = pref.get("task", "")
    label = summarize(task, 60)

    # Make sure branch exists, without moving the main worktree.
    base_commit = ensure_branch(repo_root, branch)

    worktree_parent = repo_root.parent / f"{repo_root.name}.rlaif-replay"
    worktree_parent.mkdir(exist_ok=True)
    wt_path = worktree_parent / f"wt-{int(time.time() * 1000)}"

    add_wt = run(
        ["git", "worktree", "add", str(wt_path), branch],
        repo_root,
    )
    if add_wt.returncode != 0:
        return f"fail: worktree add failed: {add_wt.stderr.strip()}"

    try:
        # Apply patch
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        ) as f:
            f.write(patch)
            patch_path = f.name
        applied = run(["git", "apply", patch_path], wt_path)
        if applied.returncode != 0:
            return f"skip: patch does not apply cleanly: {applied.stderr.strip()}"

        # Run tests
        test_cmd = test_command or ["python", "-m", "pytest", "-q", "tests/agent/test_rlaif_*.py"]
        t = run(test_cmd, wt_path)
        test_ok = t.returncode == 0

        # Run lint
        lint_cmd = lint_command or ["python", "-m", "ruff", "check", "nanobot/agent/rlaif/"]
        l = run(lint_cmd, wt_path)
        lint_ok = l.returncode == 0

        if not (test_ok and lint_ok):
            reasons = []
            if not test_ok:
                reasons.append("tests")
            if not lint_ok:
                reasons.append("lint")
            return f"skip: {', '.join(reasons)} failed (head: t={t.returncode} l={l.returncode})"

        # Commit
        rel = run(["git", "diff", "--name-only"], wt_path).stdout.strip()
        if not rel:
            return "skip: no file changes after apply"
        run(["git", "add", "--"] + rel.splitlines(), wt_path)
        msg = f"{COMMIT_MESSAGE_PREFIX} {label}"
        body = (
            f"\n\nReplayed from preferences.jsonl.\n"
            f"Original task: {task[:200]}\n"
            f"Original reason: {pref.get('reason', '')[:200]}\n"
        )
        commit = run(
            [
                "git",
                "-c",
                f"user.name={COMMIT_AUTHOR}",
                "-c",
                f"user.email={COMMIT_AUTHOR}@localhost",
                "commit",
                "-m",
                msg,
                "-m",
                body,
            ],
            wt_path,
        )
        if commit.returncode != 0:
            return f"fail: commit failed: {commit.stderr.strip()}"
        return "applied"
    finally:
        run(["git", "worktree", "remove", "--force", str(wt_path)], repo_root)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("/home/madkoding/.nanobot/workspace/proyects/nanobot"),
        help="Path to the nanobot git repo",
    )
    ap.add_argument("--branch", default=DEFAULT_BRANCH, help="Replay branch name")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to preferences.jsonl (default: runtime/rlaif/preferences.jsonl)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Replay at most N preferences (after filtering unapplied)",
    )
    ap.add_argument(
        "--tests",
        default=None,
        help="Override test command (default: pytest tests/agent/test_rlaif_*.py)",
    )
    ap.add_argument(
        "--lint",
        default=None,
        help="Override lint command (default: ruff check nanobot/agent/rlaif/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run tests+lint but do not commit anything",
    )
    args = ap.parse_args()

    dataset = args.dataset or (get_runtime_subdir("rlaif") / "preferences.jsonl")
    test_cmd = args.tests.split() if args.tests else None
    lint_cmd = args.lint.split() if args.lint else None

    prefs = load_preferences(dataset)
    todo = filter_unapplied(prefs)
    if args.limit:
        todo = todo[: args.limit]
    print(f"Total prefs: {len(prefs)}; unapplied: {len(filter_unapplied(prefs))}; replaying: {len(todo)}")
    if not todo:
        return 0

    if args.dry_run:
        # Just exercise the harness without touching the repo
        for i, p in enumerate(todo, 1):
            status, detail = await replay_one(
                p,
                repo_root=args.repo,
                index=i,
                total=len(todo),
                test_command=test_cmd,
                lint_command=lint_cmd,
            )
            print(f"  -> {status}: {detail}")
        return 0

    counts = {"applied": 0, "skip": 0, "fail": 0}
    for i, p in enumerate(todo, 1):
        result = apply_and_commit(
            p,
            repo_root=args.repo,
            branch=args.branch,
            test_command=test_cmd,
            lint_command=lint_cmd,
        )
        kind = result.split(":", 1)[0]
        if kind in counts:
            counts[kind] += 1
        else:
            counts["fail"] += 1
        print(f"  -> {result}")

    print()
    print(f"Summary: applied={counts['applied']}, skip={counts['skip']}, fail={counts['fail']}")
    print(f"Replay branch: {args.branch}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
