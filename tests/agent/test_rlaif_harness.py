from nanobot.agent.rlaif.harness import PatchHarness, PatchHarnessResult


def test_harness_detects_git_support(tmp_path) -> None:
    # Non-git directory should fall back to temp-copy backend.
    harness = PatchHarness(repo_root=tmp_path)
    assert harness._use_git is False


def test_harness_result_backend_metadata() -> None:
    result = PatchHarnessResult(
        patch="p",
        summary="s",
        test_passed=True,
        lint_passed=True,
        backend="git-worktree",
    )
    assert result.backend == "git-worktree"
    assert result.passed is True
