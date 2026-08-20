from nanobot.agent.rlaif.diff_utils import (
    extract_unified_diff,
    is_valid_unified_diff,
    summarize_unified_diff,
)


def test_extract_unified_diff_bare() -> None:
    text = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old\n"
        "+new\n"
    )
    assert extract_unified_diff(text) == text.rstrip("\n")


def test_extract_unified_diff_markdown_fence() -> None:
    text = (
        "Here is the patch:\n\n"
        "```diff\n"
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old\n"
        "+new\n"
        "```"
    )
    patch = extract_unified_diff(text)
    assert patch.startswith("--- a/file.py")
    assert "```" not in patch


def test_extract_unified_diff_with_prefix() -> None:
    text = (
        "Sure, here you go:\n"
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    patch = extract_unified_diff(text)
    assert patch.startswith("--- a/file.py")


def test_is_valid_unified_diff_true() -> None:
    patch = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old\n"
        "+new\n"
    )
    assert is_valid_unified_diff(patch) is True


def test_is_valid_unified_diff_missing_hunk() -> None:
    patch = "--- a/file.py\n+++ b/file.py\n-old\n+new\n"
    assert is_valid_unified_diff(patch) is False


def test_summarize_unified_diff() -> None:
    patch = "--- a/foo.py\n+++ b/foo.py\n--- a/bar.py\n+++ b/bar.py\n"
    assert summarize_unified_diff(patch) == "patch touching b/foo.py, b/bar.py"


def test_summarize_unified_diff_many() -> None:
    patch = "\n".join(
        f"+++ b/f{i}.py" for i in range(5)
    )
    summary = summarize_unified_diff(patch)
    assert "5 files" in summary
