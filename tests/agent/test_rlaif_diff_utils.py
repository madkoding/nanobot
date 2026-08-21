"""Tests for RLAIF diff utilities."""


from nanobot.agent.rlaif.diff_utils import (
    extract_unified_diff,
    is_valid_unified_diff,
    summarize_unified_diff,
)


class TestExtractUnifiedDiff:
    def test_returns_bare_diff(self):
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
        )
        assert extract_unified_diff(diff) == diff.rstrip("\n")

    def test_strips_markdown_fence(self):
        diff = (
            "```diff\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        result = extract_unified_diff(diff)
        assert result.startswith("--- a/foo.py")
        assert "```" not in result

    def test_extracts_from_extra_text(self):
        diff = (
            "Here is the fix:\n\n"
            "```diff\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
            "```\n"
            "Hope it helps!"
        )
        result = extract_unified_diff(diff)
        assert "--- a/foo.py" in result
        assert "Hope it helps!" not in result


class TestIsValidUnifiedDiff:
    def test_valid(self):
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
        )
        assert is_valid_unified_diff(diff) is True

    def test_missing_hunk(self):
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "-old\n"
            "+new\n"
        )
        assert is_valid_unified_diff(diff) is False


class TestSummarizeUnifiedDiff:
    def test_single_file(self):
        diff = "+++ b/foo.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        assert summarize_unified_diff(diff) == "patch touching b/foo.py"

    def test_multiple_files(self):
        diff = (
            "+++ b/foo.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
            "+++ b/bar.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        assert summarize_unified_diff(diff) == "patch touching b/foo.py, b/bar.py"
