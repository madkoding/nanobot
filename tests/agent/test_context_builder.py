"""Tests for ContextBuilder — system prompt and message assembly."""

from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.runtime_context import RuntimeContextBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _builder(tmp_path: Path, **kw) -> ContextBuilder:
    return ContextBuilder(workspace=tmp_path, **kw)


# ---------------------------------------------------------------------------
# _merge_message_content (static)
# ---------------------------------------------------------------------------


class TestMergeMessageContent:
    def test_str_plus_str(self):
        result = ContextBuilder._merge_message_content("hello", "world")
        assert result == "hello\n\nworld"

    def test_empty_left_plus_str(self):
        result = ContextBuilder._merge_message_content("", "world")
        assert result == "world"

    def test_list_plus_list(self):
        left = [{"type": "text", "text": "a"}]
        right = [{"type": "text", "text": "b"}]
        result = ContextBuilder._merge_message_content(left, right)
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "b"

    def test_str_plus_list(self):
        right = [{"type": "text", "text": "b"}]
        result = ContextBuilder._merge_message_content("hello", right)
        assert len(result) == 2
        assert result[0]["text"] == "hello"
        assert result[1]["text"] == "b"

    def test_list_plus_str(self):
        left = [{"type": "text", "text": "a"}]
        result = ContextBuilder._merge_message_content(left, "world")
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "world"

    def test_none_plus_str(self):
        result = ContextBuilder._merge_message_content(None, "hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_str_plus_none(self):
        result = ContextBuilder._merge_message_content("hello", None)
        assert result == [{"type": "text", "text": "hello"}]

    def test_none_plus_none(self):
        result = ContextBuilder._merge_message_content(None, None)
        assert result == []

    def test_list_items_not_dicts_wrapped(self):
        result = ContextBuilder._merge_message_content(["raw_item"], None)
        assert result == [{"type": "text", "text": "raw_item"}]


# ---------------------------------------------------------------------------
# _load_bootstrap_files
# ---------------------------------------------------------------------------


class TestLoadBootstrapFiles:
    def test_no_bootstrap_files(self, tmp_path):
        builder = _builder(tmp_path)
        assert builder._load_bootstrap_files() == ""

    def test_empty_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("\n", encoding="utf-8")
        builder = _builder(tmp_path)
        assert builder._load_bootstrap_files() == ""

    def test_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be helpful.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "## AGENTS.md" in result
        assert "Be helpful." in result

    def test_multiple_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Rules.", encoding="utf-8")
        (tmp_path / "SOUL.md").write_text("Soul.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "## AGENTS.md" in result
        assert "## SOUL.md" in result
        assert "Rules." in result
        assert "Soul." in result

    def test_all_bootstrap_files(self, tmp_path):
        for name in ContextBuilder.BOOTSTRAP_FILES:
            (tmp_path / name).write_text(f"Content of {name}", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        for name in ContextBuilder.BOOTSTRAP_FILES:
            assert f"## {name}" in result

    def test_legacy_tools_md_is_not_bootstrapped(self, tmp_path):
        (tmp_path / "TOOLS.md").write_text("workspace tool notes", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "TOOLS.md" not in result
        assert "workspace tool notes" not in result

    def test_utf8_content(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("用中文回复", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._load_bootstrap_files()
        assert "用中文回复" in result

    def test_selected_project_supplies_only_agents_file(self, tmp_path):
        agent_home = tmp_path / "agent-home"
        project = tmp_path / "project"
        agent_home.mkdir()
        project.mkdir()
        (agent_home / "AGENTS.md").write_text("global project rules", encoding="utf-8")
        (agent_home / "SOUL.md").write_text("global soul", encoding="utf-8")
        (agent_home / "USER.md").write_text("global user", encoding="utf-8")
        (project / "AGENTS.md").write_text("selected project rules", encoding="utf-8")
        (project / "SOUL.md").write_text("project soul collision", encoding="utf-8")
        (project / "USER.md").write_text("project user collision", encoding="utf-8")

        result = ContextBuilder(agent_home).build_system_prompt(
            workspace=project,
            include_memory_recent_history=False,
        )

        assert "selected project rules" in result
        assert "global project rules" not in result
        assert "global soul" in result
        assert "global user" in result
        assert "project soul collision" not in result
        assert "project user collision" not in result

    def test_selected_project_without_agents_does_not_fall_back(self, tmp_path):
        agent_home = tmp_path / "agent-home"
        project = tmp_path / "project"
        agent_home.mkdir()
        project.mkdir()
        (agent_home / "AGENTS.md").write_text("default workspace rules", encoding="utf-8")

        result = ContextBuilder(agent_home).build_system_prompt(
            workspace=project,
            include_memory_recent_history=False,
        )

        assert "default workspace rules" not in result

    def test_unmodified_agents_and_user_templates_are_skipped(self, tmp_path):
        from nanobot.utils.helpers import sync_workspace_templates

        sync_workspace_templates(tmp_path, silent=True)

        result = ContextBuilder(tmp_path)._load_bootstrap_files()

        assert "## AGENTS.md" not in result
        assert "## USER.md" not in result
        assert "## SOUL.md" in result

    def test_customized_user_template_is_loaded(self, tmp_path):
        from nanobot.utils.helpers import sync_workspace_templates

        sync_workspace_templates(tmp_path, silent=True)
        (tmp_path / "USER.md").write_text("User prefers Chinese.", encoding="utf-8")

        result = ContextBuilder(tmp_path)._load_bootstrap_files()

        assert "## USER.md" in result
        assert "User prefers Chinese." in result


# ---------------------------------------------------------------------------
# _is_template_content (static)
# ---------------------------------------------------------------------------


class TestIsTemplateContent:
    def test_nonexistent_template_returns_false(self):
        assert ContextBuilder._is_template_content("anything", "nonexistent/path.md") is False

    def test_content_matching_template(self):
        from importlib.resources import files as pkg_files

        tpl = pkg_files("nanobot") / "templates" / "memory" / "MEMORY.md"
        if not tpl.is_file():
            pytest.skip("MEMORY.md template not bundled")
        original = tpl.read_text(encoding="utf-8")
        assert ContextBuilder._is_template_content(original, "memory/MEMORY.md") is True

    def test_modified_content_returns_false(self):
        from importlib.resources import files as pkg_files

        tpl = pkg_files("nanobot") / "templates" / "memory" / "MEMORY.md"
        if not tpl.is_file():
            pytest.skip("MEMORY.md template not bundled")
        assert ContextBuilder._is_template_content("totally different", "memory/MEMORY.md") is False


# ---------------------------------------------------------------------------
# Bundled bootstrap templates
# ---------------------------------------------------------------------------


class TestBundledToolContract:
    def test_tool_contract_balances_general_and_coding_workflows(self):
        from importlib.resources import files as pkg_files

        tpl = pkg_files("nanobot") / "templates" / "agent" / "tool_contract.md"
        content = tpl.read_text(encoding="utf-8")

        assert "## General Tool Contract" in content
        assert "Use the narrowest structured tool" in content
        assert "Do not use `exec` as a universal workaround" in content
        assert "## File and Coding Workflows" in content
        assert "apply_patch" in content
        assert "acceptance criteria into concrete checks" in content
        assert "visual evidence reaches the model" in content
        assert "clear user request as authorization" in content
        assert "Never invent missing records or measurements" in content
        assert "scientific fitting" not in content
        assert "## Web and External Information" in content
        assert "## Messaging and Media" in content
        assert "## Scheduling and Background Work" in content
        assert "pure coding" not in content.lower()

    def test_tool_contract_is_injected_without_workspace_file(self, tmp_path):
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt()

        assert "# Tool Usage Notes" in prompt
        assert "## General Tool Contract" in prompt
        assert "Do not use `exec` as a universal workaround" in prompt


# ---------------------------------------------------------------------------
# _build_user_content
# ---------------------------------------------------------------------------


class TestBuildUserContent:
    def test_no_media_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", None)
        assert result == "hello"

    def test_empty_media_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [])
        assert result == "hello"

    def test_nonexistent_media_file_returns_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", ["/nonexistent/image.png"])
        assert result == "hello"

    def test_non_image_file_returns_string(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("not an image", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [str(txt)])
        assert result == "hello"

    def test_valid_image_returns_list(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [str(png)])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "hello"

    def test_image_meta_includes_path(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        result = builder._build_user_content("hello", [str(png)])
        assert "_meta" in result[0]
        assert "path" in result[0]["_meta"]


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_returns_nonempty_string(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_identity_section(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "workspace" in result.lower() or "python" in result.lower()

    def test_selected_project_identity_keeps_agent_data_in_agent_workspace(self, tmp_path):
        agent_home = tmp_path / "agent-home"
        project = tmp_path / "project"
        agent_home.mkdir()
        project.mkdir()

        result = ContextBuilder(agent_home)._get_identity(workspace=project)

        assert f"current project workspace is at: {project.resolve()}" in result
        assert f"agent workspace is at: {agent_home.resolve()}" in result
        assert f"{agent_home.resolve()}/SOUL.md" in result
        assert f"{project.resolve()}/SOUL.md" not in result

    def test_includes_bootstrap_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be helpful and concise.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "Be helpful and concise." in result

    def test_includes_session_summary(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt(session_summary="Previous chat about Python.")
        assert "Previous chat about Python." in result
        assert "[Archived Context Summary]" in result

    def test_sections_separated_by_separator(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Rules.", encoding="utf-8")
        builder = _builder(tmp_path)
        result = builder.build_system_prompt(session_summary="Summary.")
        assert "\n\n---\n\n" in result

    def test_no_bootstrap_no_summary(self, tmp_path):
        builder = _builder(tmp_path)
        result = builder.build_system_prompt()
        assert "## AGENTS.md" not in result
        assert "[Archived Context Summary]" not in result


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_basic_empty_history(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "hello" in str(messages[1]["content"])

    def test_runtime_context_is_not_injected_by_default(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hello", channel="cli", chat_id="direct")
        user_msg = str(messages[-1]["content"])
        assert user_msg == "hello"

    def test_explicit_runtime_context_blocks_are_appended(self, tmp_path):
        builder = _builder(tmp_path)
        messages = builder.build_messages(
            [],
            "please use @zoom tonight",
            runtime_context_blocks=[
                RuntimeContextBlock(
                    source="cli_apps",
                    content="CLI App Attachment: @zoom (installed; tool=run_cli_app).",
                ),
            ],
        )
        user_msg = str(messages[-1]["content"])

        assert "CLI App Attachment: @zoom" in user_msg
        assert "tool=run_cli_app" in user_msg
        assert user_msg.index("please use @zoom tonight") < user_msg.index(
            "CLI App Attachment: @zoom"
        )
        assert messages[-1]["_meta"]["runtime_context"]["sources"] == ["cli_apps"]

    def test_consecutive_same_role_merged(self, tmp_path):
        builder = _builder(tmp_path)
        history = [{"role": "user", "content": "previous user message"}]
        messages = builder.build_messages(history, "new message")
        assert len(messages) == 2  # system + merged user
        assert "previous user message" in str(messages[1]["content"])
        assert "new message" in str(messages[1]["content"])

    def test_different_role_appended(self, tmp_path):
        builder = _builder(tmp_path)
        history = [{"role": "assistant", "content": "previous response"}]
        messages = builder.build_messages(history, "new message")
        assert len(messages) == 3  # system + assistant + user

    def test_media_with_history(self, tmp_path):
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        builder = _builder(tmp_path)
        history = [{"role": "assistant", "content": "see this"}]
        messages = builder.build_messages(history, "check image", media=[str(png)])
        user_msg = messages[-1]["content"]
        assert isinstance(user_msg, list)
        assert any(b.get("type") == "image_url" for b in user_msg)


# ---------------------------------------------------------------------------
# extra_bootstrap_paths
# ---------------------------------------------------------------------------


class TestExtraBootstrapPaths:
    def test_no_paths_returns_no_override_section(self, tmp_path):
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt(workspace=tmp_path)
        assert "Workspace Overrides" not in prompt

    def test_empty_paths_returns_no_override_section(self, tmp_path):
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[])
        assert "Workspace Overrides" not in prompt

    def test_missing_files_returns_no_override_section(self, tmp_path):
        # Workspace exists but has no AGENTS.md / SOUL.md.
        empty_ws = tmp_path / "empty_group_ws"
        empty_ws.mkdir()
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[empty_ws])
        assert "Workspace Overrides" not in prompt

    def test_agents_md_loaded_into_override_section(self, tmp_path):
        group_ws = tmp_path / "grupo_dev"
        group_ws.mkdir()
        (group_ws / "AGENTS.md").write_text(
            "Solo programación. Respuestas breves. No código listo.", encoding="utf-8"
        )
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[group_ws])
        assert "Workspace Overrides" in prompt
        assert "Solo programación" in prompt
        assert "grupo_dev" in prompt

    def test_soul_md_loaded_alongside_agents_md(self, tmp_path):
        group_ws = tmp_path / "grupo_soul"
        group_ws.mkdir()
        (group_ws / "AGENTS.md").write_text("rules here", encoding="utf-8")
        (group_ws / "SOUL.md").write_text("tone here", encoding="utf-8")
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[group_ws])
        assert "rules here" in prompt
        assert "tone here" in prompt

    def test_unreadable_file_is_skipped(self, tmp_path):
        group_ws = tmp_path / "grupo_bad"
        group_ws.mkdir()
        (group_ws / "AGENTS.md").write_text("ok", encoding="utf-8")
        # A file with invalid UTF-8 bytes — must not raise.
        (group_ws / "SOUL.md").write_bytes(b"\xff\xfe\x00bad")
        builder = _builder(tmp_path)
        prompt = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[group_ws])
        assert "ok" in prompt
        assert "Workspace Overrides" in prompt

    def test_cache_key_differs_when_paths_differ(self, tmp_path):
        group_ws = tmp_path / "grupo_a"
        group_ws.mkdir()
        (group_ws / "AGENTS.md").write_text("group A rules", encoding="utf-8")
        builder = _builder(tmp_path)
        first = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[group_ws])
        # Different group → different prompt, no stale cache hit.
        other_ws = tmp_path / "grupo_b"
        other_ws.mkdir()
        (other_ws / "AGENTS.md").write_text("group B rules", encoding="utf-8")
        second = builder.build_system_prompt(workspace=tmp_path, extra_bootstrap_paths=[other_ws])
        assert "group A rules" in first
        assert "group B rules" in second
        assert first != second

    def test_build_messages_propagates_extra_bootstrap_paths(self, tmp_path):
        group_ws = tmp_path / "grupo_msg"
        group_ws.mkdir()
        (group_ws / "AGENTS.md").write_text("propagated", encoding="utf-8")
        builder = _builder(tmp_path)
        messages = builder.build_messages([], "hi", extra_bootstrap_paths=[group_ws])
        system_msg = messages[0]["content"]
        assert "propagated" in system_msg
        assert "Workspace Overrides" in system_msg
