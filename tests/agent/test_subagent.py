"""Tests for SubagentManager."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.filesystem import FileToolsConfig
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(provider: LLMProvider) -> LLMRuntime:
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, "test", context_window_tokens=128_000)


@pytest.mark.asyncio
async def test_subagent_uses_tool_loader():
    """Verify subagent registers tools via ToolLoader, not hard-coded imports."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=Path("/tmp"),
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    tools = sm._build_tools()
    assert tools.has("read_file")
    assert tools.has("write_file")
    assert not tools.has("message")
    assert not tools.has("spawn")


@pytest.mark.asyncio
async def test_subagent_build_tools_isolates_file_read_state(tmp_path):
    """Each spawned subagent needs a fresh file-state cache."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    first_read = sm._build_tools().get("read_file")
    second_read = sm._build_tools().get("read_file")

    assert first_read is not second_read
    assert (await first_read.execute(path="note.txt")).startswith("1| hello")
    second_result = await second_read.execute(path="note.txt")
    assert second_result.startswith("1| hello")
    assert "File unchanged" not in second_result


def test_subagent_respects_file_tool_toggle(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        tools_config=ToolsConfig(file=FileToolsConfig(enable=False)),
    )

    tools = sm._build_tools()

    file_tools = {
        "apply_patch",
        "edit_file",
        "find_files",
        "grep",
        "list_dir",
        "read_file",
        "write_file",
    }
    assert file_tools.isdisjoint(tools.tool_names)


def test_subagent_prompt_explains_grouped_skill_paths(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    global_skill = agent_workspace / "skills" / "global-custom" / "SKILL.md"
    project_skill = project / "skills" / "project-custom" / "SKILL.md"
    global_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    global_skill.write_text("---\ndescription: global skill\n---\nGlobal", encoding="utf-8")
    project_skill.write_text("---\ndescription: project skill\n---\nProject", encoding="utf-8")
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt(workspace=project)

    assert "one absolute root and relative SKILL.md paths" in prompt
    assert "Join them when using `read_file`" in prompt
    assert f"Current project workspace: {project.resolve()}" in prompt
    assert f"Nanobot's agent workspace: {agent_workspace.resolve()}" in prompt
    assert f"History log: {agent_workspace.resolve() / 'memory' / 'history.jsonl'}" in prompt
    assert "global-custom" in prompt
    assert "project-custom" not in prompt


@pytest.mark.asyncio
async def test_subagent_keeps_project_runtime_scope_with_agent_owned_tools(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    agent_workspace.mkdir()
    project.mkdir()
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    manager.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    manager._announce_result = AsyncMock()
    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
    )

    await manager._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "websocket", "chat_id": "direct"},
        status,
        _runtime(provider),
        workspace_scope=build_workspace_scope(project, "restricted"),
    )

    spec = manager.runner.run.call_args.args[0]
    assert spec.workspace == project
    assert spec.tools.get("read_file")._workspace == agent_workspace.resolve()


@pytest.mark.asyncio
async def test_subagent_forwards_fail_on_tool_error_to_runner(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        fail_on_tool_error=False,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
    )

    await sm._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "cli", "chat_id": "direct"},
        status,
        _runtime(provider),
    )

    spec = sm.runner.run.call_args.args[0]
    assert spec.fail_on_tool_error is False


@pytest.mark.asyncio
async def test_subagent_persists_and_reloads_finished_snapshot(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
        chat_id="chat-1",
    )

    await sm._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "websocket", "chat_id": "chat-1", "session_key": "websocket:chat-1"},
        status,
        _runtime(provider),
    )

    assert sm.get_status("t1") is status
    snapshot_path = tmp_path / "subagents" / "d2Vic29ja2V0OmNoYXQtMQ" / "t1.json"
    assert snapshot_path.exists()

    # Simulate restart: fresh manager should load the persisted snapshot.
    sm2 = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    restored = sm2.get_status("t1")
    assert restored is not None
    assert restored.task_id == "t1"
    assert restored.phase == "done"
    assert restored.result == "ok"
    assert restored.chat_id == "chat-1"


@pytest.mark.asyncio
async def test_subagent_expired_snapshot_is_not_reloaded(tmp_path):
    from nanobot.agent import subagent as subagent_module

    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
        chat_id="chat-1",
    )

    await sm._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "websocket", "chat_id": "chat-1", "session_key": "websocket:chat-1"},
        status,
        _runtime(provider),
    )

    snapshot_path = tmp_path / "subagents" / "d2Vic29ja2V0OmNoYXQtMQ" / "t1.json"
    assert snapshot_path.exists()

    # Patch TTL to a negative value so the persisted snapshot is expired.
    original_ttl = subagent_module.SUBAGENT_STATUS_TTL_S
    subagent_module.SUBAGENT_STATUS_TTL_S = -1.0
    try:
        sm2 = SubagentManager(
            workspace=tmp_path,
            bus=MessageBus(),
            max_tool_result_chars=16_000,
        )
        assert sm2.get_status("t1") is None
        assert not snapshot_path.exists()
    finally:
        subagent_module.SUBAGENT_STATUS_TTL_S = original_ttl


@pytest.mark.asyncio
async def test_subagent_pending_record_is_relaunched_on_resume(tmp_path):
    """A subagent that was running during shutdown gets relaunched on startup."""
    from nanobot.security.workspace_access import build_workspace_scope

    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    bus = MessageBus()

    # Create a pending record directly (simulating an unclean shutdown).
    pending_path = tmp_path / "subagents" / "d2Vic29ja2V0OmNoYXQtMQ" / "orphan.pending.json"
    pending_path.parent.mkdir(parents=True)
    scope = build_workspace_scope(tmp_path / "project", "restricted")
    pending_path.write_text(
        json.dumps(
            {
                "task_id": "orphan",
                "task": "resume me",
                "label": "Resume",
                "origin_channel": "websocket",
                "origin_chat_id": "chat-1",
                "session_key": "websocket:chat-1",
                "origin_message_id": None,
                "temperature": None,
                "workspace_scope": scope.to_dict(),
                "persisted_at": time.time(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sm = SubagentManager(
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="resumed", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    runtime = _runtime(provider)
    resumed = await sm.resume_pending(lambda _session_key: runtime)

    assert "orphan" in resumed
    # Wait for the relaunched background task to finish.
    task = sm._running_tasks.get("orphan")
    if task is not None:
        await task
    # The pending record is removed once the relaunched subagent finishes.
    assert not pending_path.exists()
    # And the finished snapshot is available.
    assert sm.get_status("orphan") is not None
    assert sm.get_status("orphan").result == "resumed"


@pytest.mark.asyncio
async def test_subagent_resume_continues_from_checkpoint_messages(tmp_path):
    """A pending record with a checkpoint resumes from the saved messages."""
    from nanobot.security.workspace_access import build_workspace_scope

    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    bus = MessageBus()

    pending_path = tmp_path / "subagents" / "d2Vic29ja2V0OmNoYXQtMQ" / "checkpointed.pending.json"
    pending_path.parent.mkdir(parents=True)
    scope = build_workspace_scope(tmp_path / "project", "restricted")
    checkpoint_messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "progress so far"},
    ]
    pending_path.write_text(
        json.dumps(
            {
                "task_id": "checkpointed",
                "task": "resume me",
                "label": "Resume",
                "origin_channel": "websocket",
                "origin_chat_id": "chat-1",
                "session_key": "websocket:chat-1",
                "origin_message_id": None,
                "temperature": None,
                "workspace_scope": scope.to_dict(),
                "persisted_at": time.time(),
                "checkpoint": {
                    "phase": "awaiting_tools",
                    "iteration": 3,
                    "messages": checkpoint_messages,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sm = SubagentManager(
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=16_000,
    )
    captured: list[dict] = []

    async def _capture_run(spec):
        captured.append({"initial_messages": spec.initial_messages})
        return AgentRunResult(
            final_content="resumed from checkpoint",
            messages=spec.initial_messages,
            stop_reason="completed",
        )

    sm.runner.run = _capture_run
    sm._announce_result = AsyncMock()

    runtime = _runtime(provider)
    resumed = await sm.resume_pending(lambda _session_key: runtime)

    assert "checkpointed" in resumed
    task = sm._running_tasks.get("checkpointed")
    if task is not None:
        await task

    assert len(captured) == 1
    assert [m.get("role") for m in captured[0]["initial_messages"][:3]] == [
        "system",
        "user",
        "assistant",
    ]
    assert captured[0]["initial_messages"][1]["content"] == "original task"
    assert captured[0]["initial_messages"][2]["content"] == "progress so far"
    # The resumed subagent refreshes the system prompt with the current template.
    assert "Subagent" in captured[0]["initial_messages"][0]["content"]
