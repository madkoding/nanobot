"""Tests for the workflow engine, loader, tool, and /workflow command."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.workflow import RunWorkflowTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import cmd_workflow, register_builtin_commands
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.config.schema import ModelPresetConfig
from nanobot.security.workspace_access import WorkspaceScope
from nanobot.workflows import WorkflowLoader, WorkflowRunner, parse_workflow_args


def _provider(default_model: str = "test-model") -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = MagicMock()
    provider.generation.max_tokens = 4096
    provider.generation.temperature = 0.1
    provider.generation.reasoning_effort = None
    return provider


def _make_loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=8000,
        model_presets={
            "default": ModelPresetConfig(
                model="test-model",
                max_tokens=4096,
                context_window_tokens=8000,
            ),
        },
    )


def _write_workflow(base: Path, name: str, source: str) -> Path:
    path = base / "workflows" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


DEMO_SOURCE = """
from nanobot.workflows.runner import AgentResult

async def run(args, ctx):
    ctx.set_phase("one")
    a = await ctx.agent(agent="build", prompt="step one")
    ctx.set_phase("two")
    b, c = await ctx.parallel([
        lambda: ctx.agent(agent="plan", prompt="p1"),
        lambda: ctx.agent(agent="plan", prompt="p2"),
    ])
    d = await ctx.pipeline([
        lambda prev: ctx.agent(agent="gen", prompt=f"pipe {prev}"),
    ], initial="start")
    return AgentResult(text="|".join([a.text, b.text, c.text, d.text]))
"""


def _runner(tmp_path: Path) -> WorkflowRunner:
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    subagents = MagicMock()
    subagents.run_inline = AsyncMock(side_effect=lambda task, **kw: f"[{task}]")
    empty_builtin = tmp_path / "empty_builtin"
    empty_builtin.mkdir(exist_ok=True)
    loader = WorkflowLoader(tmp_path, builtin_workflows_dir=empty_builtin)
    return WorkflowRunner(
        subagents=subagents,
        bus=bus,
        loader=loader,
        workspace=tmp_path,
    )


def _runner_with_builtins(tmp_path: Path) -> WorkflowRunner:
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    subagents = MagicMock()
    subagents.run_inline = AsyncMock(side_effect=lambda task, **kw: f"[{task}]")
    loader = WorkflowLoader(tmp_path)
    return WorkflowRunner(
        subagents=subagents,
        bus=bus,
        loader=loader,
        workspace=tmp_path,
    )


async def _wait_done(runner: WorkflowRunner, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while runner._running and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.01)
    assert not runner._running, "workflow run did not complete"


def test_parse_workflow_args() -> None:
    assert parse_workflow_args("") == {}
    assert parse_workflow_args("topic=Rust async internals") == {"topic": "Rust async internals"}
    assert parse_workflow_args("topic=Rust async internals verbose=true") == {
        "topic": "Rust async internals",
        "verbose": "true",
    }
    assert parse_workflow_args("bare") == {}


def test_loader_lists_builtin_and_workspace(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "demo", DEMO_SOURCE)
    empty_builtin = tmp_path / "empty_builtin"
    empty_builtin.mkdir(exist_ok=True)
    loader = WorkflowLoader(tmp_path, builtin_workflows_dir=empty_builtin)
    names = [entry["name"] for entry in loader.list_workflows()]
    assert names == ["demo"]

    loader_full = WorkflowLoader(tmp_path)
    names_full = [entry["name"] for entry in loader_full.list_workflows()]
    assert "demo" in names_full
    assert "research_plan" in names_full


def test_loader_disabled_and_override(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "demo", DEMO_SOURCE)
    empty_builtin = tmp_path / "empty_builtin"
    empty_builtin.mkdir(exist_ok=True)
    loader = WorkflowLoader(
        tmp_path, builtin_workflows_dir=empty_builtin, disabled_workflows={"demo"}
    )
    assert loader.list_workflows() == []

    # Workspace workflow shadows builtin of the same name.
    loader2 = WorkflowLoader(tmp_path)
    assert asyncio.iscoroutinefunction(loader2.load("research_plan").run)
    ws_plan = _write_workflow(
        tmp_path,
        "research_plan",
        "async def run(args, ctx):\n    return 'shadowed'",
    )
    del ws_plan
    loader3 = WorkflowLoader(tmp_path)
    shadowed = loader3.load("research_plan")
    assert asyncio.iscoroutinefunction(shadowed.run)


def test_loader_skips_invalid_module(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "broken", "def run():\n    pass\n")
    _write_workflow(tmp_path, "boom", "raise RuntimeError('nope')")
    empty_builtin = tmp_path / "empty_builtin"
    empty_builtin.mkdir(exist_ok=True)
    loader = WorkflowLoader(tmp_path, builtin_workflows_dir=empty_builtin)
    assert loader.load("broken") is None
    assert loader.load("boom") is None


def test_builtin_workflows_load(tmp_path: Path) -> None:
    loader = WorkflowLoader(tmp_path / "missing")
    names = [entry["name"] for entry in loader.list_workflows()]
    assert names == [
        "code_review",
        "debug_issue",
        "feature_plan",
        "generate_tests",
        "research_plan",
    ]
    for name in names:
        module = loader.load(name)
        assert module is not None
        assert asyncio.iscoroutinefunction(module.run)
        assert module.ARGUMENTS
        assert module.PHASES


@pytest.mark.asyncio
async def test_runner_executes_phases_and_announces(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "demo", DEMO_SOURCE)
    runner = _runner(tmp_path)
    scope = MagicMock(spec=WorkspaceScope)
    runtime = MagicMock()

    run_id = await runner.start(
        name="demo",
        args={"topic": "x"},
        runtime=runtime,
        session_key="cli:direct",
        channel="cli",
        chat_id="direct",
        workspace_scope=scope,
    )
    assert run_id
    await _wait_done(runner)

    bus = runner.bus
    assert bus.publish_inbound.called
    msg = bus.publish_inbound.await_args.args[0]
    assert isinstance(msg, InboundMessage)
    assert msg.sender_id == "workflow"
    assert msg.session_key_override == "cli:direct"
    assert msg.metadata["injected_event"] == "workflow_result"
    assert msg.metadata["workflow_run_id"] == run_id
    assert "Workflow `demo` completed successfully" in msg.content
    assert "[step one]|[p1]|[p2]|[pipe start]" in msg.content

    history = json.loads((tmp_path / "workflow_history" / f"{run_id}.json").read_text())
    assert history["status"] == "completed"
    assert [p["name"] for p in history["phases"]] == ["one", "two"]
    assert history["args"] == {"topic": "x"}


@pytest.mark.asyncio
async def test_runner_announces_failure(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "boom", "async def run(args, ctx):\n    raise ValueError('kaput')")
    runner = _runner(tmp_path)
    runtime = MagicMock()

    await runner.start(
        name="boom",
        runtime=runtime,
        session_key="cli:direct",
        channel="cli",
        chat_id="direct",
    )
    await _wait_done(runner)

    msg = runner.bus.publish_inbound.await_args.args[0]
    assert "failed (failed)" in msg.content
    assert "kaput" in msg.content
    history = next((tmp_path / "workflow_history").glob("*.json"))
    assert json.loads(history.read_text())["status"] == "failed"


@pytest.mark.asyncio
async def test_cancel_by_session(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "slow",
        "async def run(args, ctx):\n    import asyncio\n    await asyncio.sleep(3600)",
    )
    runner = _runner(tmp_path)
    runtime = MagicMock()

    await runner.start(
        name="slow",
        runtime=runtime,
        session_key="cli:direct",
        channel="cli",
        chat_id="direct",
    )
    assert len(runner._running) == 1
    cancelled = await runner.cancel_by_session("cli:direct")
    assert cancelled == 1
    assert not runner._running


@pytest.mark.asyncio
async def test_run_workflow_tool(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "demo", DEMO_SOURCE)
    runner = _runner(tmp_path)
    tool = RunWorkflowTool(runner=runner)

    request = RequestContext(
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        runtime=MagicMock(),
        message_id="m1",
    )
    token = bind_request_context(request)
    try:
        result = await tool.execute(workflow="demo", args="topic=y")
    finally:
        reset_request_context(token)
    assert isinstance(result, str) and result.startswith("Workflow 'demo' started (run: ")
    await _wait_done(runner)
    assert runner.bus.publish_inbound.called


@pytest.mark.asyncio
async def test_run_workflow_tool_requires_runtime(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tool = RunWorkflowTool(runner=runner)
    request = RequestContext(channel="cli", chat_id="direct", session_key="cli:direct")
    token = bind_request_context(request)
    try:
        result = await tool.execute(workflow="demo")
    finally:
        reset_request_context(token)
    assert isinstance(result, ToolResult) and result.is_error


@pytest.mark.asyncio
async def test_cmd_workflow_lists(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    (tmp_path / "workflows").mkdir(exist_ok=True)
    (tmp_path / "workflows" / "demo.py").write_text(DEMO_SOURCE, encoding="utf-8")
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/workflow")
    ctx = CommandContext(
        msg=msg, session=None, key=msg.session_key, raw="/workflow", args="", loop=loop
    )
    out = await cmd_workflow(ctx)
    assert isinstance(out, OutboundMessage)
    assert "**demo**" in out.content


@pytest.mark.asyncio
async def test_workflow_command_registered_on_router(tmp_path: Path) -> None:
    router = CommandRouter()
    register_builtin_commands(router)
    loop = _make_loop(tmp_path)
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/workflow")
    ctx = CommandContext(
        msg=msg, session=None, key=msg.session_key, raw="/workflow", args="", loop=loop
    )
    out = await router.dispatch(ctx)
    assert out is not None
    assert "Available workflows" in out.content or "No workflows available." in out.content


async def _run_builtin(tmp_path: Path, name: str, args: dict[str, str]) -> str:
    """Run a builtin workflow to completion and return the announced result text."""
    runner = _runner_with_builtins(tmp_path)
    runtime = MagicMock()
    await runner.start(
        name=name,
        args=args,
        runtime=runtime,
        session_key="cli:direct",
        channel="cli",
        chat_id="direct",
    )
    await _wait_done(runner)
    msg = runner.bus.publish_inbound.await_args.args[0]
    return msg.content


@pytest.mark.asyncio
async def test_code_review_workflow_runs_parallel_reviews(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "code_review", {"range": "HEAD~1..HEAD"})
    assert "Workflow `code_review` completed successfully" in content
    assert "CORRECTNESS:" in content
    assert "SECURITY:" in content
    assert "STYLE:" in content


@pytest.mark.asyncio
async def test_debug_issue_workflow_runs_phases(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "debug_issue", {"symptom": "login fails"})
    assert "Workflow `debug_issue` completed successfully" in content


@pytest.mark.asyncio
async def test_debug_issue_workflow_requires_symptom(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "debug_issue", {})
    assert "No symptom provided" in content


@pytest.mark.asyncio
async def test_feature_plan_workflow_runs_phases(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "feature_plan", {"feature": "add dark mode"})
    assert "Workflow `feature_plan` completed successfully" in content


@pytest.mark.asyncio
async def test_feature_plan_workflow_requires_feature(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "feature_plan", {})
    assert "No feature provided" in content


@pytest.mark.asyncio
async def test_generate_tests_workflow_runs_phases(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "generate_tests", {"path": "nanobot/utils/helpers.py"})
    assert "Workflow `generate_tests` completed successfully" in content


@pytest.mark.asyncio
async def test_generate_tests_workflow_requires_path(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "generate_tests", {})
    assert "No path provided" in content


@pytest.mark.asyncio
async def test_research_plan_workflow_runs_parallel(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "research_plan", {"topic": "async internals"})
    assert "Workflow `research_plan` completed successfully" in content
    assert "BRIEF:" in content
    assert "RISKS:" in content
    assert "IMPLEMENTATION:" in content


@pytest.mark.asyncio
async def test_research_plan_workflow_requires_topic(tmp_path: Path) -> None:
    content = await _run_builtin(tmp_path, "research_plan", {})
    assert "No topic provided" in content
