"""BM25 memory search tool."""

from __future__ import annotations

from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.filesystem import _FsTool


class SearchMemoryTool(_FsTool):
    """Search the agent's persistent history.jsonl with BM25."""

    _scopes = {"core", "subagent", "plan", "validator"}

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Search the agent's long-term consolidated memory using BM25. "
            "Returns the most relevant historical entries matching the query. "
            "Use this to recall facts, decisions, or context from prior turns."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search for in memory",
                    "minLength": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return (default 10, max 50)",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        if not query:
            return ToolResult.error("Error: query is required.")
        workspace = self._display_workspace()
        if workspace is None:
            return ToolResult.error("Error: no workspace available.")
        store = MemoryStore(workspace)
        limit = max(1, min(50, limit))
        entries = store.search_memory(query, limit=limit)
        if not entries:
            return f"No memory entries found for query '{query}'."
        lines = []
        for entry in entries:
            cursor = entry.get("cursor", "?")
            ts = entry.get("timestamp", "")
            content = entry.get("content", "").strip()
            if content:
                lines.append(f"[{ts}] #{cursor}: {content}")
        return "\n\n".join(lines)
