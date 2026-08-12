"""Agenda tool for managing calendar appointments."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.schema import (
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import default_workspace_scope
from nanobot.webui.agenda_api import (
    create_appointment,
    delete_appointment,
    list_appointments,
    update_appointment,
)

_AGENDA_PARAMETERS = tool_parameters_schema(
    action=StringSchema("Action", enum=["add", "list", "update", "delete"]),
    id=StringSchema("Appointment ID (update/delete)."),
    title=StringSchema("Appointment title (add)."),
    date=StringSchema("Date YYYY-MM-DD (add)."),
    time=StringSchema("Time HH:MM or null for all-day."),
    all_day=StringSchema("All-day flag: 'true'/'false'.", enum=["true", "false"]),
    notes=StringSchema("Description/notes."),
    category=StringSchema(
        "Category.", enum=["personal", "work", "health", "reminder", "journal", "other"]
    ),
    color=StringSchema("Hex color (e.g. #ef4444). Defaults to category color."),
    required=["action"],
    description="Manage appointments. add requires title+date; update/delete require id.",
)


def _parse_all_day(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


@tool_parameters(_AGENDA_PARAMETERS)
class AgendaTool(Tool):
    """Tool to manage calendar appointments."""

    def __init__(self, workspace: str, default_timezone: str = "UTC"):
        self._workspace = workspace
        self._default_timezone = default_timezone

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return True

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(workspace=ctx.workspace, default_timezone=ctx.timezone)

    @property
    def name(self) -> str:
        return "agenda"

    @property
    def description(self) -> str:
        return (
            "Manage calendar appointments: add, list, update, delete. "
            "Dates are YYYY-MM-DD; time is HH:MM or null for all-day."
        )

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        errors = super().validate_params(params)
        action = params.get("action")
        if action == "add" and not str(params.get("title") or "").strip():
            errors.append("title is required when action='add'")
        if action == "add" and not str(params.get("date") or "").strip():
            errors.append("date is required when action='add'")
        if action in ("update", "delete") and not str(params.get("id") or "").strip():
            errors.append("id is required when action='update' or 'delete'")
        return errors

    async def execute(
        self,
        action: str,
        id: str | None = None,
        title: str | None = None,
        date: str | None = None,
        time: str | None = None,
        all_day: Any = False,
        notes: str | None = None,
        category: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> str:
        scope = default_workspace_scope(self._workspace, restrict_to_workspace=True)
        if action == "list":
            return self._list_appointments(scope)
        if action == "add":
            return self._add_appointment(
                scope,
                title=title,
                date=date,
                time=time,
                all_day=_parse_all_day(all_day),
                notes=notes,
                category=category,
                color=color,
            )
        if action == "update":
            return self._update_appointment(
                scope,
                appointment_id=str(id) if id else "",
                title=title,
                date=date,
                time=time,
                all_day=_parse_all_day(all_day),
                notes=notes,
                category=category,
                color=color,
            )
        if action == "delete":
            return self._delete_appointment(scope, appointment_id=str(id) if id else "")
        return f"Unknown action: {action}"

    def _list_appointments(self, scope: Any) -> str:
        payload = list_appointments(scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        appointments = payload.get("appointments", [])
        if not appointments:
            return "No appointments found."
        lines = ["Appointments:"]
        for appt in appointments:
            time_text = "all-day" if appt.get("all_day") else (appt.get("time") or "no time")
            lines.append(
                f"- {appt.get('date')} {time_text}: {appt.get('title')} "
                f"(id: {appt.get('id')}, category: {appt.get('category')})"
            )
        return "\n".join(lines)

    def _add_appointment(
        self,
        scope: Any,
        *,
        title: str | None,
        date: str | None,
        time: str | None,
        all_day: bool,
        notes: str | None,
        category: str | None,
        color: str | None,
    ) -> str:
        payload = create_appointment(
            {
                "title": title or "",
                "date": date or "",
                "time": time,
                "all_day": all_day,
                "description": notes or "",
                "category": category or "other",
                "color": color,
            },
            scope=scope,
        )
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        appt = payload["appointment"]
        return f"Created appointment '{appt['title']}' on {appt['date']} (id: {appt['id']})"

    def _update_appointment(
        self,
        scope: Any,
        *,
        appointment_id: str,
        title: str | None,
        date: str | None,
        time: str | None,
        all_day: bool,
        notes: str | None,
        category: str | None,
        color: str | None,
    ) -> str:
        changes: dict[str, Any] = {}
        if title is not None:
            changes["title"] = title
        if date is not None:
            changes["date"] = date
        if time is not None:
            changes["time"] = time
        # Only pass all_day if caller explicitly set it (bool is always present),
        # but we always update it to keep all_day/time consistency.
        changes["all_day"] = all_day
        if notes is not None:
            changes["description"] = notes
        if category is not None:
            changes["category"] = category
        if color is not None:
            changes["color"] = color
        if not changes:
            return "No changes provided."
        payload = update_appointment(appointment_id, changes, scope=scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        appt = payload["appointment"]
        return f"Updated appointment '{appt['title']}' on {appt['date']} (id: {appt['id']})"

    def _delete_appointment(self, scope: Any, *, appointment_id: str) -> str:
        payload = delete_appointment(appointment_id, scope=scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        return f"Deleted appointment {payload.get('id')}."
