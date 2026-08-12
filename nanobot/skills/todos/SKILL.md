---
name: todos
description: Manage todo lists and calendar appointments together.
metadata:
  nanobot:
    always: true
---

# Todos and Agenda

You can manage todo lists via the `todos` tool and calendar appointments via the `agenda` tool.

## Lists

- A todo list is identified by a **slug** (e.g. `madkoding`, `compras`).
- Use `todos(action="list_lists")` to discover existing lists and their slugs.
- Use `todos(action="add_list", name="...", slug="...")` to create a new list if needed.

## Items

- Add an item: `todos(action="add_item", slug="<slug>", text="...", due_date="YYYY-MM-DD", notes="...", assignee="...")`
- Mark done: `todos(action="update_item", slug="<slug>", item_id="...", done=true)`
- List items: `todos(action="list_items", slug="<slug>")`

## Appointments

- Add an appointment: `agenda(action="add", title="...", date="YYYY-MM-DD", time="HH:MM", category="personal")`
- Use `category="reminder"` for things that are mainly alerts.

## Combined requests

When the user asks to add something to a list **and** schedule it for a date/time, do **both**:

1. Add a todo item to the requested list (or default list `madkoding` if unspecified).
2. Add an appointment on the requested date/time with a matching title.

Examples:

- User: "add chicken to my list for this Friday"
  - Interpret "my list" as the user's default list slug (`madkoding` if unsure).
  - Compute "this Friday" as the next Friday date.
  - `todos(action="add_item", slug="madkoding", text="buy chicken", due_date="<friday>")`
  - `agenda(action="add", title="buy chicken", date="<friday>", category="reminder", all_day="true")`
  - Reply confirming both.

- User: "add chicken to compras for tomorrow at 10am"
  - `todos(action="add_item", slug="compras", text="buy chicken", due_date="<tomorrow>")`
  - `agenda(action="add", title="buy chicken", date="<tomorrow>", time="10:00", category="reminder")`

## Date resolution

- Relative date expressions like "this Friday", "next Monday", "tomorrow", "day after tomorrow" must be resolved to an absolute `YYYY-MM-DD` based on today's date.
- Timezone: use the workspace timezone; if unknown, default to `America/Santiago`.
- If a time is not given, prefer `all_day="true"` for the appointment.
