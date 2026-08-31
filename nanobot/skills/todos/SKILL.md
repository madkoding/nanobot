---
name: todos
description: Manage todo lists and calendar appointments together.
metadata:
  nanobot:
    always: true
---

# Todos and Agenda

Manage todo lists via the `todos` tool and calendar appointments via the `agenda` tool.

## Lists

- A todo list is identified by a **slug** (e.g. `madkoding`, `compras`).
- Discover: `todos(action="list_lists")` → existing lists and slugs. Create: `todos(action="add_list", name="...", slug="...")`.

## Items

- Add: `todos(action="add_item", slug="<slug>", text="...", due_date="YYYY-MM-DD", notes="...", assignee="...")`
- Done: `todos(action="update_item", slug="<slug>", item_id="...", done=true)`
- List: `todos(action="list_items", slug="<slug>")`

## Appointments

- Add: `agenda(action="add", title="...", date="YYYY-MM-DD", time="HH:MM", category="personal")`; use `category="reminder"` for things that are mainly alerts.

## Combined requests

User asks to add to a list **and** schedule for a date/time → do **both**: todo item to the requested list (default slug `madkoding` if unspecified) + appointment with matching title; reply confirming both.

- "add chicken to my list for this Friday" → `todos(action="add_item", slug="madkoding", text="buy chicken", due_date="<friday>")` + `agenda(action="add", title="buy chicken", date="<friday>", category="reminder", all_day="true")`
- "add chicken to compras for tomorrow at 10am" → `todos(action="add_item", slug="compras", text="buy chicken", due_date="<tomorrow>")` + `agenda(action="add", title="buy chicken", date="<tomorrow>", time="10:00", category="reminder")`

## Date resolution

- Resolve relative dates ("this Friday", "next Monday", "tomorrow", "day after tomorrow") to absolute `YYYY-MM-DD` based on today's date.
- Timezone: workspace timezone; default `America/Santiago` if unknown.
- No time given → prefer `all_day="true"`.