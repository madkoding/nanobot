---
name: research
description: Deep research that produces a report and a publishable article. Use when the operator asks to "investigate", "make a report", "research", "article about", "write an article", "report about", or wants a document with tags and a shareable link. Combines web search, synthesis, writing, and publishing.
---

# Research

Turn a topic into a structured report and a ready-to-share article.

## When to run

- If this skill is **active in the system prompt** (because the chat is marked as research), **RUN this flow for EVERY user message** without asking whether they want to research.
- If it is not active, activate it when the operator asks to investigate, report, or write an article.

## Flow (run step by step)

### 1. Clarify scope (only if ambiguous)
If the topic is vague, ask briefly: focus, depth (summary vs. detailed), language, and audience. If it is clear, proceed without asking.

### 2. Research in depth
- Launch **between 6 and 12 queries** with `web_search` covering synonyms, different angles, primary sources, recent news, third-party analysis, and results in the user's language.
- Refine the search strategy: if the first results are shallow, rephrase the queries with more specific terms, search operators, proper nouns, or years.
- Use `web_fetch` on **at least 5 relevant sources** to read full content instead of relying on search snippets.
- Gather data, figures, dates, names, verbatim quotes, and URLs. Do not invent anything.
- If a promising result has internal links to relevant sections, follow at least 2-3 of them with `web_fetch`.

### 3. Build the report
Write a thorough report in Markdown:
- **Title** and date.
- **Executive summary** (3-5 lines).
- **Sections** with clear subheadings.
- **Data and evidence** with citations to the sources.
- **Conclusion** and, if applicable, recommendations.
- **Sources** at the end (list of URLs).

Save it with `write_file` at `research/<slug>/reporte.md`.

### 4. Build the article
From the report, write a **long, complete article** (not the raw report):
- Catchy title, engaging introduction, fluid and detailed body, closing.
- The article must have **at least 5 main sections**, with clear subheadings and natural transitions.
- Include examples, figures, dates, proper nouns, and historical or background context drawn from the sources.
- Cite the sources with direct inline links or footnotes.
- **Tags**: 3-8 relevant tags (`#tag` format).
- Tone suited to the audience.

**Required frontmatter** at the top of `articulo.md` (the WebUI uses it to show title, tags, and sources):
```yaml
---
title: <article title>
date: <YYYY-MM-DD>
tags:
  - <tag1>
  - <tag2>
sources:
  - <url1>
  - <url2>
---
```
`date` is the current date; `tags` 3-8; `sources` the URLs of the consulted sources.

Save it with `write_file` at `research/<slug>/articulo.md`.

### 5. Confirm with the operator
When done, reply briefly in the chat:
- Research title.
- File paths (`research/<slug>/reporte.md` and `articulo.md`).
- Tags.
- Ask whether they want to share it with sharemd.

### 6. Share with sharemd (only with consent)
Before publishing, **ask for explicit confirmation**. Show the article and ask whether they want to share it.

Once confirmed:
1. If `sharemd` is not installed or not on `PATH`, install it locally in the workspace to avoid touching the global system:
   ```bash
   npm install --prefix .workspace_tools sharemd   # or: npx --yes sharemd ... for a one-off run
   ```
   If Node 18+ is not available or the install fails, tell the operator and hand over the local article path instead.
2. Publish the article (use the local path if you installed with `--prefix`):
   ```bash
   sharemd research/<slug>/articulo.md
   ```
3. Save the link in `research/<slug>/sharemd.json`:
   ```json
   {"url": "https://sharemd.sh/<id>"}
   ```
4. Hand the link to the operator.

> The `.sharemd` file (edit token) is created next to the article. Do not commit it to git or share it.

## Rules
- **If the skill is active, research without asking for extra permission.** The user already entered the Research section.
- Cite real sources; do not invent data or URLs.
- The report and the article are different documents: the report is exhaustive, the article is readable.
- If the search returns no results, say so and ask how to proceed.
