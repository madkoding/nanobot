---
name: skill-creator
description: Create or update AgentSkills. Use when designing, structuring, or packaging skills with scripts, references, and assets.
---

# Skill Creator

Guidance for creating effective skills: modular, self-contained packages that give the agent specialized workflows, tool integrations, domain expertise, and bundled resources (scripts, references, assets) — "onboarding guides" that turn a general-purpose agent into a domain-specialized one.

## Core Principles

- **Concise is key**: the context window is shared with system prompt, history, and other skills. The agent is already very smart — only add context it doesn't have. Prefer examples over verbose explanations.
- **Degrees of freedom** (match specificity to task fragility):
  - **High freedom** (text instructions): multiple valid approaches, context-dependent decisions.
  - **Medium freedom** (pseudocode/parameterized scripts): preferred pattern exists, some variation OK.
  - **Low freedom** (specific scripts, few params): fragile operations, consistency critical, exact sequence required.

## Anatomy of a Skill

```
skill-name/
├── SKILL.md            # required: frontmatter (name, description) + Markdown body
├── scripts/            # optional: executable code (Python/Bash/etc.)
├── references/         # optional: docs loaded into context as needed
└── assets/             # optional: files used in output (templates, icons, fonts)
```

- **Frontmatter**: `name` + `description` are the ONLY trigger fields — describe what the skill does AND when to use it, comprehensively. "When to use" info belongs here, not in the body (body loads only after triggering).
- **scripts/**: deterministic or repeatedly-rewritten code. Token-efficient; may run without loading into context. Test added scripts by actually running them.
- **references/**: schemas, API docs, policies, detailed workflows. Keeps SKILL.md lean; load only when needed. For large files (>10k words), include grep patterns in SKILL.md. Avoid duplicating info between SKILL.md and references — detailed material goes in references.
- **assets/**: files used in the final output (templates, images, fonts), never loaded into context.

**Do NOT include**: README.md, CHANGELOG.md, INSTALLATION_GUIDE.md, or any auxiliary docs — only what an agent needs to do the job.

## Progressive Disclosure

Three levels: (1) metadata always in context; (2) SKILL.md body on trigger (<5k words / <500 lines — split into reference files near limits); (3) bundled resources as needed.

Patterns (reference files one level deep; add a table of contents for refs >100 lines):

- **High-level guide + references**: core workflow in SKILL.md; details in `FORMS.md`, `REFERENCE.md`, `EXAMPLES.md` linked with clear "read when" descriptions.
- **Domain/variant organization**: split refs by domain or variant (e.g., `aws.md`, `gcp.md`, `azure.md`) so only the relevant file loads.
- **Conditional details**: link advanced features (tracked changes, OOXML) only when needed.

## Creation Process

Follow in order, skipping only when clearly not applicable.

### Naming

Lowercase letters/digits/hyphens, <64 chars, verb-led when possible, namespace by tool when it helps triggering (`gh-address-comments`). Folder name = skill name.

### Step 1: Understand with concrete examples

Gather real usage examples from the user (avoid too many questions at once). Conclude when the functionality to support is clear.

### Step 2: Plan reusable contents

For each example, decide what to execute from scratch and which reusable resources help:
- Same code rewritten repeatedly → `scripts/rotate_pdf.py`
- Same boilerplate each time → `assets/hello-world/` template
- Schemas re-discovered each time → `references/schema.md`

### Step 3: Initialize (new skills)

Always run `init_skill.py` to generate the template skill directory:

```bash
scripts/init_skill.py <skill-name> --path <output-dir> [--resources scripts,references,assets] [--examples]
```

In `nanobot`, custom skills live under the active workspace `skills/` directory (e.g., `<workspace>/skills/my-skill/SKILL.md`) for auto-discovery. Delete `--examples` placeholders afterwards.

### Step 4: Edit the skill

Remember: the skill is written for another agent instance — include beneficial, non-obvious procedural knowledge.

1. Learn proven patterns from `references/workflows.md` (multi-step processes) and `references/output-patterns.md` (output formats/quality standards).
2. Implement reusable resources first (may need user input: assets, docs). Test scripts by running them; only create directories actually required.
3. Update SKILL.md: frontmatter per guidelines above; body with imperative/infinitive instructions for using the skill and its resources. Keep frontmatter minimal (`metadata`, `always` supported only when needed).

### Step 5: Package

```bash
scripts/package_skill.py <path/to/skill-folder> [output-dir]
```

Validates (frontmatter format, naming, structure, description quality, resource references) then creates `<skill-name>.skill` (a zip). Symlinks are rejected. Fix reported validation errors and re-run.

### Step 6: Iterate

Use on real tasks → notice struggles → update SKILL.md or resources → test again. Iteration usually happens right after use, with fresh performance context.