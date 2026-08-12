"""feature_plan: design doc, ordered implementation plan, and risk assessment."""

from nanobot.workflows.runner import AgentResult

ARGUMENTS = {
    "feature": "Feature to implement.",
    "path": "Optional repo path/area to focus on.",
}

PHASES = ["design", "plan", "risk"]


async def run(args: dict, ctx):
    feature = str(args.get("feature") or "").strip()
    if not feature:
        return AgentResult(
            text="No feature provided. Usage: /workflow feature_plan feature=<feature>"
        )
    focus = str(args.get("path") or "").strip()
    focus_note = f"\nFocus area: {focus}" if focus else ""

    ctx.set_phase("design")
    design = await ctx.agent(
        agent="build",
        prompt=f"Design the implementation for: {feature}{focus_note}",
    )

    ctx.set_phase("plan")
    plan = await ctx.agent(
        agent="plan",
        prompt=f"Break this into ordered, reviewable steps with acceptance criteria:\n\n{design.text}",
    )

    ctx.set_phase("risk")
    return await ctx.agent(
        agent="plan",
        prompt=f"List risks, unknowns, and test gaps for this plan:\n\n{plan.text}",
    )
