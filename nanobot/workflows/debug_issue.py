"""debug_issue: reproduce, hypothesize, fix, and verify a bug."""

from nanobot.workflows.runner import AgentResult

ARGUMENTS = {
    "symptom": "Observed bug or failing behavior.",
}

PHASES = ["reproduce", "hypothesize", "fix", "verify"]


async def run(args: dict, ctx):
    symptom = str(args.get("symptom") or "").strip()
    if not symptom:
        return AgentResult(
            text="No symptom provided. Usage: /workflow debug_issue symptom=<symptom>"
        )

    ctx.set_phase("reproduce")
    repro = await ctx.agent(
        agent="general",
        prompt=f"Investigate and reproduce this bug: {symptom}\nReport reproduction steps and the likely root cause.",
    )

    ctx.set_phase("hypothesize")
    hypothesis = await ctx.agent(
        agent="plan",
        prompt=f"Based on this investigation, propose a root-cause hypothesis and a minimal fix:\n\n{repro.text}",
    )

    ctx.set_phase("fix")
    fix = await ctx.agent(
        agent="build",
        prompt=f"Implement the fix. Describe what changed and why:\n\n{hypothesis.text}",
    )

    ctx.set_phase("verify")
    return await ctx.agent(
        agent="general",
        prompt=f"Verify the fix: confirm it addresses the root cause, suggest a regression test, and check the original symptom is gone:\n\n{fix.text}",
    )
