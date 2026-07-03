"""Default six-agent swarm and a couple of demo tools, so `cli.py` and the
example script have something concrete to run without every caller re-defining
prompts from scratch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .agents import Agent
from .config import Tier
from .tools import ToolRegistry


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def current_time() -> str:
        """Return the current UTC time."""
        return datetime.now(timezone.utc).isoformat()

    @registry.register
    def word_count(text: str) -> int:
        """Count words in a piece of text."""
        return len(text.split())

    # Governor's gate is narrow and always needed, regardless of the dev-tools
    # opt-in below.
    from .quality_gates import register_governance_tools

    register_governance_tools(registry)

    # fs/shell/git tools are opt-in: unset (or anything other than "1") keeps
    # Builder toy-tools-only, so a default run can't touch the filesystem.
    if os.environ.get("OLLAMA_SWARM_ENABLE_DEV_TOOLS") == "1":
        from .dev_tools import register_dev_tools

        register_dev_tools(registry)

    return registry


@dataclass
class SwarmAgents:
    planner: Agent
    architect: Agent
    builder: Agent
    critic: Agent
    governor: Agent
    finops: Agent


def default_swarm_agents() -> SwarmAgents:
    planner = Agent(
        name="Planner",
        system_prompt=(
            "You are the Planner. Break the user's goal into a short, numbered "
            "list of concrete steps. Be concise. Do not write the solution itself."
        ),
        tier=Tier.REASONING,
    )
    architect = Agent(
        name="Architect",
        system_prompt=(
            "You are the Architect. Given the goal and the Planner's step list, "
            "optionally inspect the current workspace layout with your tools. "
            "Produce a short design note: which files/components to add or "
            "change, how data flows between them, and 2-3 explicit constraints "
            "the Builder must respect. Do not write implementation code. Keep it "
            "under ~200 words unless the goal genuinely requires more."
        ),
        tier=Tier.REASONING,
        tools=["list_dir", "read_file"],
    )
    builder = Agent(
        name="Builder",
        system_prompt=(
            "You are the Builder. Given a goal and a plan (or prior attempt plus "
            "reviewer feedback), produce the actual solution. Be direct and complete."
        ),
        tier=Tier.CODING,
        tools=[
            "current_time",
            "word_count",
            "read_file",
            "write_file",
            "list_dir",
            "run_shell",
            "git_diff",
            "git_commit",
        ],
    )
    critic = Agent(
        name="Critic",
        system_prompt=(
            "You are the Critic. Judge whether the proposed solution actually "
            "satisfies the goal. Reply with either 'APPROVE' or "
            "'REQUEST_CHANGES: <specific, actionable reason>'. Be strict but fair."
        ),
        tier=Tier.REASONING,
    )
    governor = Agent(
        name="Governor",
        system_prompt=(
            "You are the Governor. You have one tool, `run_quality_gates`, which "
            "actually executes the project's test suite (and linter, if "
            "available). Call it. Trust its output over anything claimed earlier "
            "in the transcript. Reply with exactly `GOVERN: GO` if tests pass, or "
            "`GOVERN: NO-GO: <reason>` if they don't - base this strictly on the "
            "tool result."
        ),
        tier=Tier.REASONING,
        tools=["run_quality_gates"],
    )
    finops = Agent(
        name="FinOps",
        system_prompt=(
            "You are FinOps. You will receive a per-phase token usage ledger for "
            "this run. In 2-4 sentences: state total tokens used, name which "
            "phase/model consumed the most, and flag one thing a human reviewer "
            "should notice. Do not repeat the raw numbers verbatim, interpret them."
        ),
        tier=Tier.FAST,
    )
    return SwarmAgents(
        planner=planner,
        architect=architect,
        builder=builder,
        critic=critic,
        governor=governor,
        finops=finops,
    )
