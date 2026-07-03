"""Default three-agent swarm and a couple of demo tools, so `cli.py` and the
example script have something concrete to run without every caller re-defining
prompts from scratch."""

from __future__ import annotations

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

    return registry


def default_swarm_agents() -> tuple[Agent, Agent, Agent]:
    planner = Agent(
        name="Planner",
        system_prompt=(
            "You are the Planner. Break the user's goal into a short, numbered "
            "list of concrete steps. Be concise. Do not write the solution itself."
        ),
        tier=Tier.REASONING,
    )
    builder = Agent(
        name="Builder",
        system_prompt=(
            "You are the Builder. Given a goal and a plan (or prior attempt plus "
            "reviewer feedback), produce the actual solution. Be direct and complete."
        ),
        tier=Tier.CODING,
        tools=["current_time", "word_count"],
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
    return planner, builder, critic
