"""A phase-gated pipeline: PLAN -> BUILD -> REVIEW, with a bounded REVIEW->BUILD
rework edge.

GADK spreads the same idea across `src/services/sdlc_phase.py` (Phase enum,
ALLOWED_TRANSITIONS), `src/services/phase_controller.py` (gate evaluation),
`src/services/phase_store.py` (persistence), and `src/services/workflow_graphs.py`
(LangGraph-optional graph execution) — four modules and an optional heavyweight
graph-execution dependency for what is, structurally, a three-step loop with a
retry counter. This keeps the same guarantee (bounded retries, explicit history)
in one file, in plain Python control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import Agent, AgentRunResult, run_agent
from .backend import OllamaBackend
from .memory import Memory
from .router import Router
from .tools import ToolRegistry


@dataclass
class PhaseRecord:
    phase: str
    agent: str
    model_used: str
    content: str


@dataclass
class SwarmResult:
    goal: str
    history: list[PhaseRecord] = field(default_factory=list)
    approved: bool = False
    rework_count: int = 0


class Swarm:
    """Runs Planner -> Builder -> Critic. If the Critic rejects, sends the
    critique back to Builder, up to `max_rework` times, then stops and reports
    what it has (never loops forever)."""

    def __init__(
        self,
        planner: Agent,
        builder: Agent,
        critic: Agent,
        backend: OllamaBackend,
        registry: ToolRegistry,
        memory: Memory | None = None,
        max_rework: int = 2,
    ) -> None:
        self.planner = planner
        self.builder = builder
        self.critic = critic
        self.backend = backend
        self.registry = registry
        self.memory = memory
        self.max_rework = max_rework
        self.router = Router()

    def _run(self, agent: Agent, message: str, context: str | None = None) -> AgentRunResult:
        return run_agent(agent, message, self.backend, self.router, self.registry, context=context)

    def run(self, goal: str) -> SwarmResult:
        result = SwarmResult(goal=goal)
        context = None
        if self.memory:
            recalled = self.memory.recall(goal, top_k=3)
            if recalled:
                context = "\n".join(f"- {e.text}" for e in recalled)

        plan = self._run(self.planner, goal, context=context)
        result.history.append(PhaseRecord("PLAN", self.planner.name, plan.model_used, plan.content))

        build_input = f"Goal: {goal}\n\nPlan:\n{plan.content}"
        build = self._run(self.builder, build_input)
        result.history.append(PhaseRecord("BUILD", self.builder.name, build.model_used, build.content))

        for _ in range(self.max_rework + 1):
            review_input = f"Goal: {goal}\n\nProposed solution:\n{build.content}\n\nRespond with APPROVE or REQUEST_CHANGES: <reason>."
            review = self._run(self.critic, review_input)
            result.history.append(PhaseRecord("REVIEW", self.critic.name, review.model_used, review.content))

            if "APPROVE" in review.content.upper():
                result.approved = True
                break

            result.rework_count += 1
            if result.rework_count > self.max_rework:
                break

            rework_input = f"Goal: {goal}\n\nPrevious attempt:\n{build.content}\n\nReviewer feedback:\n{review.content}\n\nRevise accordingly."
            build = self._run(self.builder, rework_input)
            result.history.append(PhaseRecord("BUILD", self.builder.name, build.model_used, build.content))

        if self.memory:
            self.memory.remember(f"Goal: {goal}\nOutcome: {'approved' if result.approved else 'unresolved'}", tag="run_summary")

        return result
