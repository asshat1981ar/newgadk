"""A phase-gated pipeline: PLAN -> ARCHITECT -> IMPLEMENT -> REVIEW -> GOVERN ->
OPERATE, with two bounded rework edges (REVIEW->IMPLEMENT and GOVERN->IMPLEMENT).

GADK spreads the same idea across `src/services/sdlc_phase.py` (Phase enum,
ALLOWED_TRANSITIONS), `src/services/phase_controller.py` (gate evaluation),
`src/services/phase_store.py` (persistence), and `src/services/workflow_graphs.py`
(LangGraph-optional graph execution) — four modules and an optional heavyweight
graph-execution dependency for what is, structurally, a six-step loop with two
retry counters. This keeps the same guarantee (bounded retries, explicit history)
in one file, in plain Python control flow.

OPERATE (FinOps) always runs, pass or fail — cost/usage reporting shouldn't be
gated on whether the run succeeded.
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
    approved: bool = False       # Critic's verdict, as today
    rework_count: int = 0        # Critic rework count, as today
    governed: bool = False       # True only if Governor said GO
    governor_rework_count: int = 0
    finops_summary: str = ""     # FinOps agent's final content, for easy access without digging through history


class Swarm:
    """Runs Planner -> Architect -> Builder -> Critic -> Governor -> FinOps.

    If the Critic requests changes, sends the critique back to Builder, up to
    `max_rework` times. Once the Critic approves, the Governor runs the real
    quality gates; on NO-GO the Governor's feedback goes back to Builder, the
    Critic re-reviews (Governor only ever sees Critic-approved work), and the
    Governor runs again, up to `max_governor_rework` times. FinOps always runs
    last, regardless of how governance ended, summarizing the token ledger
    accumulated across every phase.
    """

    def __init__(
        self,
        planner: Agent,
        architect: Agent,
        builder: Agent,
        critic: Agent,
        governor: Agent,
        finops: Agent,
        backend: OllamaBackend,
        registry: ToolRegistry,
        memory: Memory | None = None,
        max_rework: int = 2,
        max_governor_rework: int = 1,
    ) -> None:
        self.planner = planner
        self.architect = architect
        self.builder = builder
        self.critic = critic
        self.governor = governor
        self.finops = finops
        self.backend = backend
        self.registry = registry
        self.memory = memory
        self.max_rework = max_rework
        self.max_governor_rework = max_governor_rework
        self.router = Router()
        self._ledger: list[dict[str, Any]] = []

    def _run(self, agent: Agent, message: str, context: str | None = None) -> AgentRunResult:
        return run_agent(agent, message, self.backend, self.router, self.registry, context=context)

    def _record(self, run_result: AgentRunResult, phase: str, agent: Agent, history: list[PhaseRecord]) -> None:
        history.append(PhaseRecord(phase, agent.name, run_result.model_used, run_result.content))
        self._ledger.append(
            {
                "phase": phase,
                "agent": agent.name,
                "model": run_result.model_used,
                "tokens_in": run_result.tokens_in,
                "tokens_out": run_result.tokens_out,
            }
        )

    @staticmethod
    def _is_go(content: str) -> bool:
        """`"NO-GO"` contains the substring `"GO"`, so a naive `"GO" in text` check
        would misread a NO-GO as a GO. Check for NO-GO first; only treat it as a
        go if that's absent and "GO" is present."""
        upper = content.upper()
        if "NO-GO" in upper:
            return False
        return "GO" in upper

    def _review_cycle(self, goal: str, build: AgentRunResult, result: SwarmResult) -> tuple[AgentRunResult, bool]:
        """The Critic REVIEW<->IMPLEMENT rework loop, bounded by `max_rework`.
        Returns the (possibly revised) build and whether the Critic approved it."""
        approved = False
        for _ in range(self.max_rework + 1):
            review_input = (
                f"Goal: {goal}\n\nProposed solution:\n{build.content}\n\n"
                "Respond with APPROVE or REQUEST_CHANGES: <reason>."
            )
            review = self._run(self.critic, review_input)
            self._record(review, "REVIEW", self.critic, result.history)

            if "APPROVE" in review.content.upper():
                approved = True
                break

            result.rework_count += 1
            if result.rework_count > self.max_rework:
                break

            rework_input = (
                f"Goal: {goal}\n\nPrevious attempt:\n{build.content}\n\n"
                f"Reviewer feedback:\n{review.content}\n\nRevise accordingly."
            )
            build = self._run(self.builder, rework_input)
            self._record(build, "IMPLEMENT", self.builder, result.history)

        return build, approved

    def _format_ledger(self) -> str:
        return "\n".join(
            f"{entry['phase']} ({entry['agent']} via {entry['model']}): "
            f"{entry['tokens_in']} in / {entry['tokens_out']} out"
            for entry in self._ledger
        )

    def run(self, goal: str) -> SwarmResult:
        result = SwarmResult(goal=goal)
        self._ledger = []
        context = None
        if self.memory:
            recalled = self.memory.recall(goal, top_k=3)
            if recalled:
                context = "\n".join(f"- {e.text}" for e in recalled)

        plan = self._run(self.planner, goal, context=context)
        self._record(plan, "PLAN", self.planner, result.history)

        architect_input = f"Goal: {goal}\n\nPlan:\n{plan.content}"
        architect = self._run(self.architect, architect_input)
        self._record(architect, "ARCHITECT", self.architect, result.history)

        build_input = f"Goal: {goal}\n\nPlan:\n{plan.content}\n\nArchitecture notes:\n{architect.content}"
        build = self._run(self.builder, build_input)
        self._record(build, "IMPLEMENT", self.builder, result.history)

        build, approved = self._review_cycle(goal, build, result)
        result.approved = approved

        if approved:
            governor_rework_count = 0
            while True:
                governor_input = f"Goal: {goal}\n\nSolution:\n{build.content}"
                governor = self._run(self.governor, governor_input)
                self._record(governor, "GOVERN", self.governor, result.history)

                if self._is_go(governor.content):
                    result.governed = True
                    break

                governor_rework_count += 1
                if governor_rework_count > self.max_governor_rework:
                    break

                rework_input = (
                    f"Goal: {goal}\n\nPrevious attempt:\n{build.content}\n\n"
                    f"Governor feedback:\n{governor.content}\n\nRevise accordingly."
                )
                build = self._run(self.builder, rework_input)
                self._record(build, "IMPLEMENT", self.builder, result.history)

                build, approved = self._review_cycle(goal, build, result)
                result.approved = approved
                if not approved:
                    break

            result.governor_rework_count = governor_rework_count

        finops_input = f"Goal: {goal}\n\nToken ledger:\n{self._format_ledger()}"
        finops = self._run(self.finops, finops_input)
        self._record(finops, "OPERATE", self.finops, result.history)
        result.finops_summary = finops.content

        if self.memory:
            if result.approved and result.governed:
                outcome = "approved and governed"
            elif result.approved and not result.governed:
                outcome = "approved but ungoverned (governor rework exhausted)"
            else:
                outcome = "unresolved"
            self.memory.remember(f"Goal: {goal}\nOutcome: {outcome}", tag="run_summary")

        return result
