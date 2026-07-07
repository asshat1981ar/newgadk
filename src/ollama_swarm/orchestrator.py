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
    security_verdict: str = ""   # Last Security agent verdict
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
        scaffolder: Agent,
        architect: Agent,
        builder: Agent,
        test_gen: Agent,
        critic: Agent,
        security: Agent,
        governor: Agent,
        finops: Agent,
        backend: OllamaBackend,
        registry: ToolRegistry,
        memory: Memory | None = None,
        max_rework: int = 2,
        max_governor_rework: int = 1,
    ) -> None:
        self.planner = planner
        self.scaffolder = scaffolder
        self.architect = architect
        self.builder = builder
        self.test_gen = test_gen
        self.critic = critic
        self.security = security
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
        """Check if the governor verdict is GO or NO-GO, defaulting non-standard values to False."""
        upper = content.upper()
        if "NO-GO" in upper or "NO GO" in upper:
            return False
        if "GOVERN: GO" in upper:
            return True
        if "GOVERN:" in upper:
            return False
        return "GO" in upper

    @staticmethod
    def _tool_evidence(build: AgentRunResult) -> str:
        """Summarize the Builder's actual tool activity for the Critic. The Critic
        only sees final prose otherwise, and an agent that really did the work via
        tools shouldn't be rejected for not narrating it."""
        if not build.tool_calls_made:
            return "Tool execution evidence: none (no tools were called)."
        tool_results = [m["content"] for m in build.transcript if m.get("role") == "tool"]
        excerpts = "\n".join(f"- {r[:300]}" for r in tool_results[-5:])
        return (
            f"Tool execution evidence: {build.tool_calls_made} real tool call(s) were "
            f"executed during implementation. Recent tool results:\n{excerpts}"
        )

    @staticmethod
    def _is_security_go(content: str) -> tuple[bool, bool]:
        """Return (is_go, is_warn) for a Security agent response."""
        upper = content.upper()
        if "SECURITY: NO-GO" in upper:
            return False, False
        if "SECURITY: WARN" in upper:
            return True, True
        if "SECURITY: GO" in upper:
            return True, False
        # Default to warn if verdict is ambiguous
        return True, True

    def _review_cycle(self, goal: str, build: AgentRunResult, result: SwarmResult) -> tuple[AgentRunResult, bool]:
        """The Critic REVIEW<->IMPLEMENT rework loop, bounded by `max_rework`.
        Returns the (possibly revised) build and whether the Critic approved it."""
        approved = False
        for _ in range(self.max_rework + 1):
            review_input = (
                f"Goal: {goal}\n\nProposed solution:\n{build.content}\n\n"
                f"{self._tool_evidence(build)}\n\n"
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
        # Validate all required agents up-front so failures are explicit.
        _required = (
            "planner", "scaffolder", "architect", "builder",
            "test_gen", "critic", "security", "governor", "finops",
        )
        for _slot in _required:
            if getattr(self, _slot) is None:
                raise TypeError(f"Swarm.run() requires a non-None agent for '{_slot}'")

        result = SwarmResult(goal=goal)
        self._ledger = []
        context = None
        if self.memory:
            recalled = self.memory.recall(goal, top_k=3)
            if recalled:
                context = "\n".join(f"- {e.text}" for e in recalled)

        # --- PLAN ---
        plan = self._run(self.planner, goal, context=context)
        self._record(plan, "PLAN", self.planner, result.history)

        # --- ARCHITECT ---
        architect_input = f"Goal: {goal}\n\nPlan:\n{plan.content}"
        architect = self._run(self.architect, architect_input)
        self._record(architect, "ARCHITECT", self.architect, result.history)

        # --- SCAFFOLD (always-on) ---
        from .scaffold import scaffold_project, detect_language
        project_name = goal.split()[0] if goal.split() else "project"
        written_files = scaffold_project(architect.content, project_name=project_name)
        scaffold_summary = (
            f"Scaffolded {len(written_files)} file(s) for language "
            f"'{detect_language(architect.content)}': {', '.join(written_files.keys()) or 'none (all existed)'}"
        )
        # Record scaffold as a synthetic phase entry
        from .agents import AgentRunResult
        scaffold_run = AgentRunResult(
            content=scaffold_summary,
            model_used="scaffold",
            tokens_in=0,
            tokens_out=0,
            tool_calls_made=len(written_files),
            transcript=[],
        )
        self._record(scaffold_run, "SCAFFOLD", self.scaffolder, result.history)

        # --- IMPLEMENT ---
        build_input = (
            f"Goal: {goal}\n\nPlan:\n{plan.content}\n\n"
            f"Architecture notes:\n{architect.content}\n\n"
            f"Scaffolded files already in workspace:\n{scaffold_summary}"
        )
        build = self._run(self.builder, build_input)
        self._record(build, "IMPLEMENT", self.builder, result.history)

        # --- TEST-GEN ---
        test_gen_input = (
            f"Goal: {goal}\n\nThe Builder has implemented the solution. "
            f"Implementation summary:\n{build.content}\n\n"
            "Write comprehensive tests covering every public function, class, and edge case. "
            "Use the test framework appropriate for the detected language."
        )
        test_gen_run = self._run(self.test_gen, test_gen_input)
        self._record(test_gen_run, "TEST-GEN", self.test_gen, result.history)

        # Provide test output to Builder rework loop context
        build_with_tests_context = (
            f"{build.content}\n\n"
            f"Generated tests:\n{test_gen_run.content}\n\n"
            f"{self._tool_evidence(test_gen_run)}"
        )
        # Create a synthetic run result that merges implementation + test evidence
        import dataclasses
        augmented_build = dataclasses.replace(
            build,
            content=build_with_tests_context,
        )

        # --- REVIEW (with tests context) ---
        augmented_build, approved = self._review_cycle(goal, augmented_build, result)
        result.approved = approved

        if approved:
            # --- SECURITY ---
            security_input = (
                f"Goal: {goal}\n\nImplementation has been approved by the Critic. "
                "Run the security scan now."
            )
            security_run = self._run(self.security, security_input)
            self._record(security_run, "SECURITY", self.security, result.history)

            security_go, security_warn = self._is_security_go(security_run.content)
            result.security_verdict = security_run.content

            # On SECURITY: NO-GO, send findings back to Builder for one fix pass,
            # then re-run security once more (no loop — one chance to fix).
            if not security_go:
                sec_fix_input = (
                    f"Goal: {goal}\n\nPrevious attempt:\n{augmented_build.content}\n\n"
                    f"Security findings:\n{security_run.content}\n\nFix the security issues."
                )
                build_fixed = self._run(self.builder, sec_fix_input)
                self._record(build_fixed, "IMPLEMENT", self.builder, result.history)
                augmented_build = dataclasses.replace(build_fixed, content=build_fixed.content)

                security_run2 = self._run(self.security, security_input)
                self._record(security_run2, "SECURITY", self.security, result.history)
                security_go, _ = self._is_security_go(security_run2.content)
                result.security_verdict = security_run2.content

            # --- GOVERN ---
            governor_rework_count = 0
            while True:
                governor_input = f"Goal: {goal}\n\nSolution:\n{augmented_build.content}"
                governor = self._run(self.governor, governor_input)
                self._record(governor, "GOVERN", self.governor, result.history)

                if self._is_go(governor.content):
                    result.governed = True
                    break

                governor_rework_count += 1
                if governor_rework_count > self.max_governor_rework:
                    break

                rework_input = (
                    f"Goal: {goal}\n\nPrevious attempt:\n{augmented_build.content}\n\n"
                    f"Governor feedback:\n{governor.content}\n\nRevise accordingly."
                )
                build_revised = self._run(self.builder, rework_input)
                self._record(build_revised, "IMPLEMENT", self.builder, result.history)
                augmented_build = dataclasses.replace(build_revised, content=build_revised.content)

                augmented_build, approved = self._review_cycle(goal, augmented_build, result)
                result.approved = approved
                if not approved:
                    break

            result.governor_rework_count = governor_rework_count

        # --- OPERATE (FinOps — always runs) ---
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
