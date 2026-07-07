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

    from .security_gates import register_security_tools

    register_security_tools(registry)

    # fs/shell/git tools are opt-in: unset (or anything other than "1") keeps
    # Builder toy-tools-only, so a default run can't touch the filesystem.
    if os.environ.get("OLLAMA_SWARM_ENABLE_DEV_TOOLS") == "1":
        from .dev_tools import register_dev_tools

        register_dev_tools(registry)

    return registry


@dataclass
class SwarmAgents:
    planner: Agent
    scaffolder: Agent
    architect: Agent
    builder: Agent
    test_gen: Agent
    critic: Agent
    security: Agent
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
    # Scaffolder is a label only — SCAFFOLD is deterministic (see
    # orchestrator.py's SCAFFOLD phase, which calls scaffold_project()
    # directly and never routes through run_agent(scaffolder, ...)). This
    # Agent exists solely so PhaseRecord/self._record() has a name to
    # attribute the SCAFFOLD phase to; it is never sent to the LLM.
    scaffolder = Agent(
        name="Scaffolder",
        system_prompt="Unused: SCAFFOLD is a deterministic phase, not an LLM call.",
        tier=Tier.CODING,
    )
    architect = Agent(
        name="Architect",
        system_prompt=(
            "You are the Architect. Given the goal and the planner's step list, "
            "optionally inspect the current workspace layout with your tools. "
            "Produce a short design note: which files/components to add or "
            "change, how data flows between them, and 2-3 explicit constraints "
            "the builder must respect. Also state the target language explicitly "
            "(e.g. 'Language: Python'). Do not write implementation code. Keep it "
            "under ~200 words unless the goal genuinely requires more."
        ),
        tier=Tier.REASONING,
        tools=["list_dir", "read_file"],
    )
    builder = Agent(
        name="Builder",
        system_prompt=(
            "You are the Builder. Given a goal and a plan (or prior attempt plus "
            "reviewer feedback and/or failing tests), produce the actual solution. "
            "Be direct and complete. The workspace skeleton already exists — build "
            "on top of it, do not recreate files that are already there unless "
            "explicitly told to replace them."
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
    test_gen = Agent(
        name="TestEngineer",
        system_prompt=(
            "You are the Test Engineer. Read the Builder's implementation in the "
            "workspace and write comprehensive tests covering every public function, "
            "class, and edge case. Use the workspace's existing test framework "
            "(pytest for Python, node:test for Node, cargo test for Rust). "
            "Write tests to the tests/ directory. Aim for full coverage — "
            "happy paths, error paths, boundary conditions."
        ),
        tier=Tier.CODING,
        tools=["read_file", "write_file", "list_dir", "run_shell"],
    )
    critic = Agent(
        name="Critic",
        system_prompt=(
            "You are the Critic. Judge whether the proposed solution actually "
            "satisfies the goal, considering both the implementation and the "
            "generated tests. Reply with either 'APPROVE' or "
            "'REQUEST_CHANGES: <specific, actionable reason>'. Be strict but fair."
        ),
        tier=Tier.REASONING,
    )
    security = Agent(
        name="Security",
        system_prompt=(
            "You are the Security Auditor. Call `run_security_scan` with no arguments — "
            "it already knows the workspace path and requires no input from you. Never "
            "ask a clarifying question or request a path; just call it. Trust its output "
            "completely. Reply with exactly one of:\n"
            "  SECURITY: GO                  — no issues found\n"
            "  SECURITY: WARN: <details>     — scanner missing or low-severity findings\n"
            "  SECURITY: NO-GO: <finding>    — high/critical severity findings\n"
            "Do not add commentary beyond the verdict line."
        ),
        tier=Tier.REASONING,
        tools=["run_security_scan"],
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
        scaffolder=scaffolder,
        architect=architect,
        builder=builder,
        test_gen=test_gen,
        critic=critic,
        security=security,
        governor=governor,
        finops=finops,
    )
