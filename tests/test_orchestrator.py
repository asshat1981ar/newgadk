from __future__ import annotations

from unittest.mock import patch

from ollama_swarm.agents import Agent
from ollama_swarm.config import Tier, SETTINGS
from ollama_swarm.memory import Memory
from ollama_swarm.orchestrator import Swarm
from ollama_swarm.tools import ToolRegistry

import pytest


def _role(messages) -> str:
    system = messages[0]["content"]
    if "Security" in system:
        return "security"
    if "Test Engineer" in system:
        return "test_gen"
    if "Governor" in system:
        return "governor"
    if "Architect" in system:
        return "architect"
    if "FinOps" in system:
        return "finops"
    if "Critic" in system:
        return "critic"
    if "Builder" in system:
        return "builder"
    return "planner"


def _make_agents():
    planner   = Agent(name="Planner",      system_prompt="You are the Planner.",           tier=Tier.REASONING)
    scaffolder= Agent(name="Scaffolder",   system_prompt="You are the Scaffolder.",         tier=Tier.CODING)
    architect = Agent(name="Architect",    system_prompt="You are the Architect.",           tier=Tier.REASONING)
    builder   = Agent(name="Builder",      system_prompt="You are the Builder.",             tier=Tier.CODING)
    test_gen  = Agent(name="TestEngineer", system_prompt="You are the Test Engineer.",       tier=Tier.CODING)
    critic    = Agent(name="Critic",       system_prompt="You are the Critic.",              tier=Tier.REASONING)
    security  = Agent(name="Security",     system_prompt="You are the Security Auditor.",    tier=Tier.REASONING)
    governor  = Agent(name="Governor",     system_prompt="You are the Governor.",            tier=Tier.REASONING)
    finops    = Agent(name="FinOps",       system_prompt="You are the FinOps agent.",        tier=Tier.FAST)
    return planner, scaffolder, architect, builder, test_gen, critic, security, governor, finops


@pytest.fixture(autouse=True)
def patch_scaffold(tmp_path):
    """Prevent scaffold_project from touching the real filesystem."""
    orig = SETTINGS.workspace_root
    SETTINGS.workspace_root = str(tmp_path)
    with patch("ollama_swarm.scaffold.scaffold_project", return_value={"README.md": "# test"}), \
         patch("ollama_swarm.scaffold.detect_language", return_value="python"):
        yield
    SETTINGS.workspace_root = orig


def _make_swarm(backend, **kwargs):
    planner, scaffolder, architect, builder, test_gen, critic, security, governor, finops = _make_agents()
    return Swarm(
        planner=planner,
        scaffolder=scaffolder,
        architect=architect,
        builder=builder,
        test_gen=test_gen,
        critic=critic,
        security=security,
        governor=governor,
        finops=finops,
        backend=backend,
        registry=ToolRegistry(),
        **kwargs,
    )


def test_swarm_full_happy_path_runs_all_nine_phases_in_order(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        content = {
            "planner":   "1. do the thing",
            "architect": "design notes",
            "builder":   "solution v1",
            "test_gen":  "def test_foo(): pass",
            "critic":    "APPROVE",
            "security":  "SECURITY: GO",
            "governor":  "GOVERN: GO",
            "finops":    "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    swarm = _make_swarm(backend)
    result = swarm.run("ship the feature")

    assert result.approved is True
    assert result.governed is True
    assert result.rework_count == 0
    assert result.governor_rework_count == 0
    assert result.finops_summary == "cost summary"
    phases = [r.phase for r in result.history]
    assert "PLAN" in phases
    assert "SCAFFOLD" in phases
    assert "ARCHITECT" in phases
    assert "IMPLEMENT" in phases
    assert "TEST-GEN" in phases
    assert "REVIEW" in phases
    assert "SECURITY" in phases
    assert "GOVERN" in phases
    assert "OPERATE" in phases


def test_swarm_governor_no_go_once_then_go(fake_backend) -> None:
    governor_calls = {"n": 0}

    def script(messages):
        role = _role(messages)
        if role == "governor":
            governor_calls["n"] += 1
            if governor_calls["n"] == 1:
                return {"message": {"content": "GOVERN: NO-GO: missing test coverage"}}
            return {"message": {"content": "GOVERN: GO"}}
        content = {
            "planner":   "1. do the thing",
            "architect": "design notes",
            "builder":   "solution",
            "test_gen":  "tests",
            "critic":    "APPROVE",
            "security":  "SECURITY: GO",
            "finops":    "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    swarm = _make_swarm(backend)
    result = swarm.run("ship the feature")

    assert result.approved is True
    assert result.governed is True
    assert result.governor_rework_count == 1
    phases = [r.phase for r in result.history]
    assert phases.count("GOVERN") == 2


def test_swarm_governor_no_go_exhausts_rework_budget_but_operate_still_runs(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        if role == "governor":
            return {"message": {"content": "GOVERN: NO-GO: still failing gates"}}
        content = {
            "planner":   "1. do the thing",
            "architect": "design notes",
            "builder":   "solution",
            "test_gen":  "tests",
            "critic":    "APPROVE",
            "security":  "SECURITY: GO",
            "finops":    "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    swarm = _make_swarm(backend, max_governor_rework=1)
    result = swarm.run("an impossible goal")

    assert result.approved is True
    assert result.governed is False
    phases = [r.phase for r in result.history]
    assert "OPERATE" in phases
    assert result.finops_summary == "cost summary"


def test_swarm_critic_never_approves_skips_govern_but_operate_still_runs(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        if role == "critic":
            return {"message": {"content": "REQUEST_CHANGES: still not good enough"}}
        if role == "governor":
            return {"message": {"content": "GOVERN: GO"}}
        return {"message": {"content": f"content for {role}"}}

    backend = fake_backend(script)
    swarm = _make_swarm(backend, max_rework=1)
    result = swarm.run("an impossible goal")

    assert result.approved is False
    assert result.governed is False
    assert result.rework_count == 2  # exceeds max_rework=1, loop terminates
    phases = [r.phase for r in result.history]
    assert "GOVERN" not in phases
    assert "OPERATE" in phases


def test_swarm_security_no_go_skips_govern_but_operate_still_runs(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        if role == "security":
            return {"message": {"content": "SECURITY: NO-GO: eval() on untrusted input"}}
        if role == "critic":
            return {"message": {"content": "APPROVE"}}
        if role == "governor":
            return {"message": {"content": "GOVERN: GO"}}
        return {"message": {"content": f"content for {role}"}}

    backend = fake_backend(script)
    swarm = _make_swarm(backend)
    result = swarm.run("an insecure goal")

    assert result.approved is True
    assert result.governed is False
    phases = [r.phase for r in result.history]
    assert phases.count("SECURITY") == 2  # one fix-and-rescan attempt, still NO-GO
    assert "GOVERN" not in phases
    assert "OPERATE" in phases


def test_swarm_security_ambiguous_verdict_fails_closed(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        if role == "security":
            return {"message": {"content": "I need to know which workspace to scan."}}
        if role == "critic":
            return {"message": {"content": "APPROVE"}}
        if role == "governor":
            return {"message": {"content": "GOVERN: GO"}}
        return {"message": {"content": f"content for {role}"}}

    backend = fake_backend(script)
    swarm = _make_swarm(backend)
    result = swarm.run("a goal with a confused security agent")

    assert result.governed is False
    phases = [r.phase for r in result.history]
    assert "GOVERN" not in phases


def test_swarm_writes_a_run_summary_to_memory(tmp_path, fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        content = {
            "planner":   "plan",
            "architect": "design notes",
            "builder":   "solution",
            "test_gen":  "tests",
            "critic":    "APPROVE",
            "security":  "SECURITY: GO",
            "governor":  "GOVERN: GO",
            "finops":    "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    memory = Memory(backend, db_path=str(tmp_path / "mem.db"))
    swarm = _make_swarm(backend, memory=memory)

    swarm.run("remember this goal")

    recalled = memory.recall("remember this goal", tag="run_summary")
    assert len(recalled) == 1
    assert "approved and governed" in recalled[0].text


def test_swarm_survives_memory_db_deleted_mid_run(tmp_path, fake_backend) -> None:
    # Live runs showed the Builder can delete the memory DB while "cleaning up"
    # the workspace; a completed run must still be returned, not crash on the
    # final run-summary write.
    db_path = tmp_path / "mem.db"

    def script(messages):
        role = _role(messages)
        if role == "builder":
            db_path.unlink(missing_ok=True)
        content = {
            "planner":   "plan",
            "architect": "design notes",
            "builder":   "solution",
            "test_gen":  "tests",
            "critic":    "APPROVE",
            "security":  "SECURITY: GO",
            "governor":  "GOVERN: GO",
            "finops":    "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    memory = Memory(backend, db_path=str(db_path))
    swarm = _make_swarm(backend, memory=memory)

    result = swarm.run("a goal whose workspace gets wiped")

    assert result.approved is True
    assert result.governed is True
    assert result.finops_summary == "cost summary"
