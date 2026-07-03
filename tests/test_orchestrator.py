from __future__ import annotations

from ollama_swarm.agents import Agent
from ollama_swarm.config import Tier
from ollama_swarm.memory import Memory
from ollama_swarm.orchestrator import Swarm
from ollama_swarm.tools import ToolRegistry


def _role(messages) -> str:
    system = messages[0]["content"]
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


def _make_agents() -> tuple[Agent, Agent, Agent, Agent, Agent, Agent]:
    planner = Agent(name="Planner", system_prompt="You are the Planner.", tier=Tier.REASONING)
    architect = Agent(name="Architect", system_prompt="You are the Architect.", tier=Tier.REASONING)
    builder = Agent(name="Builder", system_prompt="You are the Builder.", tier=Tier.CODING)
    critic = Agent(name="Critic", system_prompt="You are the Critic.", tier=Tier.REASONING)
    governor = Agent(name="Governor", system_prompt="You are the Governor.", tier=Tier.REASONING)
    finops = Agent(name="FinOps", system_prompt="You are the FinOps agent.", tier=Tier.FAST)
    return planner, architect, builder, critic, governor, finops


def test_swarm_full_happy_path_runs_all_six_phases_in_order(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        content = {
            "planner": "1. do the thing",
            "architect": "design notes",
            "builder": "solution v1",
            "critic": "APPROVE",
            "governor": "GOVERN: GO",
            "finops": "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    planner, architect, builder, critic, governor, finops = _make_agents()
    swarm = Swarm(planner, architect, builder, critic, governor, finops, backend, ToolRegistry())

    result = swarm.run("ship the feature")

    assert result.approved is True
    assert result.governed is True
    assert result.rework_count == 0
    assert result.governor_rework_count == 0
    assert result.finops_summary == "cost summary"
    phases = [r.phase for r in result.history]
    assert phases == ["PLAN", "ARCHITECT", "IMPLEMENT", "REVIEW", "GOVERN", "OPERATE"]


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
            "planner": "1. do the thing",
            "architect": "design notes",
            "builder": "solution",
            "critic": "APPROVE",
            "finops": "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    planner, architect, builder, critic, governor, finops = _make_agents()
    swarm = Swarm(planner, architect, builder, critic, governor, finops, backend, ToolRegistry())

    result = swarm.run("ship the feature")

    assert result.approved is True
    assert result.governed is True
    assert result.governor_rework_count == 1
    phases = [r.phase for r in result.history]
    assert phases == [
        "PLAN", "ARCHITECT", "IMPLEMENT", "REVIEW", "GOVERN",
        "IMPLEMENT", "REVIEW", "GOVERN", "OPERATE",
    ]
    assert phases.count("GOVERN") == 2


def test_swarm_governor_no_go_exhausts_rework_budget_but_operate_still_runs(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        if role == "governor":
            return {"message": {"content": "GOVERN: NO-GO: still failing gates"}}
        content = {
            "planner": "1. do the thing",
            "architect": "design notes",
            "builder": "solution",
            "critic": "APPROVE",
            "finops": "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    planner, architect, builder, critic, governor, finops = _make_agents()
    swarm = Swarm(
        planner, architect, builder, critic, governor, finops, backend, ToolRegistry(),
        max_governor_rework=1,
    )

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
    planner, architect, builder, critic, governor, finops = _make_agents()
    swarm = Swarm(
        planner, architect, builder, critic, governor, finops, backend, ToolRegistry(),
        max_rework=1,
    )

    result = swarm.run("an impossible goal")

    assert result.approved is False
    assert result.governed is False
    assert result.rework_count == 2  # exceeds max_rework=1, loop terminates
    phases = [r.phase for r in result.history]
    assert "GOVERN" not in phases
    assert "OPERATE" in phases


def test_swarm_writes_a_run_summary_to_memory(tmp_path, fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        content = {
            "planner": "plan",
            "architect": "design notes",
            "builder": "solution",
            "critic": "APPROVE",
            "governor": "GOVERN: GO",
            "finops": "cost summary",
        }[role]
        return {"message": {"content": content}}

    backend = fake_backend(script)
    memory = Memory(backend, db_path=str(tmp_path / "mem.db"))
    planner, architect, builder, critic, governor, finops = _make_agents()
    swarm = Swarm(planner, architect, builder, critic, governor, finops, backend, ToolRegistry(), memory=memory)

    swarm.run("remember this goal")

    recalled = memory.recall("remember this goal", tag="run_summary")
    assert len(recalled) == 1
    assert "approved and governed" in recalled[0].text
