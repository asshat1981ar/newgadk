from __future__ import annotations

from ollama_swarm.memory import Memory
from ollama_swarm.orchestrator import Swarm
from ollama_swarm.presets import default_registry, default_swarm_agents


def _role(messages) -> str:
    system = messages[0]["content"]
    if "Planner" in system:
        return "planner"
    if "Builder" in system:
        return "builder"
    return "critic"


def test_swarm_approves_on_first_pass(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        return {"message": {"content": {"planner": "1. do the thing", "builder": "solution v1", "critic": "APPROVE"}[role]}}

    backend = fake_backend(script)
    planner, builder, critic = default_swarm_agents()
    swarm = Swarm(planner, builder, critic, backend, default_registry())

    result = swarm.run("ship the feature")

    assert result.approved is True
    assert result.rework_count == 0
    phases = [r.phase for r in result.history]
    assert phases == ["PLAN", "BUILD", "REVIEW"]


def test_swarm_reworks_once_then_approves(fake_backend) -> None:
    critic_calls = {"n": 0}

    def script(messages):
        role = _role(messages)
        if role == "planner":
            return {"message": {"content": "1. do the thing"}}
        if role == "builder":
            return {"message": {"content": "solution v1" if critic_calls["n"] == 0 else "solution v2"}}
        critic_calls["n"] += 1
        return {"message": {"content": "APPROVE" if critic_calls["n"] > 1 else "REQUEST_CHANGES: add tests"}}

    backend = fake_backend(script)
    planner, builder, critic = default_swarm_agents()
    swarm = Swarm(planner, builder, critic, backend, default_registry())

    result = swarm.run("ship the feature")

    assert result.approved is True
    assert result.rework_count == 1
    phases = [r.phase for r in result.history]
    assert phases == ["PLAN", "BUILD", "REVIEW", "BUILD", "REVIEW"]


def test_swarm_stops_after_max_rework_without_looping_forever(fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        if role == "critic":
            return {"message": {"content": "REQUEST_CHANGES: still not good enough"}}
        return {"message": {"content": f"content for {role}"}}

    backend = fake_backend(script)
    planner, builder, critic = default_swarm_agents()
    swarm = Swarm(planner, builder, critic, backend, default_registry(), max_rework=1)

    result = swarm.run("an impossible goal")

    assert result.approved is False
    assert result.rework_count == 2  # exceeds max_rework=1, loop terminates


def test_swarm_writes_a_run_summary_to_memory(tmp_path, fake_backend) -> None:
    def script(messages):
        role = _role(messages)
        return {"message": {"content": {"planner": "plan", "builder": "solution", "critic": "APPROVE"}[role]}}

    backend = fake_backend(script)
    memory = Memory(backend, db_path=str(tmp_path / "mem.db"))
    planner, builder, critic = default_swarm_agents()
    swarm = Swarm(planner, builder, critic, backend, default_registry(), memory=memory)

    swarm.run("remember this goal")

    recalled = memory.recall("remember this goal", tag="run_summary")
    assert len(recalled) == 1
    assert "approved" in recalled[0].text
