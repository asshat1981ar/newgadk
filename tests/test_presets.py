from __future__ import annotations

from ollama_swarm.config import Tier
from ollama_swarm.presets import SwarmAgents, default_registry, default_swarm_agents


def test_default_swarm_agents_populates_all_six_roles_with_correct_tiers() -> None:
    agents = default_swarm_agents()

    assert isinstance(agents, SwarmAgents)
    assert agents.planner.tier == Tier.REASONING
    assert agents.architect.tier == Tier.REASONING
    assert agents.builder.tier == Tier.CODING
    assert agents.critic.tier == Tier.REASONING
    assert agents.governor.tier == Tier.REASONING
    assert agents.finops.tier == Tier.FAST


def test_governor_has_exactly_one_narrow_tool() -> None:
    agents = default_swarm_agents()
    assert agents.governor.tools == ["run_quality_gates"]


def test_finops_has_no_tools() -> None:
    agents = default_swarm_agents()
    assert agents.finops.tools == []


def test_architect_has_read_only_workspace_tools() -> None:
    agents = default_swarm_agents()
    assert agents.architect.tools == ["list_dir", "read_file"]


def test_default_registry_always_registers_run_quality_gates(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_SWARM_ENABLE_DEV_TOOLS", raising=False)
    registry = default_registry()
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert "run_quality_gates" in names


def test_dev_tools_registered_only_when_env_flag_set(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_SWARM_ENABLE_DEV_TOOLS", "1")
    registry = default_registry()
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert {"write_file", "run_shell", "read_file", "list_dir", "git_diff", "git_commit"} <= names


def test_dev_tools_absent_when_env_flag_unset(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_SWARM_ENABLE_DEV_TOOLS", raising=False)
    registry = default_registry()
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert "write_file" not in names
    assert "run_shell" not in names
