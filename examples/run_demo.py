"""Live demo: runs the swarm against a real Ollama daemon (local or cloud-proxied).

    python examples/run_demo.py
"""

from __future__ import annotations

from ollama_swarm.backend import OllamaBackend
from ollama_swarm.memory import Memory
from ollama_swarm.orchestrator import Swarm
from ollama_swarm.presets import default_registry, default_swarm_agents

GOAL = "Write a one-paragraph explanation of why bounded retry loops matter in agent systems."


def main() -> None:
    backend = OllamaBackend()
    registry = default_registry()
    planner, builder, critic = default_swarm_agents()
    memory = Memory(backend)

    swarm = Swarm(planner, builder, critic, backend, registry, memory=memory, max_rework=1)
    result = swarm.run(GOAL)

    for record in result.history:
        print(f"\n=== {record.phase} ({record.agent} via {record.model_used}) ===")
        print(record.content)

    print(f"\n--- {'APPROVED' if result.approved else 'UNRESOLVED'} after {result.rework_count} rework cycle(s) ---")


if __name__ == "__main__":
    main()
