"""python -m ollama_swarm.cli "<goal>" """

from __future__ import annotations

import sys

from .backend import OllamaBackend
from .memory import Memory
from .orchestrator import Swarm
from .presets import default_registry, default_swarm_agents


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m ollama_swarm.cli "<goal>"')
        raise SystemExit(1)

    goal = " ".join(sys.argv[1:])
    backend = OllamaBackend()
    registry = default_registry()
    agents = default_swarm_agents()
    memory = Memory(backend)

    swarm = Swarm(
        agents.planner,
        agents.architect,
        agents.builder,
        agents.critic,
        agents.governor,
        agents.finops,
        backend,
        registry,
        memory=memory,
    )
    result = swarm.run(goal)

    for record in result.history:
        print(f"\n=== {record.phase} ({record.agent} via {record.model_used}) ===")
        print(record.content)

    verdict = "APPROVED" if result.approved else "UNRESOLVED"
    governed = "governed" if result.governed else "ungoverned"
    print(f"\n--- {verdict} ({governed}) after {result.rework_count} critic + {result.governor_rework_count} governor rework cycle(s) ---")


if __name__ == "__main__":
    main()
