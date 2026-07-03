"""Live demo: runs the full 6-phase swarm against a real Ollama daemon.

Enables the opt-in dev tools and uses a goal that exercises them, so this run
proves out write_file/run_shell/run_quality_gates end-to-end — not just chat.

    OLLAMA_SWARM_ENABLE_DEV_TOOLS=1 python examples/run_demo.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OLLAMA_SWARM_ENABLE_DEV_TOOLS", "1")

from ollama_swarm.backend import OllamaBackend
from ollama_swarm.memory import Memory
from ollama_swarm.orchestrator import Swarm
from ollama_swarm.presets import default_registry, default_swarm_agents

GOAL = (
    "In the workspace, write a file hello.py containing a single line that prints "
    "the word hi, then run it with run_shell to confirm it works."
)


def main() -> None:
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
        max_rework=1,
        max_governor_rework=1,
    )
    result = swarm.run(GOAL)

    for record in result.history:
        print(f"\n=== {record.phase} ({record.agent} via {record.model_used}) ===")
        print(record.content)

    verdict = "APPROVED" if result.approved else "UNRESOLVED"
    governed = "governed" if result.governed else "ungoverned"
    print(f"\n--- {verdict} ({governed}) after {result.rework_count} critic + {result.governor_rework_count} governor rework cycle(s) ---")


if __name__ == "__main__":
    main()
