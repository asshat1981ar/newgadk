from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from .backend import OllamaBackend
from .config import SETTINGS, MODEL_CATALOG, Tier, models_for
from .memory import Memory
from .orchestrator import Swarm
from .presets import default_registry, default_swarm_agents

_CONFIG_HAS_WORKSPACE_ROOT = False

def load_config(config_path: str | None) -> None:
    """Loads a JSON configuration file and updates SETTINGS and MODEL_CATALOG in-place."""
    if config_path is None:
        return

    global _CONFIG_HAS_WORKSPACE_ROOT
    _CONFIG_HAS_WORKSPACE_ROOT = False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        if not isinstance(config_data, dict):
            raise TypeError("JSON config must be a dictionary")

        # Update SETTINGS in-place
        if "mode" in config_data:
            SETTINGS.mode = config_data["mode"]
        if "host" in config_data:
            SETTINGS.host = config_data["host"]
        if "cloud_host" in config_data:
            SETTINGS.cloud_host = config_data["cloud_host"]
        if "api_key" in config_data:
            SETTINGS.api_key = config_data["api_key"]
        if "timeout" in config_data:
            SETTINGS.timeout_s = float(config_data["timeout"])
        if "timeout_s" in config_data:
            SETTINGS.timeout_s = float(config_data["timeout_s"])
        if "max_tool_turns" in config_data:
            SETTINGS.max_tool_turns = int(config_data["max_tool_turns"])
        if "output_dir" in config_data:
            SETTINGS.workspace_root = config_data["output_dir"]
        if "workspace_root" in config_data:
            SETTINGS.workspace_root = config_data["workspace_root"]

        if "output_dir" in config_data or "workspace_root" in config_data:
            _CONFIG_HAS_WORKSPACE_ROOT = True

        # Update MODEL_CATALOG in-place
        catalog_data = (
            config_data.get("model_catalog")
            or config_data.get("models")
            or config_data.get("MODEL_CATALOG")
        )
        if catalog_data is not None:
            if not isinstance(catalog_data, dict):
                raise TypeError("models/model_catalog must be a dictionary")
            for tier_str, models_list in catalog_data.items():
                if not isinstance(models_list, (list, tuple)):
                    raise TypeError(f"Model list for tier '{tier_str}' must be a list/sequence")
                # Match Tier enum
                try:
                    matching_tier = next(t for t in Tier if t.value.lower() == tier_str.lower())
                    MODEL_CATALOG[matching_tier] = list(models_list)
                except (StopIteration, ValueError):
                    try:
                        tier_enum = Tier(tier_str.lower())
                        MODEL_CATALOG[tier_enum] = list(models_list)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}", file=sys.stderr)
        raise

def parse_args(sys_args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama Developer Assistant CLI")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--mode", help="Ollama mode (daemon or direct-cloud)")
    parser.add_argument("--host", help="Ollama host")
    parser.add_argument("--cloud-host", help="Ollama cloud host")
    parser.add_argument("--api-key", help="Ollama API key")
    parser.add_argument("--output-dir", help="Output directory / workspace root")
    parser.add_argument("--timeout", type=float, help="Timeout in seconds")
    parser.add_argument("--max-tool-turns", type=int, help="Max tool turns")
    parser.add_argument("--no-dev-tools", action="store_true", help="Disable developer tools")
    parser.add_argument("goal", nargs="*", help="Software goal to achieve")
    return parser.parse_args(sys_args)

def interactive_clarification(backend: OllamaBackend, idea: str, num_questions: int = 3) -> str:
    # Query a Tier.REASONING model to ask exactly clarifying questions.
    reasoning_models = models_for(Tier.REASONING)
    prompt_q = (
        f"You are a developer assistant. The user wants to build: '{idea}'.\n"
        f"Ask exactly {num_questions} high-impact clarifying questions to refine this software goal.\n"
        f"Respond with only the {num_questions} questions, one per line (numbered 1, 2, 3), and nothing else."
    )
    _, chat_res = backend.chat_with_fallback(
        models=reasoning_models,
        messages=[{"role": "user", "content": prompt_q}]
    )
    questions_text = chat_res["message"]["content"].strip()
    
    # Split into individual questions
    raw_lines = [line.strip() for line in questions_text.splitlines() if line.strip()]
    questions = []
    _prefix_re = re.compile(r'^\s*(?:\d+[\.\s\-*:\t]+|[-*•][\s\-*:\t]*)\s*')
    for line in raw_lines:
        # Apply stripping in a loop to handle compound prefixes like "- 3."
        cleaned = line
        while True:
            stripped = _prefix_re.sub('', cleaned)
            if stripped == cleaned:
                break
            cleaned = stripped
        if cleaned:
            questions.append(cleaned)
    
    # Ensure we have exactly num_questions questions
    questions = questions[:num_questions]
    while len(questions) < num_questions:
        questions.append("What other features or requirements should we consider?")

    answers = []
    try:
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}: {q}")
            ans = input("Answer: ").strip()
            answers.append(ans)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...", file=sys.stderr)
        sys.exit(0)

    # Query the model to synthesize a detailed goal
    synthesis_prompt = (
        f"Initial software idea: {idea}\n\n"
        f"Clarifying questions and answers:\n"
    )
    for q, a in zip(questions, answers):
        synthesis_prompt += f"Question: {q}\nAnswer: {a}\n\n"
    synthesis_prompt += (
        "Synthesize a detailed, concrete software development goal from the above.\n"
        "Provide the synthesized goal clearly, directly, and in detail."
    )

    _, synthesis_res = backend.chat_with_fallback(
        models=reasoning_models,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )
    return synthesis_res["message"]["content"].strip()

def main() -> None:
    # 1. Parse CLI arguments
    args = parse_args(sys.argv[1:])

    # 2. Config loading (if CLI specifies --config)
    # Precedence: CLI args > JSON config > Env vars > Dataclass defaults
    # Reset the global flag so each main() call starts clean.
    global _CONFIG_HAS_WORKSPACE_ROOT
    _CONFIG_HAS_WORKSPACE_ROOT = False
    try:
        load_config(args.config)
    except Exception:
        sys.exit(1)

    # 3. Apply CLI arguments override in-place to SETTINGS
    if args.mode is not None:
        SETTINGS.mode = args.mode
    if args.host is not None:
        SETTINGS.host = args.host
    if args.cloud_host is not None:
        SETTINGS.cloud_host = args.cloud_host
    if args.api_key is not None:
        SETTINGS.api_key = args.api_key
    if args.timeout is not None:
        SETTINGS.timeout_s = float(args.timeout)
    if args.max_tool_turns is not None:
        SETTINGS.max_tool_turns = int(args.max_tool_turns)

    # Handle output_dir precedence and default
    if args.output_dir is not None:
        SETTINGS.workspace_root = args.output_dir
    else:
        # Check if workspace_root has been updated. If it is still the dataclass default "./workspace"
        # and has not been overridden by JSON config, set it to the default.
        if SETTINGS.workspace_root == "./workspace" and not _CONFIG_HAS_WORKSPACE_ROOT:
            SETTINGS.workspace_root = "~/teamwork_projects/ollama_developer_assistant"

    # Workspace isolation: update SETTINGS.workspace_root with the absolute path
    SETTINGS.workspace_root = os.path.abspath(os.path.expanduser(SETTINGS.workspace_root))
    os.makedirs(SETTINGS.workspace_root, exist_ok=True)

    # Swarm execution setup: unless --no-dev-tools is specified, set OS env
    if args.no_dev_tools:
        os.environ["OLLAMA_SWARM_ENABLE_DEV_TOOLS"] = "0"
    else:
        if "OLLAMA_SWARM_ENABLE_DEV_TOOLS" not in os.environ:
            os.environ["OLLAMA_SWARM_ENABLE_DEV_TOOLS"] = "1"

    # Create backend
    backend = OllamaBackend()

    # Determine goal
    goal = " ".join(args.goal).strip() if args.goal else ""

    # 4. Interactive prompt loop (if no positional goal is passed)
    if not goal:
        try:
            idea = input("Enter your initial software idea: ").strip()
            while not idea:
                idea = input("Enter your initial software idea (cannot be empty): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...", file=sys.stderr)
            sys.exit(0)

        goal = interactive_clarification(backend, idea, num_questions=3)
        print(f"\nSynthesized Goal:\n{goal}\n")

    # 5. Swarm pipeline execution
    registry = default_registry()
    agents = default_swarm_agents()
    # The memory DB must live outside workspace_root: agents have full write
    # access inside the workspace and can (and did, in live runs) delete an
    # open DB file while "cleaning up" their deliverable.
    memory_db = os.path.join(
        os.path.dirname(SETTINGS.workspace_root),
        os.path.basename(SETTINGS.workspace_root) + "_memory.db",
    )
    memory = Memory(backend, db_path=memory_db)

    swarm = Swarm(
        planner=agents.planner,
        scaffolder=agents.scaffolder,
        architect=agents.architect,
        builder=agents.builder,
        test_gen=agents.test_gen,
        critic=agents.critic,
        security=agents.security,
        governor=agents.governor,
        finops=agents.finops,
        backend=backend,
        registry=registry,
        memory=memory,
    )

    result = swarm.run(goal)

    # Print history
    for record in result.history:
        print(f"\n=== {record.phase} ({record.agent} via {record.model_used}) ===")
        print(record.content)

    # Print security verdict if present
    if result.security_verdict:
        print(f"\nSecurity Verdict: {result.security_verdict}")

    # Print governor verdict
    print(f"\nGovernor Verdict: {'GO' if result.governed else 'NO-GO'}")

    # Print FinOps summary
    print(f"\nFinOps Summary:\n{result.finops_summary}")

if __name__ == "__main__":
    main()
