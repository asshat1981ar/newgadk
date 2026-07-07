# Project: Ollama Developer Assistant CLI

## Architecture
The Ollama Developer Assistant CLI is built as an extension to `ollama-swarm`. It wraps the multi-agent orchestration layer (`Swarm`, `OllamaBackend`, `Agent`) in an interactive prompt loop that refines a user's initial software idea before triggering the SDLC generation pipeline.

### Component Design
1. **Interactive Prompt Loop (PM Agent)**:
   - Prompts the user for their software idea.
   - Queries a lightweight model (using `FAST` or `REASONING` tier) to formulate 2-3 high-impact clarifying questions.
   - Gathers answers from the user interactively.
   - Synthesizes the original idea and answers into a detailed, structured goal.
2. **Configuration Loader**:
   - Loads connection settings (endpoints, API keys, models) from environment variables or a specified JSON configuration file.
   - Integrates with the existing `Settings` dataclass by overriding `SETTINGS` values at runtime.
3. **Workspace Sanitization & Isolation**:
   - Configures the output directory via CLI arguments (or defaults to `~/teamwork_projects/ollama_developer_assistant`).
   - Ensures `SETTINGS.workspace_root` is updated to prevent directory traversal and constrain agent tool usage.
4. **Orchestrator Swarm Invocation**:
   - Initiates the `Swarm` pipeline (`PLAN -> ARCHITECT -> IMPLEMENT -> REVIEW -> GOVERN -> OPERATE`).
   - Executes standard linting/testing quality gates inside the generated project folder.

### Code Layout
- `src/ollama_swarm/assistant.py`: The CLI module containing prompt logic, configuration parsing, and the main entrypoint.
- `tests/test_assistant.py`: Automated pytest suite mocking stdin/stdout and the LLM backend to verify CLI behavior.
- `verify.py`: Verification script running simulated inputs to compile and generate real outputs.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | E2E Testing Track | Design E2E test suite, write TEST_INFRA.md, publish TEST_READY.md | none | DONE |
| M2 | CLI & Config | Add CLI skeleton, arg parsing, JSON config file loading, and workspace isolation | none | DONE |
| M3 | Clarification Loop | Implement interactive prompt loop & clarifying questions flow using OllamaBackend | M2 | DONE |
| M4 | Swarm Generation | Connect the synthesized goal to the Swarm execution loop | M3 | DONE |
| M5 | Testing & Packaging | Add pytest coverage for CLI logic; configure entry point in pyproject.toml | M4 | DONE |
| M6 | Verification Script | Create `verify.py` to end-to-end run the assistant with simulated inputs and verify generation | M5 | DONE |
| M7 | Adversarial Hardening | white-box coverage hardening (Tier 5) | M6, M1 | DONE |

## Interface Contracts
### `assistant.py` ↔ `config.py`
- `load_config(config_path: str | None) -> None`: Overwrites fields in `SETTINGS` (e.g. `mode`, `host`, `api_key`) from JSON configuration file.
- Command-line arguments override both the configuration file and environment variables.

### `assistant.py` ↔ `OllamaBackend`
- `interactive_clarification(backend: OllamaBackend, idea: str, num_questions: int = 3) -> str`: Streams/runs a chat prompt to elicit clarifying questions and returns the synthesized goal.
