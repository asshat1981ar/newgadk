# Ollama Developer Assistant CLI - E2E Testing Infrastructure

This document outlines the architecture, design, and execution strategy for the End-to-End (E2E) test suite of the Ollama Developer Assistant CLI.

---

## 1. Architecture Overview

The E2E test suite sits at the top of the test pyramid, validating the integration of all components: the CLI entry point, the Swarm orchestrator, the 6-agent lifecycle, the Tool Registry, the local file/subprocess tools (sandbox), SQLite-based memory recall, and the connection adapter/router.

Since the assistant relies on an external LLM provider (Ollama local daemon or Ollama Cloud endpoint), the E2E suite requires mechanisms to run deterministically, without network requests, and with full control over model responses.

---

## 2. Backend Mocking Strategies

To achieve full coverage and reliability under network-isolated test runners, we support two complementary mocking strategies:

### Strategy A: In-Process CLI Mocking (Fast, Integrated)
Suitable for rapid testing of logic flows, agent rework loop dynamics, and router fallbacks.
- **Mechanism**: The tests invoke `cli.main()` directly within the pytest process.
- **Interception**: Pytest's `monkeypatch` redirects `sys.argv` to simulate command-line arguments. It patches `ollama.Client.chat` and `ollama.Client.embeddings` to return scripted or dynamic mock responses.
- **Pros**: Direct memory state assertions, extremely fast execution, co-located code.
- **Cons**: Does not test Python process startup, environment variable parsing by subprocesses, or real HTTP-level serialization.

### Strategy B: Subprocess CLI Mocking (True Black-Box)
Suitable for validating CLI arguments, exit codes, output formatting, signal handling, and workspace setups.
- **Mechanism**: The tests invoke the CLI in a real subprocess: `python3 -m ollama_swarm.cli "<goal>"`.
- **Interception**: A pytest fixture spins up a lightweight mock HTTP server in a background thread (running on a random local port, e.g., `http://localhost:54321`). The test sets `OLLAMA_HOST="http://localhost:54321"` and `OLLAMA_MODE="daemon"` (or `OLLAMA_CLOUD_HOST` for `direct-cloud`). The CLI subprocess then hits the local mock HTTP server.
- **Pros**: 100% realistic environment, validates shell-level operations, tests HTTP headers (e.g., Bearer auth keys).
- **Cons**: Slower execution due to process spawn and HTTP socket overhead.

---

## 3. Simulating User Interactive Prompts

Currently, the CLI takes the user's objective via arguments (e.g. `sys.argv[1:]`). In the future, if interactive prompts are added (such as `input("Proceed? [y/N]")` or multi-turn chats):
- **In-process simulation**: Use `monkeypatch` to override `builtins.input` with a callable that pops strings from a predefined list.
- **Subprocess simulation**: Use `subprocess.Popen` with `stdin=subprocess.PIPE` and write inputs into the write-pipe, or use a tool like `pexpect` for complex terminal interactions.

---

## 4. Test Tier Structure

The E2E suite covers 60 test cases grouped into four tiers:

### Tier 1: Feature Coverage (25 tests)
Validates core functionality for each of the 5 main features:
1. **Multi-Agent Phase Pipeline**: Verify sequential flow (PLAN -> ARCHITECT -> IMPLEMENT -> REVIEW -> GOVERN -> OPERATE) and ledger compiling.
2. **Critic Rework Loop**: Bounded review cycles, transition on `APPROVE` or `REQUEST_CHANGES`, and feedback injection.
3. **Quality Gates**: `run_quality_gates` invocation, test output evaluation, governor NO-GO reworks, and Ruff execution.
4. **Memory / RAG**: Storage of run summaries (`run_summary`) and recall injection to Planner.
5. **Model Router & Fallback**: Routing healthy-first, model failures fallback, and model demotion.

### Tier 2: Boundary & Corner Cases (25 tests)
Tests extremes and failure modes:
- Empty/ultra-long goal strings.
- Rework budgets (`max_rework`, `max_governor_rework`) set to 0.
- Max tool turns limit hit during Builder implementation.
- Lockups/permissions errors on SQLite database files.
- Missing tool definitions or ruff linters missing from path.
- Connection/network socket timeouts.

### Tier 3: Cross-Feature Combinations (5 tests)
Validates interactions between features (e.g., how the router behaves when memory embeddings fail; how sandbox failures are routed to the Critic).

### Tier 4: Real-World Application Scenarios (5 tests)
Validates holistic user developer flows (e.g., automated bug fixing, fixing syntax/lint errors, malicious prompt blocking).

---

## 5. Environment & Execution

Run E2E tests with:
```bash
# Run only E2E tests
pytest tests/test_assistant_e2e.py -v

# Run with stdout printing for debugging
pytest tests/test_assistant_e2e.py -s -v
```

### Configurable Env Variables
- `OLLAMA_SWARM_ENABLE_DEV_TOOLS`: Set to `1` to enable local filesystem and shell execution tools for the Builder.
- `OLLAMA_SWARM_WORKSPACE`: Configures the sandbox workspace path (defaults to `./workspace`).
