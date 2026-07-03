# ollama-swarm

A reimagining of [GADK](https://github.com/asshat1981ar/gadk)'s AI orchestration layer,
rebuilt natively around Ollama instead of layered on top of it.

GADK is a real, working multi-agent SDLC system (8 agents, a 6-phase gated pipeline,
already migrated off OpenRouter onto Ollama Cloud models). It has, however, accreted a
lot of framework surface around that core idea: DSPy modules, an optional DBOS workflow
engine, optional LangGraph graphs, an optional Memori memory layer, a `google-adk`
dependency gated behind `TEST_MODE` in every agent file, and — the specific thing that
triggered this project — **three separate, overlapping implementations of "talk to
Ollama"**: `src/services/ollama_cloud_backend.py` (cloud-only, direct Bearer auth),
`ModelRouter.get_backend()` (assumes the same shape for `ollama/`-prefixed strings),
and a sketched-but-unmerged `OllamaBackend` in `docs/plans/2026-04-26-ollama-cloud-agents.md`.
Its model-capability table is also duplicated verbatim between `src/config.py` and
`src/services/model_router.py`.

This project keeps the ideas that earn their weight — tiered model routing, fallback
chains, phase-gated agent pipelines with bounded rework, cross-run memory — and drops
everything that exists only to route around a missing abstraction. Total dependency:
the `ollama` package. That's it.

## What changed, and why

| GADK | Here | Why |
|---|---|---|
| `OllamaCloudBackend` + `ModelRouter.get_backend()` + draft `OllamaBackend` (3 implementations) | `backend.py::OllamaBackend` (1 class, `mode="daemon"\|"direct-cloud"`) | A local `ollama serve` already proxies `:cloud` models transparently once `ollama signin` has run — verified live in this sandbox (`kimi-k2.6:cloud` runs through `http://localhost:11434`, no direct cloud call needed). Direct-cloud mode is kept only for daemon-less deploys (containers/serverless). |
| `MODEL_CAPABILITY_MAP` duplicated in `config.py` *and* `model_router.py` | `config.py::MODEL_CATALOG` (one table) | Two copies of the same map drift. This has one. |
| `ModelRouter._classify_task_capability` / `_classify_task_complexity` — regex over free text, e.g. any task description containing "hi" anywhere routes to the cheap/fast tier | `router.py::Router.fallback_chain(tier)` — caller states the tier | The caller (an agent definition) always knows what kind of work it's doing; it's the one writing the prompt. Guessing that back out of natural language is unnecessary and provably brittle. |
| `src/tools/dispatcher.py` + `src/capabilities/{contracts,registry,service}.py` (4 files) | `tools.py::ToolRegistry` (1 file, ~90 lines) | Ollama's `/api/chat` speaks OpenAI-style tool schemas natively (verified live — `tool_calls` come back structured, no custom parsing needed). A registry that derives the JSON schema from a function's type hints and dispatches by name is the whole job. |
| `src/memory/{memory_graph,graph_store}.py` + `src/services/{vector_index,embed_quota,embedder}.py` + optional Memori triple store | `memory.py::Memory` (1 file, SQLite + cosine similarity) | The embedding model is just another Ollama model (`nomic-embed-text`, verified live — 768-dim vectors). At single-swarm scale, brute-force cosine over a SQLite table is correct and needs no vector DB. |
| `sdlc_phase.py` + `phase_controller.py` + `phase_store.py` + `workflow_graphs.py` (4 files, optional LangGraph dependency) | `orchestrator.py::Swarm` (1 file, plain control flow) | The actual invariant — bounded rework, explicit history — is a three-step loop with a retry counter. It doesn't need a graph execution engine. |
| Every agent file: conditional `google.adk.agents.Agent` import gated by `Config.TEST_MODE`, i.e. two agents per agent (real + mock) | `agents.py::Agent` (plain dataclass) + `run_agent()` | No ADK dependency, so no split. Tests substitute a duck-typed fake backend instead of a parallel mock agent implementation. |

## Architecture

```
Agent (name, system_prompt, tier, tools)
   │
   ▼
run_agent()  ──►  Router.fallback_chain(tier)  ──►  OllamaBackend.chat_with_fallback()
   │                                                        │
   │◄────────────────────── tool_calls? ───────────────────┘
   ▼
ToolRegistry.dispatch()  (loop until final answer or max_tool_turns)

Swarm.run(goal):
  PLAN (Planner) ──► ARCHITECT (Architect) ──► IMPLEMENT (Builder) ──► REVIEW (Critic)
                                                     ▲                      │
                                                     └── rework, bounded ────┘  (max_rework, default 2)
                                                                            │ APPROVE
                                                                            ▼
                                    IMPLEMENT ◄── NO-GO, bounded ──── GOVERN (Governor)
                                    (max_governor_rework, default 1)        │ GO (or budget exhausted)
                                                                            ▼
                                                              OPERATE (FinOps — always runs)
```

- **Architect** inspects the workspace read-only (`list_dir`, `read_file`) and writes a
  short design note the Builder must respect.
- **Governor** has exactly one tool, `run_quality_gates`, which actually executes
  `pytest` (and `ruff check` when installed) in the workspace — approval stops being an
  LLM's opinion and becomes a real test run. On NO-GO the work goes back through
  IMPLEMENT and REVIEW before the Governor sees it again.
- **FinOps** always runs last, summarizing a per-phase token ledger built from the
  `prompt_eval_count`/`eval_count` fields every Ollama response already carries.
  (GADK's Pulse role was deliberately *not* ported as an agent — latency/health
  bookkeeping is deterministic, so it lives in `Router`/`AgentRunResult` as plain data.)

Memory is consulted before PLAN (relevant past runs, if any) and written to after
the pipeline finishes (`goal -> outcome` summary), so future runs on similar goals
get context for free.

## Dev tools (opt-in)

Builder can get real filesystem/shell/git capability — `read_file`, `write_file`,
`list_dir`, `run_shell`, `git_diff`, `git_commit` — but only when explicitly enabled:

```bash
OLLAMA_SWARM_ENABLE_DEV_TOOLS=1 python examples/run_demo.py
```

All of these are confined to `Settings.workspace_root` (default `./workspace`, override
with `OLLAMA_SWARM_WORKSPACE`); path traversal out of the workspace raises. **Caveat:**
`run_shell` and `git_commit` execute real subprocesses — `cwd` confinement is the only
sandboxing (no container, no resource limits). Fine for a local single-user tool;
not safe for untrusted goals. The `run_quality_gates` tool is always registered
(it's narrow and the Governor needs it), independent of this flag.

## Verified live in this sandbox

Everything below was run against a real local `ollama serve` (v0.30.7) with a signed-in
cloud account — not mocked:

- Chat completion via `kimi-k2.6:cloud` (`ollama.Client(host="http://localhost:11434")`)
- Native tool-calling (`tools=[...]` → structured `tool_calls` in the response)
- Embeddings via `nomic-embed-text` (pulled locally, 768-dim vectors)
- The full `Swarm` (Planner → Builder → Critic) end-to-end on a real goal — see
  `examples/run_demo.py` output in the session transcript: approved on the first pass,
  routed Planner/Critic to `glm-5:cloud` and Builder to `qwen3-coder-next:cloud`
  per `config.MODEL_CATALOG`, no local pull needed for either (cloud models
  resolve on first use).

## Running it

```bash
pip install -e ".[dev]"
pytest -q                       # 16 tests, all offline (scripted fake backend)

python examples/run_demo.py     # live run against your Ollama daemon
python -m ollama_swarm.cli "explain X"   # same, your own goal
```

Requires a running `ollama serve` (local models work with no signin; `:cloud` models
need `ollama signin` once). Set `OLLAMA_MODE=direct-cloud` + `OLLAMA_API_KEY` to skip
the local daemon entirely and talk straight to `https://ollama.com`.

## What's deliberately not here

No DSPy, no DBOS, no LangGraph, no Memori, no `google-adk`, no multi-tenant scaffolding,
no cost tracker, no MCP server. GADK needs some of that for its actual production
scope (targeting a real repo, running autonomously, multi-tenant future). This project
is scoped to answer one question — what does the *AI-facing core* look like if it's
designed for Ollama from day one instead of migrated onto it — not to replace GADK.
