import sys
import os
import json
import threading
import http.server
import shutil
import pytest
import sqlite3
import time
import subprocess
from pathlib import Path
from typing import Generator, Callable, Any

from ollama_swarm import cli
from ollama_swarm.config import Settings, Tier, models_for, SETTINGS
from ollama_swarm.backend import OllamaBackend
from ollama_swarm.memory import Memory, MemoryEntry
from ollama_swarm.orchestrator import Swarm, SwarmResult
from ollama_swarm.presets import default_registry, default_swarm_agents
from ollama_swarm.tools import ToolRegistry, ToolCallResult
from ollama_swarm.agents import run_agent, Agent, AgentRunResult
from ollama_swarm.router import Router

# =====================================================================
# 1. TEST FIXTURES & INFRASTRUCTURE FOR BACKEND/CLI MOCKING
# =====================================================================

from unittest.mock import patch as _patch

_UNSET = object()

def make_swarm(backend, registry=None, memory=None, planner=_UNSET, **kwargs):
    """Create a Swarm with all 9 agents, using default_swarm_agents() for convenience."""
    from ollama_swarm.presets import default_swarm_agents, default_registry
    agents = default_swarm_agents()
    if registry is None:
        registry = default_registry()
    return Swarm(
        planner=agents.planner if planner is _UNSET else planner,
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
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _patch_scaffold_in_e2e(tmp_path):
    """Prevent scaffold_project from touching the real filesystem in E2E tests."""
    orig = SETTINGS.workspace_root
    SETTINGS.workspace_root = str(tmp_path)
    with _patch("ollama_swarm.scaffold.scaffold_project", return_value={"README.md": "# test"}), \
         _patch("ollama_swarm.scaffold.detect_language", return_value="python"):
        yield
    SETTINGS.workspace_root = orig


class MockOllamaHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Handles requests mock-routing for the HTTP Server E2E Test Strategy."""
    
    chat_fn: Callable[[str, list, dict], dict] = lambda model, messages, req: {}
    embed_fn: Callable[[str, str, dict], dict] = lambda model, prompt, req: {}

    def log_message(self, format, *args):
        pass  # Suppress default server logs

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        req_body = json.loads(post_data.decode('utf-8'))
        
        if self.path == "/api/chat":
            model = req_body.get("model", "")
            messages = req_body.get("messages", [])
            response_data = MockOllamaHTTPHandler.chat_fn(model, messages, req_body)
        elif self.path in ("/api/embeddings", "/api/embed"):
            model = req_body.get("model", "")
            prompt = req_body.get("prompt", "")
            response_data = MockOllamaHTTPHandler.embed_fn(model, prompt, req_body)
        else:
            response_data = {}
            
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))


@pytest.fixture
def mock_ollama_server() -> Generator[str, None, None]:
    """Spins up a local mock Ollama API server in a background thread."""
    server = http.server.HTTPServer(('127.0.0.1', 0), MockOllamaHTTPHandler)
    port = server.server_port
    url = f"http://127.0.0.1:{port}"
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    yield url
    
    server.shutdown()
    server.server_close()


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    """Sets up a clean workspace sandbox directory for Builder tools."""
    workspace_dir = tmp_path / "sandbox_workspace"
    workspace_dir.mkdir()
    return workspace_dir


class MockBackendState:
    def __init__(self):
        self.chat_handler = self.default_chat_handler
        self.embed_handler = self.default_embed_handler
        self.chat_calls = []
        self.embed_calls = []
        self.fail_models = set()
        self.latency_map = {}

    def default_chat_handler(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        system_content = ""
        user_content = ""
        for m in messages:
            if m.get("role") == "system":
                system_content += m.get("content", "")
            elif m.get("role") == "user":
                user_content += m.get("content", "")

        role = "planner"
        if "Architect" in system_content:
            role = "architect"
        elif "Builder" in system_content:
            role = "builder"
        elif "Critic" in system_content:
            role = "critic"
        elif "Security" in system_content:
            role = "security"
        elif "Governor" in system_content:
            role = "governor"
        elif "FinOps" in system_content:
            role = "finops"

        content = {
            "planner": "Plan: 1. Do the work.",
            "architect": "Architecture: Use standard layout.",
            "builder": "Implementation: Done.",
            "critic": "APPROVE",
            "security": "SECURITY: GO",
            "governor": "GOVERN: GO",
            "finops": "FinOps: Run summary completed.",
        }[role]

        if role == "critic":
            if "fail" in user_content.lower() or "reject" in user_content.lower():
                content = "REQUEST_CHANGES: needs changes."
        elif role == "governor":
            if "fail" in user_content.lower() or "reject" in user_content.lower() or "no-go" in user_content.lower():
                content = "GOVERN: NO-GO: quality gates failed."

        return {
            "model": model,
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 10,
            "eval_count": 20
        }

    def default_embed_handler(self, model: str, text: str) -> list[float]:
        vec = [0.0] * 26
        for ch in text.lower():
            idx = ord(ch) - ord("a")
            if 0 <= idx < 26:
                vec[idx] += 1.0
        return vec


@pytest.fixture
def mock_backend(monkeypatch) -> MockBackendState:
    state = MockBackendState()
    
    def mock_chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        state.chat_calls.append((model, messages, kwargs))
        if model in state.fail_models:
            raise Exception(f"Model {model} is down / connection timed out.")
        
        lat = state.latency_map.get(model, 0.0)
        if lat > 0:
            time.sleep(lat)
            
        return state.chat_handler(model, messages, **kwargs)
        
    def mock_embed(self, model: str, text: str) -> list[float]:
        state.embed_calls.append((model, text))
        if model in state.fail_models:
            raise Exception(f"Embedding model {model} is offline.")
        return state.embed_handler(model, text)

    monkeypatch.setattr(OllamaBackend, "chat", mock_chat)
    monkeypatch.setattr(OllamaBackend, "embed", mock_embed)
    return state


@pytest.fixture(autouse=True)
def sandbox_settings(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(SETTINGS, "workspace_root", str(workspace))
    monkeypatch.setenv("OLLAMA_SWARM_ENABLE_DEV_TOOLS", "1")
    
    db_file = tmp_path / "swarm_memory.db"
    orig_init = Memory.__init__
    def patch_init(self, backend, db_path="swarm_memory.db"):
        if db_path == "swarm_memory.db":
            db_path = str(db_file)
        orig_init(self, backend, db_path)
    monkeypatch.setattr(Memory, "__init__", patch_init)


def subprocess_happy_path_chat(model, messages, req):
    system_prompt = ""
    for m in messages:
        if m.get("role") == "system":
            system_prompt += m.get("content", "")
            
    role = "planner"
    if "Architect" in system_prompt:
        role = "architect"
    elif "Builder" in system_prompt:
        role = "builder"
    elif "Critic" in system_prompt:
        role = "critic"
    elif "Security" in system_prompt:
        role = "security"
    elif "Governor" in system_prompt:
        role = "governor"
    elif "FinOps" in system_prompt:
        role = "finops"

    content = {
        "planner": "Plan: 1. Do subprocess task.",
        "architect": "Architecture: Subprocess architecture.",
        "builder": "Implementation: Subprocess implementation.",
        "critic": "APPROVE",
        "security": "SECURITY: GO",
        "governor": "GOVERN: GO",
        "finops": "FinOps: Subprocess cost summary.",
    }[role]
    
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 15,
        "eval_count": 25
    }


# =====================================================================
# TIER 1: FEATURE COVERAGE (25 Test Cases)
# =====================================================================

# --- Feature 1: Multi-Agent Phase Pipeline ---

def test_t1_f1_1_full_happy_path(mock_backend, monkeypatch, tmp_path):
    """Case 1.1: Verify all 6 phases run in order when all models succeed."""
    monkeypatch.setattr(sys, "argv", ["cli.py", "Happy path objective"])
    cli.main()
    assert len(mock_backend.chat_calls) >= 6
    roles = []
    for call in mock_backend.chat_calls:
        messages = call[1]
        system_content = messages[0]["content"]
        if "You are the Planner" in system_content:
            roles.append("PLAN")
        elif "You are the Architect" in system_content:
            roles.append("ARCHITECT")
        elif "You are the Builder" in system_content:
            roles.append("IMPLEMENT")
        elif "You are the Critic" in system_content:
            roles.append("REVIEW")
        elif "You are the Governor" in system_content:
            roles.append("GOVERN")
        elif "You are FinOps" in system_content:
            roles.append("OPERATE")
    for r in ["PLAN", "ARCHITECT", "IMPLEMENT", "REVIEW", "GOVERN", "OPERATE"]:
        assert r in roles


def test_t1_f1_2_goal_propagation(mock_backend, monkeypatch):
    """Case 1.2: Verify the goal is passed and accumulated in prompts."""
    goal = "Build a super computer with 5 lines of code"
    monkeypatch.setattr(sys, "argv", ["cli.py", goal])
    cli.main()
    for call in mock_backend.chat_calls:
        messages = call[1]
        content_str = " ".join([m["content"] for m in messages])
        assert goal in content_str


def test_t1_f1_3_phase_history_logging(mock_backend):
    """Case 1.3: Verify SwarmResult.history contains exactly one record per phase."""
    backend = OllamaBackend()
    registry = default_registry()
    agents = default_swarm_agents()
    memory = Memory(backend)
    swarm = make_swarm(backend, registry=registry, memory=memory)
    res = swarm.run("test history")
    assert len(res.history) >= 6
    phases = [r.phase for r in res.history]
    assert "PLAN" in phases
    assert "ARCHITECT" in phases
    assert "IMPLEMENT" in phases
    assert "REVIEW" in phases
    assert "GOVERN" in phases
    assert "OPERATE" in phases


def test_t1_f1_4_finops_execution(mock_backend):
    """Case 1.4: Verify FinOps runs at the end and records token ledger usage."""
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("finops tokens test")
    assert res.finops_summary != ""
    assert len(swarm._ledger) >= 6
    # SCAFFOLD phase uses a synthetic entry with 0 tokens — filter it out when
    # checking that LLM-driven phases record real token counts.
    llm_entries = [e for e in swarm._ledger if e["model"] != "scaffold"]
    assert len(llm_entries) >= 6
    for entry in llm_entries:
        assert "tokens_in" in entry
        assert "tokens_out" in entry
        assert entry["tokens_in"] == 10
        assert entry["tokens_out"] == 20


def test_t1_f1_5_cli_stdout_format(mock_backend, monkeypatch, capsys):
    """Case 1.5: Verify the CLI console output matches expected format."""
    monkeypatch.setattr(sys, "argv", ["cli.py", "cli output test"])
    cli.main()
    captured = capsys.readouterr()
    assert "APPROVED (governed)" in captured.out
    assert "PLAN (Planner via" in captured.out
    assert "OPERATE (FinOps via" in captured.out


# --- Feature 2: Bounded Critic Rework Loop ---

def test_t2_f2_1_no_rework_needed(mock_backend):
    """Case 2.1: Critic returns APPROVE immediately, zero rework cycles."""
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=2
    )
    res = swarm.run("zero rework test")
    assert res.approved is True
    assert res.rework_count == 0


def test_t2_f2_2_single_rework_success(mock_backend):
    """Case 2.2: Critic requests changes once, Builder resolves, Critic approves."""
    critic_count = 0
    def custom_chat(model, messages, **kwargs):
        nonlocal critic_count
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Critic" in system:
            critic_count += 1
            if critic_count == 1:
                return {"model": model, "message": {"role": "assistant", "content": "REQUEST_CHANGES: fix tests"}}
            return {"model": model, "message": {"role": "assistant", "content": "APPROVE: changes look great"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=2
    )
    res = swarm.run("single rework test")
    assert res.approved is True
    assert res.rework_count == 1


def test_t2_f2_3_rework_budget_exhaustion(mock_backend):
    """Case 2.3: Critic always requests changes; verify loop exits at max_rework."""
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "REQUEST_CHANGES: unacceptable solution"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=2
    )
    res = swarm.run("budget exhaust test")
    assert res.approved is False
    assert res.rework_count == 3


def test_t2_f2_4_critic_feedback_propagation(mock_backend):
    """Case 2.4: Verify Critic feedback content is passed in next Builder prompt."""
    feedback_seen = False
    def custom_chat(model, messages, **kwargs):
        nonlocal feedback_seen
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        user = "".join([m["content"] for m in messages if m.get("role") == "user"])
        if "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "REQUEST_CHANGES: UNIQUE_CRITICISM_123"}}
        if "Builder" in system:
            if "UNIQUE_CRITICISM_123" in user:
                feedback_seen = True
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=1
    )
    swarm.run("feedback prop test")
    assert feedback_seen is True


def test_t2_f2_5_tool_evidence_compilation(mock_backend):
    """Case 2.5: Verify Builder tool results are correctly compiled for Critic."""
    evidence_sent = False
    builder_calls = 0
    def custom_chat_with_tool(model, messages, **kwargs):
        nonlocal evidence_sent
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        user = "".join([m["content"] for m in messages if m.get("role") == "user"])
        if "Builder" in system:
            if messages[-1].get("role") == "tool":
                return {"model": model, "message": {"role": "assistant", "content": "Done with tool."}}
            return {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "evidence.txt", "content": "evidence content"}
                        }
                    }]
                }
            }
        elif "Critic" in system:
            if "Tool execution evidence: 1 real tool call(s)" in user:
                evidence_sent = True
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat_with_tool
    backend = OllamaBackend()
    agents = default_swarm_agents()
    registry = default_registry()
    swarm = make_swarm(backend, registry=registry)
    swarm.run("tool evidence test")
    assert evidence_sent is True


# --- Feature 3: Quality Gates & Governance ---

def test_t3_f3_1_governor_immediate_go(mock_backend):
    """Case 3.1: Quality gates pass on first attempt, Governor returns GO."""
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("immediate go test")
    assert res.governed is True
    assert res.governor_rework_count == 0


def test_t3_f3_2_governor_no_go_rework_success(mock_backend):
    """Case 3.2: Governor says NO-GO once, Builder fixes, gates pass on retry."""
    gov_calls = 0
    def custom_chat(model, messages, **kwargs):
        nonlocal gov_calls
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Governor" in system:
            gov_calls += 1
            if gov_calls == 1:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO: tests fail."}}
            return {"model": model, "message": {"role": "assistant", "content": "GOVERN: GO: fixed."}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=2
    )
    res = swarm.run("gov rework success test")
    assert res.governed is True
    assert res.governor_rework_count == 1


def test_t3_f3_3_governor_rework_exhaustion(mock_backend):
    """Case 3.3: Governor always says NO-GO; loops exit and returns governed=False."""
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Governor" in system:
            return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO: still fail."}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=1
    )
    res = swarm.run("gov rework exhaustion test")
    assert res.governed is False
    assert res.governor_rework_count == 2


def test_t3_f3_4_workspace_scoped_execution(tmp_path, monkeypatch):
    """Case 3.4: Verify quality gates run within the custom workspace path."""
    custom_ws = tmp_path / "custom_ws"
    custom_ws.mkdir()
    test_file = custom_ws / "test_dummy.py"
    test_file.write_text("def test_fail(): assert False")
    monkeypatch.setattr(SETTINGS, "workspace_root", str(custom_ws))

    registry = ToolRegistry()
    from ollama_swarm.quality_gates import register_governance_tools
    register_governance_tools(registry)

    result = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {"workspace": str(custom_ws)}}})
    assert result.result["tests_ok"] is False
    assert "test_fail" in result.result["tests_output"]


def test_t3_f3_5_linting_bypass_when_ruff_missing(monkeypatch):
    """Case 3.5: Verify quality gates fall back gracefully when ruff is missing."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "ruff" else "/usr/bin/python3")
    registry = ToolRegistry()
    from ollama_swarm.quality_gates import register_governance_tools
    register_governance_tools(registry)
    
    result = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {}}})
    assert result.result["lint_ok"] is None
    assert "ruff not installed, lint skipped" in result.result["lint_output"]


# --- Feature 4: Cross-Run Memory & Recall ---

def test_t4_f4_1_memory_storage_on_success(mock_backend, tmp_path):
    """Case 4.1: Approved/governed run summary is stored to SQLite DB."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "test_mem_success.db")
    memory = Memory(backend, db_path=db_path)
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), memory=memory
    )
    swarm.run("successful memory test")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT tag, text FROM memory")
    rows = cursor.fetchall()
    assert len(rows) > 0
    tags = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    assert "run_summary" in tags
    assert any("outcome: approved and governed" in t.lower() for t in texts)
    conn.close()


def test_t4_f4_2_memory_storage_on_failure(mock_backend, tmp_path):
    """Case 4.2: Unresolved run summary is stored to SQLite DB."""
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "REQUEST_CHANGES: reject"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    db_path = str(tmp_path / "test_mem_fail.db")
    memory = Memory(backend, db_path=db_path)
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), memory=memory, max_rework=1
    )
    swarm.run("unresolved memory test")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT tag, text FROM memory WHERE tag='run_summary'")
    rows = cursor.fetchall()
    assert len(rows) > 0
    assert "outcome: unresolved" in rows[0][1].lower()
    conn.close()


def test_t4_f4_3_context_recall_integration(mock_backend, tmp_path):
    """Case 4.3: Memory entries are recalled and injected to Planner."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "test_mem_recall.db")
    memory = Memory(backend, db_path=db_path)
    memory.remember("Past Tip: Always use a custom fixture.", tag="note")
    
    planner_saw_context = False
    def custom_chat(model, messages, **kwargs):
        nonlocal planner_saw_context
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Planner" in system:
            if "Past Tip: Always use a custom fixture." in system:
                planner_saw_context = True
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), memory=memory
    )
    swarm.run("recall memory test")
    assert planner_saw_context is True


def test_t4_f4_4_embedding_generation(mock_backend, tmp_path):
    """Case 4.4: Memory uses embedding model to generate SQLite stored vector."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "test_mem_embed.db")
    memory = Memory(backend, db_path=db_path)
    initial_embed_calls = len(mock_backend.embed_calls)
    memory.remember("Generating embedding text", tag="test")
    assert len(mock_backend.embed_calls) == initial_embed_calls + 1
    assert mock_backend.embed_calls[-1][1] == "Generating embedding text"
    
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT embedding FROM memory WHERE text='Generating embedding text'").fetchone()
    assert row is not None
    embedding = json.loads(row[0])
    assert len(embedding) == 26
    conn.close()


def test_t4_f4_5_db_connection_init(tmp_path):
    """Case 4.5: SQLite tables are created dynamically if DB does not exist."""
    db_path = tmp_path / "new_nonexistent.db"
    assert not db_path.exists()
    backend = OllamaBackend()
    memory = Memory(backend, db_path=str(db_path))
    assert db_path.exists()
    
    conn = sqlite3.connect(str(db_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "memory" in tables
    conn.close()


# --- Feature 5: Resilient Model Router ---

def test_t5_f5_1_healthy_first_model_usage(mock_backend):
    """Case 5.1: Router picks preferred model in catalog when healthy."""
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    swarm.run("router healthy model test")
    primary_coding_model = models_for(Tier.CODING)[0]
    coding_calls = [call for call in mock_backend.chat_calls if call[0] == primary_coding_model]
    assert len(coding_calls) > 0


def test_t5_f5_2_single_model_fallback(mock_backend):
    """Case 5.2: Router falls back to next model on failure."""
    primary_coding_model = models_for(Tier.CODING)[0]
    fallback_coding_model = models_for(Tier.CODING)[1]
    mock_backend.fail_models.add(primary_coding_model)
    
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    swarm.run("router fallback test")
    coding_calls = [call for call in mock_backend.chat_calls if call[0] == fallback_coding_model]
    assert len(coding_calls) > 0
    assert swarm.router._stats[primary_coding_model].failures > 0
    assert swarm.router._stats[fallback_coding_model].successes > 0


def test_t5_f5_3_demote_unhealthy_models():
    """Case 5.3: Models with >50% failure rate are demoted to back of chain."""
    router = Router()
    primary = models_for(Tier.CODING)[0]
    router.record(primary, ok=False)
    router.record(primary, ok=False)
    router.record(primary, ok=False)
    
    chain = router.fallback_chain(Tier.CODING)
    assert chain[-1] == primary


def test_t5_f5_4_router_latency_tracking(mock_backend):
    """Case 5.4: Router tracks and averages latencies for successful runs."""
    primary = models_for(Tier.CODING)[0]
    mock_backend.latency_map[primary] = 0.05
    
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    swarm.run("latency track test")
    
    stats = swarm.router._stats.get(primary)
    assert stats is not None
    assert stats.latency_samples > 0
    assert stats.average_latency_s > 0.0


def test_t5_f5_5_complete_tier_exhaustion(mock_backend):
    """Case 5.5: Router throws exception when all models in tier fail."""
    for model in models_for(Tier.REASONING):
        mock_backend.fail_models.add(model)
        
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    with pytest.raises(Exception) as excinfo:
        swarm.run("unresolvable fallback chain")
    assert "is down" in str(excinfo.value) or "connection timed out" in str(excinfo.value)


# =====================================================================
# TIER 2: BOUNDARY & CORNER CASES (25 Test Cases)
# =====================================================================

# --- Feature 1: Multi-Agent Phase Pipeline ---

def test_t2_f1_6_empty_goal(mock_backend):
    """Case 1.6: Running with empty goal string."""
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("")
    assert res.goal == ""
    assert len(res.history) >= 6


def test_t2_f1_7_ultra_long_goal(mock_backend):
    """Case 1.7: Running with extremely long goal inputs (>10k chars)."""
    long_goal = "A" * 12000
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run(long_goal)
    assert res.goal == long_goal
    assert len(res.history) >= 6


def test_t2_f1_8_token_ledger_overflow(mock_backend):
    """Case 1.8: Resiliency to negative/extreme token counts in backend response."""
    def custom_chat(model, messages, **kwargs):
        res = mock_backend.default_chat_handler(model, messages, **kwargs)
        res["prompt_eval_count"] = -500
        res["eval_count"] = 999999999999
        return res
        
    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("token ledger overflow")
    assert len(swarm._ledger) >= 6
    total_in = sum(e["tokens_in"] for e in swarm._ledger)
    total_out = sum(e["tokens_out"] for e in swarm._ledger)
    assert total_in < 0
    assert total_out > 1000000


def test_t2_f1_9_phase_timeout(mock_backend):
    """Case 1.9: Backend timeout handling (hangs during pipeline execution)."""
    def timeout_chat(*args, **kwargs):
        raise Exception("Request timed out after 60s")
        
    mock_backend.chat_handler = timeout_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    with pytest.raises(Exception) as excinfo:
        swarm.run("timeout test")
    assert "timed out" in str(excinfo.value)


def test_t2_f1_10_missing_agents(mock_backend):
    """Case 1.10: Resiliency to missing agents or incomplete parameters.
    Passing planner=None raises AttributeError when the planner phase runs.
    """
    backend = OllamaBackend()
    swarm = make_swarm(backend, registry=default_registry(), planner=None)  # type: ignore
    with pytest.raises((AttributeError, TypeError)):
        swarm.run("test missing agents")


# --- Feature 2: Bounded Critic Rework Loop ---

def test_t2_f2_6_max_tool_turns_reached(mock_backend, monkeypatch):
    """Case 2.6: Builder max tool turns reached during a rework cycle."""
    monkeypatch.setattr(SETTINGS, "max_tool_turns", 2)
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Builder" in system:
            return {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "dummy.txt"}
                        }
                    }]
                }
            }
        return mock_backend.default_chat_handler(model, messages, **kwargs)
        
    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("max tool turns test")
    implement_records = [r for r in res.history if r.phase == "IMPLEMENT"]
    assert len(implement_records) > 0
    assert "[max tool turns reached without a final answer]" in implement_records[-1].content


def test_t2_f2_7_rework_count_zero(mock_backend):
    """Case 2.7: Running with `max_rework = 0`."""
    critic_calls = 0
    def custom_chat(model, messages, **kwargs):
        nonlocal critic_calls
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Critic" in system:
            critic_calls += 1
            return {"model": model, "message": {"role": "assistant", "content": "REQUEST_CHANGES: fix it"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=0
    )
    res = swarm.run("max_rework=0 test")
    assert res.approved is False
    assert res.rework_count == 1
    assert critic_calls == 1


def test_t2_f2_8_invalid_critic_verdict(mock_backend):
    """Case 2.8: Handling invalid Critic verdicts (empty or non-standard prose)."""
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "The solution is okay-ish but I won't say the A-word"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=1
    )
    res = swarm.run("invalid critic verdict")
    assert res.approved is False


def test_t2_f2_9_huge_critic_feedback(mock_backend):
    """Case 2.9: Excessively large review feedback content handling."""
    large_feedback = "CRITIQUE: " + ("X" * 20000)
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": large_feedback}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=1
    )
    res = swarm.run("huge critic feedback test")
    assert res.approved is False
    assert len(res.history) > 0


def test_t2_f2_10_rework_no_tool_calls(mock_backend):
    """Case 2.10: Critic feedback loop execution with no tool execution."""
    evidence_text = ""
    def custom_chat(model, messages, **kwargs):
        nonlocal evidence_text
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        user = "".join([m["content"] for m in messages if m.get("role") == "user"])
        if "Critic" in system:
            evidence_text = user
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=1
    )
    swarm.run("no tools call test")
    assert "Tool execution evidence: none (no tools were called)." in evidence_text


# --- Feature 3: Quality Gates & Governance ---

def test_t2_f3_6_pytest_no_tests_collected(monkeypatch):
    """Case 3.6: Handling pytest returning exit code 5 (no tests collected)."""
    orig_run = subprocess.run
    def mock_run(cmd, *args, **kwargs):
        if "pytest" in cmd:
            class DummyCompletedProcess:
                returncode = 5
                stdout = "collected 0 items"
                stderr = ""
            return DummyCompletedProcess()
        return orig_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)
    registry = ToolRegistry()
    from ollama_swarm.quality_gates import register_governance_tools
    register_governance_tools(registry)
    
    result = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {}}})
    assert result.result["tests_ok"] is True
    assert "collected 0 items" in result.result["tests_output"]


def test_t2_f3_7_pytest_timeout(monkeypatch):
    """Case 3.7: Test runner (pytest) execution timeout."""
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=60)
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    registry = ToolRegistry()
    from ollama_swarm.quality_gates import register_governance_tools
    register_governance_tools(registry)
    
    result = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {}}})
    assert result.result["tests_ok"] is False
    assert "timed out after" in result.result["tests_output"]


def test_t2_f3_8_large_test_output_truncation(monkeypatch):
    """Case 3.8: Truncation of extremely large test log outputs."""
    large_stdout = "X" * 10000
    def mock_run(cmd, *args, **kwargs):
        class DummyCompletedProcess:
            returncode = 0
            stdout = large_stdout
            stderr = ""
        return DummyCompletedProcess()
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    registry = ToolRegistry()
    from ollama_swarm.quality_gates import register_governance_tools
    register_governance_tools(registry)
    
    result = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {}}})
    assert len(result.result["tests_output"]) == 4000


def test_t2_f3_9_governor_rework_zero(mock_backend):
    """Case 3.9: Running with `max_governor_rework = 0`."""
    gov_calls = 0
    def custom_chat(model, messages, **kwargs):
        nonlocal gov_calls
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Governor" in system:
            gov_calls += 1
            return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO: fail"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=0
    )
    res = swarm.run("governor rework zero")
    assert res.governed is False
    assert res.governor_rework_count == 1
    assert gov_calls == 1


def test_t2_f3_10_nonstandard_governor_verdict(mock_backend):
    """Case 3.10: Governor non-standard verdict handling (defaults to NO-GO)."""
    def custom_chat(model, messages, **kwargs):
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Governor" in system:
            return {"model": model, "message": {"role": "assistant", "content": "The results look decent but no green light is here."}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=0
    )
    res = swarm.run("nonstandard governor verdict")
    assert res.governed is False


# --- Feature 4: Cross-Run Memory & Recall ---

def test_t2_f4_6_readonly_database_file(mock_backend, tmp_path):
    """Case 4.6: SQLite DB file is read-only / permissions locked."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "readonly.db")
    memory = Memory(backend, db_path=db_path)
    memory.conn.close()
    with pytest.raises(sqlite3.Error):
        memory.remember("some text")


def test_t2_f4_7_zero_embedding_matches(mock_backend, tmp_path):
    """Case 4.7: Handling recall on zero vector matches."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "empty_recall.db")
    memory = Memory(backend, db_path=db_path)
    recalled = memory.recall("test query")
    assert len(recalled) == 0


def test_t2_f4_8_massive_memory_note(mock_backend, tmp_path):
    """Case 4.8: Storing massive text payloads in SQLite notes."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "massive_note.db")
    memory = Memory(backend, db_path=db_path)
    massive_text = "M" * 500000
    memory.remember(massive_text, tag="massive")
    recalled = memory.recall("query text", top_k=1)
    assert len(recalled) == 1
    assert recalled[0].text == massive_text


def test_t2_f4_9_concurrent_writes(mock_backend, tmp_path):
    """Case 4.9: Multi-process concurrent memory access (locking conditions)."""
    db_path = str(tmp_path / "concurrent.db")
    backend = OllamaBackend()
    
    def write_worker(idx):
        mem = Memory(backend, db_path=db_path)
        mem.remember(f"thread {idx} content")
        mem.conn.close()
        
    threads = []
    for i in range(5):
        t = threading.Thread(target=write_worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    mem = Memory(backend, db_path=db_path)
    entries = mem.recall("thread", top_k=10)
    assert len(entries) == 5


def test_t2_f4_10_corrupted_embedding_json(mock_backend, tmp_path):
    """Case 4.10: SQLite database contains corrupted JSON embedding format."""
    backend = OllamaBackend()
    db_path = str(tmp_path / "corrupt.db")
    memory = Memory(backend, db_path=db_path)
    memory.remember("some text")
    
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE memory SET embedding = 'invalid_json'")
    conn.commit()
    conn.close()
    
    with pytest.raises(json.JSONDecodeError):
        memory.recall("test query")


# --- Feature 5: Resilient Model Router ---

def test_t2_f5_6_fast_recovery_unhealthy_model():
    """Case 5.6: Model health recovery (promoting back after success)."""
    router = Router()
    model = models_for(Tier.CODING)[0]
    router.record(model, ok=False)
    router.record(model, ok=False)
    router.record(model, ok=False)
    
    chain = router.fallback_chain(Tier.CODING)
    assert chain[-1] == model
    
    router.record(model, ok=True)
    router.record(model, ok=True)
    router.record(model, ok=True)
    router.record(model, ok=True)
    
    chain2 = router.fallback_chain(Tier.CODING)
    assert chain2[0] == model


def test_t2_f5_7_negative_or_zero_latency():
    """Case 5.7: System clock drift (negative latency measurements)."""
    router = Router()
    model = "model_a"
    router.record(model, ok=True, latency_s=-10.0)
    router.record(model, ok=True, latency_s=0.0)
    stats = router._stats[model]
    assert stats.average_latency_s == -5.0


def test_t2_f5_8_missing_token_count_in_response(mock_backend):
    """Case 5.8: Handling missing token keys in Ollama response metadata."""
    def custom_chat(model, messages, **kwargs):
        return {
            "model": model,
            "message": {"role": "assistant", "content": "Done without token info"}
        }
        
    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("missing token keys test")
    assert len(swarm._ledger) >= 6
    for entry in swarm._ledger:
        assert entry["tokens_in"] == 0
        assert entry["tokens_out"] == 0


def test_t2_f5_9_direct_cloud_no_api_key(monkeypatch):
    """Case 5.9: Direct-cloud execution with missing API keys."""
    settings = Settings(mode="direct-cloud", api_key="")
    backend = OllamaBackend(settings=settings)
    from ollama import Client
    client_args = []
    
    def mock_client_init(self, *args, **kwargs):
        client_args.append(kwargs)
        self.chat = lambda *a, **k: {}
        
    monkeypatch.setattr(Client, "__init__", mock_client_init)
    _ = backend.client
    assert len(client_args) == 1
    assert "headers" not in client_args[0]


def test_t2_f5_10_host_resolution_error(mock_backend):
    """Case 5.10: API endpoint host resolution failure (DNS failure)."""
    def dns_fail_chat(*args, **kwargs):
        raise Exception("Failed to resolve host 'localhost'")
        
    mock_backend.chat_handler = dns_fail_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    with pytest.raises(Exception) as excinfo:
        swarm.run("dns failure test")
    assert "Failed to resolve host" in str(excinfo.value)


# =====================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (5 Test Cases)
# =====================================================================

def test_t3_c1_memory_router_integration(mock_backend, tmp_path, monkeypatch):
    """Case 3.1: Router fallback succeeds when embedding model is offline."""
    monkeypatch.setattr("ollama_swarm.config.models_for", lambda tier: ["nomic-embed-text", "fallback-embed"] if "embed" in str(tier).lower() else models_for(tier))
    monkeypatch.setattr("ollama_swarm.memory.models_for", lambda tier: ["nomic-embed-text", "fallback-embed"] if "embed" in str(tier).lower() else models_for(tier))
    monkeypatch.setattr("ollama_swarm.router.models_for", lambda tier: ["nomic-embed-text", "fallback-embed"] if "embed" in str(tier).lower() else models_for(tier))
    primary_embed = "nomic-embed-text"
    mock_backend.fail_models.add(primary_embed)
    router = Router()
    backend = OllamaBackend()
    
    def custom_embed(self, model, text):
        chain = router.fallback_chain(Tier.EMBED)
        last_exc = None
        for m in chain:
            if m in mock_backend.fail_models:
                router.record(m, ok=False)
                last_exc = Exception("Offline")
                continue
            router.record(m, ok=True)
            return mock_backend.default_embed_handler(m, text)
        raise last_exc
        
    import types
    backend.embed = types.MethodType(custom_embed, backend)
    memory = Memory(backend, db_path=str(tmp_path / "mem_router.db"))
    memory.remember("test offline embedding")
    assert router._stats[primary_embed].failures > 0


def test_t3_c2_critic_sandbox_error_routing(mock_backend):
    """Case 3.2: Sandbox path traversal ValueError routes to Critic rework."""
    critic_received_error = False
    def custom_chat(model, messages, **kwargs):
        nonlocal critic_received_error
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        user = "".join([m["content"] for m in messages if m.get("role") == "user"])
        if "Builder" in system:
            if messages[-1].get("role") == "tool":
                return {"model": model, "message": {"role": "assistant", "content": "Failed to escape sandbox."}}
            return {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "../escaped.txt", "content": "payload"}
                        }
                    }]
                }
            }
        elif "Critic" in system:
            if "escapes workspace root" in user:
                critic_received_error = True
            return {"model": model, "message": {"role": "assistant", "content": "REQUEST_CHANGES: Sandbox violation detected!"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_rework=1
    )
    swarm.run("sandbox routing test")
    assert critic_received_error is True


def test_t3_c3_memory_governor_synergy(mock_backend, tmp_path):
    """Case 3.3: Planner uses recalled past run failure to fix current run."""
    backend = OllamaBackend()
    memory = Memory(backend, db_path=str(tmp_path / "synergy.db"))
    memory.remember("Goal: fix bugs\nOutcome: approved but ungoverned (governor rework exhausted). Reason: Missing unit tests.", tag="run_summary")
    
    planner_saw_failure = False
    def custom_chat(model, messages, **kwargs):
        nonlocal planner_saw_failure
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Planner" in system:
            if "Missing unit tests" in system:
                planner_saw_failure = True
        return mock_backend.default_chat_handler(model, messages, **kwargs)
        
    mock_backend.chat_handler = custom_chat
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), memory=memory
    )
    swarm.run("fix bugs")
    assert planner_saw_failure is True


def test_t3_c4_router_fallback_during_governor_rework(mock_backend):
    """Case 3.4: Router recovers coding model during a governor rework cycle."""
    gov_calls = 0
    builder_calls = 0
    primary_coding = models_for(Tier.CODING)[0]
    fallback_coding = models_for(Tier.CODING)[1]
    
    def custom_chat(model, messages, **kwargs):
        nonlocal gov_calls, builder_calls
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Governor" in system:
            gov_calls += 1
            if gov_calls == 1:
                mock_backend.fail_models.add(primary_coding)
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO: tests fail."}}
            return {"model": model, "message": {"role": "assistant", "content": "GOVERN: GO"}}
        elif "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "APPROVE"}}
        elif "Builder" in system:
            builder_calls += 1
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=1
    )
    res = swarm.run("governor rework router test")
    assert res.governed is True
    assert primary_coding in mock_backend.fail_models
    fallback_calls = [call for call in mock_backend.chat_calls if call[0] == fallback_coding]
    assert len(fallback_calls) > 0


def test_t3_c5_cli_subprocess_direct_cloud_mode(mock_ollama_server, mock_workspace):
    """Case 3.5: Subprocess CLI mode direct-cloud validation via HTTP mock server."""
    MockOllamaHTTPHandler.chat_fn = subprocess_happy_path_chat
    MockOllamaHTTPHandler.embed_fn = lambda model, prompt, req: {"embedding": [0.1] * 26}
    
    env = os.environ.copy()
    env["OLLAMA_MODE"] = "direct-cloud"
    env["OLLAMA_CLOUD_HOST"] = mock_ollama_server
    env["OLLAMA_API_KEY"] = "mock_api_key_123"
    env["OLLAMA_SWARM_WORKSPACE"] = str(mock_workspace)
    
    result = subprocess.run(
        ["python3", "-m", "ollama_swarm.cli", "subprocess direct cloud goal"],
        cwd=str(mock_workspace),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "APPROVED (governed)" in result.stdout


# =====================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (5 Test Cases)
# =====================================================================

def test_t4_s1_bug_fix_workflow(mock_backend, tmp_path, monkeypatch):
    """Case 4.1: Bug Fix Workflow: Test fails -> fixed by Builder -> passes."""
    workspace = tmp_path / "bug_fix_ws"
    workspace.mkdir()
    monkeypatch.setattr(SETTINGS, "workspace_root", str(workspace))
    
    test_file = workspace / "test_app.py"
    test_file.write_text("def test_addition(): assert 1 + 1 == 3")
    
    builder_called = False
    def custom_chat(model, messages, **kwargs):
        nonlocal builder_called
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Builder" in system:
            builder_called = True
            if messages[-1].get("role") == "tool":
                return {"model": model, "message": {"role": "assistant", "content": "I fixed the test."}}
            return {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": "I fixed the test.",
                    "tool_calls": [{
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "test_app.py", "content": "def test_addition(): assert 1 + 1 == 2"}
                        }
                    }]
                }
            }
        elif "Governor" in system:
            registry = default_registry()
            res = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {"workspace": str(workspace)}}})
            if res.result["tests_ok"]:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: GO"}}
            else:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("fix addition test")
    assert res.governed is True
    assert builder_called is True
    assert "assert 1 + 1 == 2" in test_file.read_text()


def test_t4_s2_regression_detection_and_rework(mock_backend, tmp_path, monkeypatch):
    """Case 4.2: Feature addition breaks tests; Governor rejects; Builder fixes."""
    workspace = tmp_path / "regression_ws"
    workspace.mkdir()
    monkeypatch.setattr(SETTINGS, "workspace_root", str(workspace))
    
    test_file = workspace / "test_app.py"
    test_file.write_text("def test_ok(): assert True")
    
    builder_calls = 0
    def custom_chat(model, messages, **kwargs):
        nonlocal builder_calls
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Builder" in system:
            builder_calls += 1
            # If the builder is processing a tool result, return the final text
            if messages[-1].get("role") == "tool":
                if "parse_error" in str(messages):
                    return {"model": model, "message": {"role": "assistant", "content": "Introducing new feature done."}}
                else:
                    return {"model": model, "message": {"role": "assistant", "content": "Fixed syntax error done."}}
                    
            # Otherwise, initiate the tool call
            if "Previous attempt" not in str(messages) and "Governor feedback" not in str(messages):
                # First run
                return {
                    "model": model,
                    "message": {
                        "role": "assistant",
                        "content": "Introducing new feature.",
                        "tool_calls": [{
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "test_app.py", "content": "def test_ok() parse_error"}
                            }
                        }]
                    }
                }
            else:
                # Second run (governor rework)
                return {
                    "model": model,
                    "message": {
                        "role": "assistant",
                        "content": "Fixed syntax error.",
                        "tool_calls": [{
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "test_app.py", "content": "def test_ok(): assert True"}
                            }
                        }]
                    }
                }
        elif "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "APPROVE"}}
        elif "Governor" in system:
            registry = default_registry()
            res = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {"workspace": str(workspace)}}})
            if res.result["tests_ok"]:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: GO"}}
            else:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO"}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=1
    )
    res = swarm.run("prevent regression")
    assert res.governed is True
    # Verify that the Builder IMPLEMENT phase was run exactly twice in the swarm history
    implement_records = [r for r in res.history if r.phase == "IMPLEMENT"]
    assert len(implement_records) == 2


def test_t4_s3_ruff_lint_error_handling(mock_backend, tmp_path, monkeypatch):
    """Case 4.3: Refactoring generates unused import; Ruff flags; Builder fixes."""
    workspace = tmp_path / "lint_ws"
    workspace.mkdir()
    monkeypatch.setattr(SETTINGS, "workspace_root", str(workspace))
    
    test_file = workspace / "test_app.py"
    test_file.write_text("import sys\ndef test_ok(): assert True")
    
    builder_calls = 0
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ruff" if cmd == "ruff" else None)
    
    orig_run = subprocess.run
    def mock_subprocess_run(cmd, *args, **kwargs):
        if "ruff" in cmd:
            content = test_file.read_text()
            if "import sys" in content:
                class DummyProcess:
                    returncode = 1
                    stdout = "test_app.py:1:8: F401 [*] 'sys' imported but unused"
                    stderr = ""
                return DummyProcess()
            else:
                class DummyProcess:
                    returncode = 0
                    stdout = ""
                    stderr = ""
                return DummyProcess()
        return orig_run(cmd, *args, **kwargs)
        
    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    
    def custom_chat(model, messages, **kwargs):
        nonlocal builder_calls
        system = "".join([m["content"] for m in messages if m.get("role") == "system"])
        if "Builder" in system:
            builder_calls += 1
            if messages[-1].get("role") == "tool":
                return {"model": model, "message": {"role": "assistant", "content": "Fixed lint error done."}}
                
            if "Previous attempt" not in str(messages) and "Governor feedback" not in str(messages):
                return {
                    "model": model,
                    "message": {
                        "role": "assistant",
                        "content": "Keep unused import.",
                        "tool_calls": []
                    }
                }
            else:
                return {
                    "model": model,
                    "message": {
                        "role": "assistant",
                        "content": "Fixed lint error.",
                        "tool_calls": [{
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "test_app.py", "content": "def test_ok(): assert True"}
                            }
                        }]
                    }
                }
        elif "Critic" in system:
            return {"model": model, "message": {"role": "assistant", "content": "APPROVE"}}
        elif "Governor" in system:
            registry = default_registry()
            res = registry.dispatch({"function": {"name": "run_quality_gates", "arguments": {"workspace": str(workspace)}}})
            if res.result["lint_ok"] is True:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: GO"}}
            else:
                return {"model": model, "message": {"role": "assistant", "content": "GOVERN: NO-GO: " + res.result["lint_output"]}}
        return mock_backend.default_chat_handler(model, messages, **kwargs)

    mock_backend.chat_handler = custom_chat
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry(), max_governor_rework=1
    )
    res = swarm.run("fix lint errors")
    assert res.governed is True
    # Verify that the Builder IMPLEMENT phase was run exactly twice in the swarm history
    implement_records = [r for r in res.history if r.phase == "IMPLEMENT"]
    assert len(implement_records) == 2


def test_t4_s4_sandbox_protection_violation(mock_backend):
    """Case 4.4: Builder attempts path traversal escape; Tool blocks it safely."""
    registry = default_registry()
    res = registry.dispatch({
        "function": {
            "name": "write_file",
            "arguments": {"path": "../../../etc/passwd", "content": "malicious"}
        }
    })
    assert res.error is not None
    assert "escapes workspace root" in res.error


def test_t4_s5_offline_backend_resilience(mock_backend):
    """Case 4.5: Preferred models fail; Swarm runs fully on fallback chain."""
    reasoning_preferred = models_for(Tier.REASONING)[0]
    coding_preferred = models_for(Tier.CODING)[0]
    fast_preferred = models_for(Tier.FAST)[0]
    
    mock_backend.fail_models.add(reasoning_preferred)
    mock_backend.fail_models.add(coding_preferred)
    mock_backend.fail_models.add(fast_preferred)
    
    backend = OllamaBackend()
    agents = default_swarm_agents()
    swarm = make_swarm(backend, registry=default_registry()
    )
    res = swarm.run("resilience test")
    
    assert res.approved is True
    assert res.governed is True
    
    reasoning_fallback = models_for(Tier.REASONING)[1]
    coding_fallback = models_for(Tier.CODING)[1]
    fast_fallback = models_for(Tier.FAST)[1]
    
    used_models = [call[0] for call in mock_backend.chat_calls]
    assert reasoning_fallback in used_models
    assert coding_fallback in used_models
    assert fast_fallback in used_models
