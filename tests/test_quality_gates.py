from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ollama_swarm.config import SETTINGS
from ollama_swarm.quality_gates import register_governance_tools
from ollama_swarm.tools import ToolRegistry


def _run_quality_gates(workspace: Path) -> dict:
    registry = ToolRegistry()
    register_governance_tools(registry)
    call = _make_call(str(workspace))
    orig_root = SETTINGS.workspace_root
    SETTINGS.workspace_root = str(workspace)
    try:
        result = registry.dispatch(call)
    finally:
        SETTINGS.workspace_root = orig_root
    assert result.error is None, result.error
    return result.result


def _make_call(workspace: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        function=SimpleNamespace(name="run_quality_gates", arguments={"workspace": workspace})
    )


def test_passing_test_suite_reports_tests_ok(tmp_path: Path) -> None:
    (tmp_path / "test_passing.py").write_text("def test_ok():\n    assert True\n")

    result = _run_quality_gates(tmp_path)

    assert result["tests_ok"] is True
    assert isinstance(result["tests_output"], str)


def test_failing_test_suite_reports_not_ok_with_details(tmp_path: Path) -> None:
    (tmp_path / "test_failing.py").write_text("def test_fails():\n    assert False\n")

    result = _run_quality_gates(tmp_path)

    assert result["tests_ok"] is False
    assert "assert" in result["tests_output"] or "FAILED" in result["tests_output"]


def test_no_tests_collected_is_not_treated_as_failure(tmp_path: Path) -> None:
    # Empty workspace: pytest exits 5 ("no tests collected"), which shouldn't
    # block a gate meant to catch regressions, not mandate test coverage.
    result = _run_quality_gates(tmp_path)

    assert result["tests_ok"] is True


def test_return_shape_matches_contract(tmp_path: Path) -> None:
    (tmp_path / "test_passing.py").write_text("def test_ok():\n    assert True\n")

    result = _run_quality_gates(tmp_path)

    assert set(result.keys()) == {"tests_ok", "tests_output", "lint_ok", "lint_output"}
    assert isinstance(result["tests_ok"], bool)
    assert isinstance(result["tests_output"], str)
    assert result["lint_ok"] is None or isinstance(result["lint_ok"], bool)
    assert isinstance(result["lint_output"], str)


@pytest.mark.skipif(shutil.which("ruff") is not None, reason="ruff is installed; covered by the other lint test")
def test_lint_skipped_when_ruff_not_installed(tmp_path: Path) -> None:
    (tmp_path / "test_passing.py").write_text("def test_ok():\n    assert True\n")

    result = _run_quality_gates(tmp_path)

    assert result["lint_ok"] is None
    assert "ruff" in result["lint_output"].lower()
    assert "skip" in result["lint_output"].lower()


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed in this sandbox")
def test_lint_runs_when_ruff_installed(tmp_path: Path) -> None:
    (tmp_path / "test_passing.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "bad_style.py").write_text("import os\n")  # unused import -> ruff failure

    result = _run_quality_gates(tmp_path)

    assert result["lint_ok"] is False

    (tmp_path / "bad_style.py").unlink()
    clean_result = _run_quality_gates(tmp_path)
    assert clean_result["lint_ok"] is True


def test_workspace_outside_root_is_rejected(tmp_path: Path, monkeypatch) -> None:
    # The Governor's model chooses the argument; pytest executes conftest.py,
    # so an unconfined path is arbitrary code execution outside the sandbox.
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(SETTINGS, "workspace_root", str(root))
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    registry = ToolRegistry()
    register_governance_tools(registry)
    result = registry.dispatch(_make_call(str(outside)))

    assert result.error is not None
    assert "escapes" in result.error


def test_relative_dot_resolves_to_workspace_root(tmp_path: Path, monkeypatch) -> None:
    # Live runs showed the Governor passing "."; that must mean the swarm's
    # workspace, not whatever CWD the host process happens to have.
    root = tmp_path / "ws"
    root.mkdir()
    (root / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    monkeypatch.setattr(SETTINGS, "workspace_root", str(root))

    registry = ToolRegistry()
    register_governance_tools(registry)
    result = registry.dispatch(_make_call("."))

    assert result.error is None
    assert result.result["tests_ok"] is True


def test_nonexistent_workspace_raises() -> None:
    registry = ToolRegistry()
    register_governance_tools(registry)
    call = _make_call("/no/such/workspace/path/at/all")

    result = registry.dispatch(call)

    assert result.error is not None
