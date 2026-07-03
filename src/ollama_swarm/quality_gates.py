"""Governance tools: a real quality gate the Governor agent can run instead of
trusting the Critic's word alone. One tool, narrowly scoped (Governor gets only
this, not a general shell), so a review agent can call real `pytest`/`ruff`
processes and ground its GO/NO-GO verdict in actual output."""

from __future__ import annotations

import shutil
import subprocess

from .config import SETTINGS
from .tools import ToolRegistry

_MAX_OUTPUT_CHARS = 4000
_TESTS_TIMEOUT_S = 60
_LINT_TIMEOUT_S = 30
# pytest exits 5 ("no tests collected") when a workspace has no test files at
# all. Treated as passing here: an empty/pre-test workspace shouldn't block a
# gate that exists to catch regressions, not to mandate test coverage.
_PYTEST_NO_TESTS_COLLECTED = 5


def _truncate(text: str) -> str:
    return text[-_MAX_OUTPUT_CHARS:] if len(text) > _MAX_OUTPUT_CHARS else text


def register_governance_tools(registry: ToolRegistry) -> None:
    @registry.register
    def run_quality_gates(workspace: str = "") -> dict:
        """Run the test suite (and linter, if available) for a workspace."""
        # Empty default resolves to the swarm's own workspace, not the host
        # process CWD — the Governor gates the work the swarm produced, not the
        # project the swarm happens to be running from.
        workspace = workspace or SETTINGS.workspace_root
        try:
            proc = subprocess.run(
                ["python3", "-m", "pytest", "-q"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_TESTS_TIMEOUT_S,
            )
            tests_ok = proc.returncode == 0 or proc.returncode == _PYTEST_NO_TESTS_COLLECTED
            tests_output = _truncate(proc.stdout + proc.stderr)
        except subprocess.TimeoutExpired:
            tests_ok = False
            tests_output = f"pytest timed out after {_TESTS_TIMEOUT_S}s"

        lint_ok: bool | None
        if shutil.which("ruff") is None:
            lint_ok = None
            lint_output = "ruff not installed, lint skipped"
        else:
            try:
                proc = subprocess.run(
                    ["ruff", "check", "."],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=_LINT_TIMEOUT_S,
                )
                lint_ok = proc.returncode == 0
                lint_output = _truncate(proc.stdout + proc.stderr)
            except subprocess.TimeoutExpired:
                lint_ok = False
                lint_output = f"ruff timed out after {_LINT_TIMEOUT_S}s"

        return {
            "tests_ok": tests_ok,
            "tests_output": tests_output,
            "lint_ok": lint_ok,
            "lint_output": lint_output,
        }
