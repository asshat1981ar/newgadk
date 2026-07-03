# Placeholder for Package A/B — overwritten during integration.

from __future__ import annotations

from .tools import ToolRegistry


def register_governance_tools(registry: ToolRegistry) -> None:
    @registry.register
    def run_quality_gates(workspace: str = ".") -> dict:
        """Run the project's test suite (and linter, if available)."""
        return {
            "tests_ok": True,
            "tests_output": "stub",
            "lint_ok": None,
            "lint_output": "stub",
        }
