# Placeholder for Package A/B — overwritten during integration.

from __future__ import annotations

from .tools import ToolRegistry


def register_dev_tools(registry: ToolRegistry) -> None:
    @registry.register
    def read_file(path: str) -> str:
        """Read a file's contents."""
        return "stub"

    @registry.register
    def write_file(path: str, content: str) -> str:
        """Write content to a file."""
        return "stub"

    @registry.register
    def list_dir(path: str = ".") -> str:
        """List a directory's contents."""
        return "stub"

    @registry.register
    def run_shell(command: str, timeout_s: int = 30) -> dict:
        """Run a shell command."""
        return {"returncode": 0, "stdout": "stub", "stderr": ""}

    @registry.register
    def git_diff(path: str = ".") -> str:
        """Show the git diff."""
        return "stub"

    @registry.register
    def git_commit(message: str) -> str:
        """Create a git commit."""
        return "stub"
