"""Real filesystem/shell/git tools for the Builder agent.

Opt-in only (see `presets.py::default_registry`) — every path argument is
routed through `resolve_safe_path` against `Settings.workspace_root` so a
model can't wander outside the sandboxed workspace, but `run_shell`/
`git_commit` still execute real subprocesses with no further sandboxing
(no container, no resource limits). Acceptable for a local single-user tool,
not safe against untrusted input.
"""

from __future__ import annotations

import shlex
import subprocess

from .config import SETTINGS
from .tools import ToolRegistry
from .workspace import resolve_safe_path

_TRUNCATE_CHARS = 4000


def _truncate(text: str) -> str:
    return text[-_TRUNCATE_CHARS:] if len(text) > _TRUNCATE_CHARS else text


def register_dev_tools(registry: ToolRegistry) -> None:
    @registry.register
    def read_file(path: str) -> str:
        """Read and return the contents of a file in the workspace."""
        target = resolve_safe_path(SETTINGS.workspace_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        return target.read_text()

    @registry.register
    def write_file(path: str, content: str) -> str:
        """Write content to a file in the workspace, creating parent dirs as needed."""
        target = resolve_safe_path(SETTINGS.workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    @registry.register
    def list_dir(path: str = ".") -> list[str]:
        """List entries in a workspace directory, sorted by name."""
        target = resolve_safe_path(SETTINGS.workspace_root, path)
        return sorted(entry.name for entry in target.iterdir())

    @registry.register
    def run_shell(command: str, timeout_s: int = 30) -> dict:
        """Run a shell command with cwd pinned to the workspace root."""
        root = resolve_safe_path(SETTINGS.workspace_root, ".")
        try:
            completed = subprocess.run(
                shlex.split(command),
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"timed out after {timeout_s}s"}
        return {
            "returncode": completed.returncode,
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
        }

    @registry.register
    def git_diff(path: str = ".") -> str:
        """Show the git diff for the workspace (empty string if no repo or no changes)."""
        root = resolve_safe_path(SETTINGS.workspace_root, ".")
        result = subprocess.run(
            ["git", "diff"],
            cwd=root,
            capture_output=True,
            text=True,
            shell=False,
        )
        return result.stdout

    @registry.register
    def git_commit(message: str) -> str:
        """Stage all changes and commit them, auto-initializing a repo if needed."""
        root = resolve_safe_path(SETTINGS.workspace_root, ".")
        if not (root / ".git").is_dir():
            subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, shell=False)
        # A fresh sandbox may have no git identity configured at any level; fall
        # back to a local one so commit can't fail with a confusing error.
        for key, value in (("user.email", "ollama-swarm@local"), ("user.name", "ollama-swarm")):
            configured = subprocess.run(["git", "config", key], cwd=root, capture_output=True, text=True, shell=False)
            if configured.returncode != 0:
                subprocess.run(["git", "config", key, value], cwd=root, capture_output=True, text=True, shell=False)
        subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True, text=True, shell=False)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=root,
            capture_output=True,
            text=True,
            shell=False,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output or "nothing to commit"
