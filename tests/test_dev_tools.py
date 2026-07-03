from __future__ import annotations

import subprocess

import pytest

import ollama_swarm.config as config_module
import ollama_swarm.dev_tools as dev_tools_module
from ollama_swarm.tools import ToolRegistry
from ollama_swarm.workspace import resolve_safe_path


@pytest.fixture
def registry(tmp_path, monkeypatch) -> ToolRegistry:
    monkeypatch.setattr(config_module.SETTINGS, "workspace_root", str(tmp_path))
    registry = ToolRegistry()
    dev_tools_module.register_dev_tools(registry)
    return registry


def _call(registry: ToolRegistry, name: str, **kwargs):
    from types import SimpleNamespace

    tool_call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=kwargs))
    result = registry.dispatch(tool_call)
    assert result.error is None, result.error
    return result.result


def test_read_write_round_trip(registry: ToolRegistry) -> None:
    written = _call(registry, "write_file", path="notes/hello.txt", content="hi there")
    assert "wrote" in written
    assert _call(registry, "read_file", path="notes/hello.txt") == "hi there"


def test_read_file_missing_raises_clear_error(registry: ToolRegistry) -> None:
    from types import SimpleNamespace

    tool_call = SimpleNamespace(function=SimpleNamespace(name="read_file", arguments={"path": "nope.txt"}))
    result = registry.dispatch(tool_call)
    assert result.error is not None


def test_path_traversal_is_blocked_at_resolve_safe_path(tmp_path) -> None:
    with pytest.raises(ValueError):
        resolve_safe_path(str(tmp_path), "../../etc/passwd")


def test_path_traversal_is_blocked_through_a_tool(registry: ToolRegistry) -> None:
    from types import SimpleNamespace

    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="write_file", arguments={"path": "../../etc/passwd", "content": "pwned"})
    )
    result = registry.dispatch(tool_call)
    assert result.error is not None
    assert "escapes" in result.error


def test_list_dir_returns_sorted_entries(registry: ToolRegistry) -> None:
    _call(registry, "write_file", path="b.txt", content="b")
    _call(registry, "write_file", path="a.txt", content="a")

    entries = _call(registry, "list_dir")

    assert entries == ["a.txt", "b.txt"]


def test_run_shell_runs_a_portable_command(registry: ToolRegistry) -> None:
    result = _call(registry, "run_shell", command="python3 -c \"print(1)\"")

    assert result["returncode"] == 0
    assert "1" in result["stdout"]


def test_run_shell_timeout_reports_negative_returncode_instead_of_raising(registry: ToolRegistry) -> None:
    result = _call(
        registry, "run_shell", command="python3 -c \"import time; time.sleep(5)\"", timeout_s=1
    )

    assert result["returncode"] == -1
    assert "timed out" in result["stderr"]


def test_git_diff_and_commit_against_a_real_repo(registry: ToolRegistry, tmp_path) -> None:
    # `git diff` alone doesn't show untracked files, so the meaningful assertion
    # is the pre-commit -> commit -> post-commit cycle: diff is non-empty for a
    # tracked-then-modified file, and empty again once committed.
    _call(registry, "write_file", path="README.md", content="v1\n")
    first_commit = _call(registry, "git_commit", message="initial commit")
    assert first_commit

    _call(registry, "write_file", path="README.md", content="v2\n")
    diff_after_edit = _call(registry, "git_diff")
    assert "v2" in diff_after_edit

    _call(registry, "git_commit", message="update readme")
    diff_after_commit = _call(registry, "git_diff")
    assert diff_after_commit == ""

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "initial commit" in log.stdout
    assert "update readme" in log.stdout


def test_workspace_root_field_defaults_to_dedicated_subdir(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_SWARM_WORKSPACE", raising=False)
    settings = config_module.Settings()
    assert settings.workspace_root == "./workspace"
