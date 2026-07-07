"""Tests for scaffold.py — language detection and project skeleton creation."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ollama_swarm.scaffold import derive_project_name, detect_language, scaffold_project
from ollama_swarm.config import SETTINGS


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path):
    """Pin SETTINGS.workspace_root to a tmp dir for every test."""
    orig = SETTINGS.workspace_root
    SETTINGS.workspace_root = str(tmp_path)
    yield tmp_path
    SETTINGS.workspace_root = orig


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_detects_python(self):
        assert detect_language("We'll build this in Python using pytest and Flask.") == "python"

    def test_detects_node(self):
        assert detect_language("Build a Node.js Express REST API.") == "node"

    def test_detects_rust(self):
        assert detect_language("Create a Rust CLI tool using Cargo and tokio.") == "rust"

    def test_detects_shell(self):
        assert detect_language("Write a bash shell script that deploys artifacts.") == "shell"

    def test_generic_fallback(self):
        assert detect_language("Build something cool and exciting.") == "generic"

    def test_case_insensitive(self):
        assert detect_language("PYTHON FLASK PYTEST") == "python"

    def test_highest_score_wins(self):
        # More python keywords than node
        text = "Python flask pytest pip pyproject but also some javascript"
        assert detect_language(text) == "python"

    def test_empty_string(self):
        assert detect_language("") == "generic"


# ---------------------------------------------------------------------------
# derive_project_name
# ---------------------------------------------------------------------------

class TestDeriveProjectName:
    def test_plain_goal_uses_first_word(self):
        assert derive_project_name("todo app with reminders") == "todo"

    def test_skips_markdown_lead_in(self):
        # Synthesized goals open with markdown; "**Synthesized" must not
        # become "__synthesized".
        goal = "**Synthesized Software Development Goal:** build an add function"
        assert derive_project_name(goal) == "synthesized"

    def test_skips_pure_punctuation_words(self):
        assert derive_project_name("--- ### calculator project") == "calculator"

    def test_skips_words_starting_with_digit(self):
        assert derive_project_name("2048 game clone") == "game"

    def test_empty_goal_falls_back(self):
        assert derive_project_name("") == "project"

    def test_symbol_only_goal_falls_back(self):
        assert derive_project_name("*** !!! ???") == "project"


# ---------------------------------------------------------------------------
# scaffold_project — Python
# ---------------------------------------------------------------------------

class TestScaffoldPython:
    def test_creates_pyproject_toml(self, isolated_workspace):
        scaffold_project("This is a Python Flask application using pytest.", "my_app")
        assert (isolated_workspace / "pyproject.toml").exists()

    def test_creates_src_package(self, isolated_workspace):
        scaffold_project("Python tool", "word_counter")
        src_dir = isolated_workspace / "src" / "word_counter"
        assert src_dir.is_dir()
        assert (src_dir / "__init__.py").exists()
        assert (src_dir / "main.py").exists()

    def test_creates_tests_directory(self, isolated_workspace):
        scaffold_project("Python tool", "word_counter")
        assert (isolated_workspace / "tests" / "__init__.py").exists()
        assert (isolated_workspace / "tests" / "test_main.py").exists()

    def test_creates_gitignore(self, isolated_workspace):
        scaffold_project("Python tool", "my_app")
        assert (isolated_workspace / ".gitignore").exists()

    def test_creates_readme(self, isolated_workspace):
        scaffold_project("Python tool", "my_app")
        readme = (isolated_workspace / "README.md").read_text()
        assert "my_app" in readme

    def test_safe_project_name_sanitization(self, isolated_workspace):
        # Hyphens and spaces become underscores in Python package name
        scaffold_project("Python tool", "My Cool App!")
        pyproject = (isolated_workspace / "pyproject.toml").read_text()
        assert "my_cool_app_" in pyproject or "my_cool_app" in pyproject

    def test_does_not_overwrite_existing_files(self, isolated_workspace):
        # Write a sentinel value to pyproject.toml first
        (isolated_workspace / "pyproject.toml").write_text("SENTINEL")
        scaffold_project("Python tool", "my_app")
        assert (isolated_workspace / "pyproject.toml").read_text() == "SENTINEL"

    def test_returns_written_files_mapping(self, isolated_workspace):
        written = scaffold_project("Python pytest tool", "counter")
        assert isinstance(written, dict)
        assert len(written) > 0
        for rel_path in written:
            assert (isolated_workspace / rel_path).exists()

    def test_returns_empty_if_all_exist(self, isolated_workspace):
        scaffold_project("Python tool", "counter")
        # Second call — all files already exist
        written = scaffold_project("Python tool", "counter")
        assert written == {}

    def test_garbage_project_name_falls_back(self, isolated_workspace):
        scaffold_project("Python tool", "**??!")
        assert (isolated_workspace / "src" / "project").is_dir()

    def test_placeholder_test_imports_nothing_from_package(self, isolated_workspace):
        # The Builder may rename or delete the scaffolded package; the
        # placeholder must not depend on it.
        scaffold_project("Python tool", "my_app")
        content = (isolated_workspace / "tests" / "test_main.py").read_text()
        assert "my_app" not in content
        tree = ast.parse(content)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert imports == []

    def test_placeholder_tests_pass_quality_gate(self, isolated_workspace):
        # The Governor runs pytest in the workspace; a fresh scaffold must
        # pass that gate before the Builder has written anything.
        scaffold_project("Python tool", "my_app")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=isolated_workspace,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# scaffold_project — Node
# ---------------------------------------------------------------------------

class TestScaffoldNode:
    def test_creates_package_json(self, isolated_workspace):
        scaffold_project("Build a Node.js Express API.", "api_server")
        assert (isolated_workspace / "package.json").exists()

    def test_creates_src_index(self, isolated_workspace):
        scaffold_project("Build a Node.js app.", "my_app")
        assert (isolated_workspace / "src" / "index.js").exists()

    def test_creates_test_file(self, isolated_workspace):
        scaffold_project("Node app", "my_app")
        assert (isolated_workspace / "tests" / "index.test.js").exists()

    def test_package_json_has_name(self, isolated_workspace):
        scaffold_project("Build a Node.js app.", "my-app")
        content = (isolated_workspace / "package.json").read_text()
        assert "my-app" in content


# ---------------------------------------------------------------------------
# scaffold_project — Rust
# ---------------------------------------------------------------------------

class TestScaffoldRust:
    def test_creates_cargo_toml(self, isolated_workspace):
        scaffold_project("Build a Rust CLI using cargo and tokio.", "my_tool")
        assert (isolated_workspace / "Cargo.toml").exists()

    def test_creates_main_rs(self, isolated_workspace):
        scaffold_project("Rust tool", "my_tool")
        assert (isolated_workspace / "src" / "main.rs").exists()

    def test_creates_lib_rs(self, isolated_workspace):
        scaffold_project("Rust tool", "my_tool")
        assert (isolated_workspace / "src" / "lib.rs").exists()


# ---------------------------------------------------------------------------
# scaffold_project — Shell
# ---------------------------------------------------------------------------

class TestScaffoldShell:
    def test_creates_sh_script(self, isolated_workspace):
        scaffold_project("Write a bash shell deployment script.", "deploy")
        sh_files = list(isolated_workspace.glob("*.sh"))
        assert len(sh_files) == 1

    def test_sh_script_has_shebang(self, isolated_workspace):
        scaffold_project("bash shell script for CI", "ci_deploy")
        sh_file = next(isolated_workspace.glob("*.sh"))
        assert sh_file.read_text().startswith("#!/usr/bin/env bash")


# ---------------------------------------------------------------------------
# scaffold_project — Generic fallback
# ---------------------------------------------------------------------------

class TestScaffoldGeneric:
    def test_creates_readme(self, isolated_workspace):
        scaffold_project("Build something cool", "cool_thing")
        assert (isolated_workspace / "README.md").exists()

    def test_creates_main_placeholder(self, isolated_workspace):
        scaffold_project("Build something cool", "cool_thing")
        assert (isolated_workspace / "main.txt").exists()
