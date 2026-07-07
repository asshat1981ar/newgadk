"""Tests for security_gates.py — scanner dispatch, verdict classification, and WARN-on-missing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ollama_swarm.config import SETTINGS
from ollama_swarm.security_gates import (
    run_security_scan,
    _run_bandit,
    _run_npm_audit,
    _run_cargo_audit,
    _sniff_language,
)


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path):
    orig = SETTINGS.workspace_root
    SETTINGS.workspace_root = str(tmp_path)
    yield tmp_path
    SETTINGS.workspace_root = orig


# ---------------------------------------------------------------------------
# _sniff_language
# ---------------------------------------------------------------------------

class TestSniffLanguage:
    def test_detects_python_from_pyproject(self, isolated_workspace):
        (isolated_workspace / "pyproject.toml").write_text("[project]")
        assert _sniff_language(str(isolated_workspace)) == "python"

    def test_detects_python_from_py_file(self, isolated_workspace):
        (isolated_workspace / "main.py").write_text("")
        assert _sniff_language(str(isolated_workspace)) == "python"

    def test_detects_node_from_package_json(self, isolated_workspace):
        (isolated_workspace / "package.json").write_text("{}")
        assert _sniff_language(str(isolated_workspace)) == "node"

    def test_detects_rust_from_cargo_toml(self, isolated_workspace):
        (isolated_workspace / "Cargo.toml").write_text("")
        assert _sniff_language(str(isolated_workspace)) == "rust"

    def test_generic_for_empty_dir(self, isolated_workspace):
        assert _sniff_language(str(isolated_workspace)) == "generic"


# ---------------------------------------------------------------------------
# _run_bandit
# ---------------------------------------------------------------------------

class TestRunBandit:
    def test_warn_when_bandit_not_installed(self, isolated_workspace):
        with patch("shutil.which", return_value=None):
            result = _run_bandit(str(isolated_workspace))
        assert result["ok"] is None
        assert result["severity"] == "warn"
        assert "not installed" in result["findings"]

    def test_ok_when_bandit_clean(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "No issues identified."
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/bandit"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_bandit(str(isolated_workspace))
        assert result["ok"] is True
        assert result["severity"] == "ok"

    def test_warn_on_low_severity(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "Severity: Low - some minor thing"
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/bandit"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_bandit(str(isolated_workspace))
        assert result["severity"] == "warn"

    def test_no_go_on_high_severity(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "Severity: High - SQL injection detected"
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/bandit"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_bandit(str(isolated_workspace))
        assert result["severity"] == "no-go"

    def test_warn_on_timeout(self, isolated_workspace):
        import subprocess
        with patch("shutil.which", return_value="/usr/bin/bandit"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("bandit", 60)):
            result = _run_bandit(str(isolated_workspace))
        assert result["severity"] == "warn"
        assert "timed out" in result["findings"]


# ---------------------------------------------------------------------------
# _run_npm_audit
# ---------------------------------------------------------------------------

class TestRunNpmAudit:
    def test_warn_when_npm_not_installed(self, isolated_workspace):
        with patch("shutil.which", return_value=None):
            result = _run_npm_audit(str(isolated_workspace))
        assert result["ok"] is None
        assert result["severity"] == "warn"
        assert "not installed" in result["findings"]

    def test_ok_when_audit_clean(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"vulnerabilities": {}}'
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/npm"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_npm_audit(str(isolated_workspace))
        assert result["ok"] is True
        assert result["severity"] == "ok"

    def test_no_go_on_critical(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = '{"severity":"critical","name":"lodash"}'
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/npm"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_npm_audit(str(isolated_workspace))
        assert result["severity"] == "no-go"

    def test_no_go_on_high(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = '{"severity":"high","name":"express"}'
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/npm"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_npm_audit(str(isolated_workspace))
        assert result["severity"] == "no-go"


# ---------------------------------------------------------------------------
# _run_cargo_audit
# ---------------------------------------------------------------------------

class TestRunCargoAudit:
    def test_warn_when_cargo_not_installed(self, isolated_workspace):
        with patch("shutil.which", return_value=None):
            result = _run_cargo_audit(str(isolated_workspace))
        assert result["ok"] is None
        assert result["severity"] == "warn"

    def test_ok_when_audit_clean(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "No vulnerabilities found"
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/cargo"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_cargo_audit(str(isolated_workspace))
        assert result["ok"] is True
        assert result["severity"] == "ok"

    def test_no_go_on_vulnerabilities(self, isolated_workspace):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "error[vulnerability]: RUSTSEC-2022-0001"
        mock_proc.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/cargo"), \
             patch("subprocess.run", return_value=mock_proc):
            result = _run_cargo_audit(str(isolated_workspace))
        assert result["severity"] == "no-go"


# ---------------------------------------------------------------------------
# run_security_scan — integration
# ---------------------------------------------------------------------------

class TestRunSecurityScan:
    def test_includes_language_in_result(self, isolated_workspace):
        (isolated_workspace / "pyproject.toml").write_text("[project]")
        with patch("shutil.which", return_value=None):
            result = run_security_scan(str(isolated_workspace))
        assert result["language"] == "python"

    def test_generic_language_gives_warn(self, isolated_workspace):
        # No recognized files → generic → no scanner → WARN
        result = run_security_scan(str(isolated_workspace))
        assert result["severity"] == "warn"
        assert result["language"] == "generic"

    def test_uses_settings_workspace_when_empty_string(self, isolated_workspace):
        # Pass empty string — should default to SETTINGS.workspace_root
        result = run_security_scan("")
        assert "language" in result

    def test_python_workspace_routes_to_bandit(self, isolated_workspace):
        (isolated_workspace / "pyproject.toml").write_text("[project]")
        with patch("ollama_swarm.security_gates._run_bandit") as mock_bandit:
            mock_bandit.return_value = {"ok": True, "severity": "ok", "findings": "clean"}
            result = run_security_scan(str(isolated_workspace))
        mock_bandit.assert_called_once_with(str(isolated_workspace))

    def test_node_workspace_routes_to_npm_audit(self, isolated_workspace):
        (isolated_workspace / "package.json").write_text("{}")
        with patch("ollama_swarm.security_gates._run_npm_audit") as mock_npm:
            mock_npm.return_value = {"ok": True, "severity": "ok", "findings": "clean"}
            result = run_security_scan(str(isolated_workspace))
        mock_npm.assert_called_once_with(str(isolated_workspace))

    def test_rust_workspace_routes_to_cargo_audit(self, isolated_workspace):
        (isolated_workspace / "Cargo.toml").write_text("")
        with patch("ollama_swarm.security_gates._run_cargo_audit") as mock_cargo:
            mock_cargo.return_value = {"ok": True, "severity": "ok", "findings": "clean"}
            result = run_security_scan(str(isolated_workspace))
        mock_cargo.assert_called_once_with(str(isolated_workspace))
