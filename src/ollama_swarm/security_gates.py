"""Security scanning tool for the Security agent phase.

Runs language-appropriate static-analysis / dependency-audit tools against the
generated workspace.  When the required scanner is not installed the result is
WARN (not NO-GO) — the user explicitly chose this behaviour.

Supported scanners:
  python  → bandit  (pip install bandit)
  node    → npm audit
  rust    → cargo audit  (cargo install cargo-audit)
  generic → skip (WARN)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import SETTINGS
from .scaffold import detect_language
from .tools import ToolRegistry
from .workspace import resolve_safe_path

_MAX_OUTPUT_CHARS = 4000
_SCAN_TIMEOUT_S = 60


def _truncate(text: str) -> str:
    return text[-_MAX_OUTPUT_CHARS:] if len(text) > _MAX_OUTPUT_CHARS else text


def _sniff_language(workspace: str) -> str:
    """Infer the project language from workspace file layout."""
    root = Path(workspace)
    if (root / "pyproject.toml").exists() or list(root.glob("**/*.py")):
        return "python"
    if (root / "package.json").exists():
        return "node"
    if (root / "Cargo.toml").exists():
        return "rust"
    return "generic"


def _run_bandit(workspace: str) -> dict:
    if shutil.which("bandit") is None:
        return {
            "ok": None,
            "severity": "warn",
            "findings": "bandit not installed — security scan skipped (WARN)",
        }
    try:
        proc = subprocess.run(
            ["bandit", "-r", ".", "-f", "text", "-ll"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_SCAN_TIMEOUT_S,
        )
        # bandit exit codes: 0 = no issues, 1 = issues found, >1 = error
        ok = proc.returncode == 0
        output = _truncate(proc.stdout + proc.stderr)
        severity = "ok" if ok else "warn"
        # High severity issues → NO-GO
        if "Severity: High" in output or "Issue: [B" in output and "HIGH" in output.upper():
            severity = "no-go"
        return {"ok": ok, "severity": severity, "findings": output}
    except subprocess.TimeoutExpired:
        return {"ok": False, "severity": "warn", "findings": f"bandit timed out after {_SCAN_TIMEOUT_S}s"}


def _run_npm_audit(workspace: str) -> dict:
    if shutil.which("npm") is None:
        return {
            "ok": None,
            "severity": "warn",
            "findings": "npm not installed — security scan skipped (WARN)",
        }
    try:
        proc = subprocess.run(
            ["npm", "audit", "--json"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_SCAN_TIMEOUT_S,
        )
        output = _truncate(proc.stdout + proc.stderr)
        ok = proc.returncode == 0
        severity = "ok" if ok else "warn"
        if '"severity":"critical"' in output or '"severity":"high"' in output:
            severity = "no-go"
        return {"ok": ok, "severity": severity, "findings": output}
    except subprocess.TimeoutExpired:
        return {"ok": False, "severity": "warn", "findings": f"npm audit timed out after {_SCAN_TIMEOUT_S}s"}


def _run_cargo_audit(workspace: str) -> dict:
    if shutil.which("cargo") is None:
        return {
            "ok": None,
            "severity": "warn",
            "findings": "cargo not installed — security scan skipped (WARN)",
        }
    try:
        proc = subprocess.run(
            ["cargo", "audit"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_SCAN_TIMEOUT_S,
        )
        output = _truncate(proc.stdout + proc.stderr)
        ok = proc.returncode == 0
        severity = "ok" if ok else "no-go"
        return {"ok": ok, "severity": severity, "findings": output}
    except subprocess.TimeoutExpired:
        return {"ok": False, "severity": "warn", "findings": f"cargo audit timed out after {_SCAN_TIMEOUT_S}s"}


def _get_scanner(lang: str):
    """Return the scanner callable for *lang*, looked up dynamically so tests can patch individual scanners."""
    import ollama_swarm.security_gates as _m
    return {
        "python": _m._run_bandit,
        "node":   _m._run_npm_audit,
        "rust":   _m._run_cargo_audit,
    }.get(lang)


def run_security_scan(workspace: str = "") -> dict:
    """Run the appropriate security scanner for the workspace language.

    Returns:
        {
            "language": str,
            "ok": bool | None,   # None = skipped (scanner missing)
            "severity": "ok" | "warn" | "no-go",
            "findings": str,
        }
    """
    # A model-supplied path is confined like every other tool's: scanners
    # execute against real files, so an unconfined path points them (and any
    # subprocess they spawn) at arbitrary directories on disk.
    if workspace:
        workspace = str(resolve_safe_path(SETTINGS.workspace_root, workspace))
    else:
        workspace = SETTINGS.workspace_root
    lang = _sniff_language(workspace)
    scanner = _get_scanner(lang)
    if scanner is None:
        result: dict = {
            "ok": None,
            "severity": "warn",
            "findings": f"no scanner available for language '{lang}' — skipped (WARN)",
        }
    else:
        result = scanner(workspace)
    result["language"] = lang
    return result


def register_security_tools(registry: ToolRegistry) -> None:
    registry.register(run_security_scan)
