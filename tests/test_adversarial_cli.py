from __future__ import annotations

import os
import json
import tempfile
import sys
from unittest.mock import patch, MagicMock
import pytest

from ollama_swarm.config import SETTINGS, MODEL_CATALOG, Tier
from ollama_swarm.assistant import load_config, main, parse_args
from ollama_swarm.workspace import resolve_safe_path

@pytest.fixture(autouse=True)
def restore_settings_and_catalog():
    orig_mode = SETTINGS.mode
    orig_host = SETTINGS.host
    orig_cloud_host = SETTINGS.cloud_host
    orig_api_key = SETTINGS.api_key
    orig_timeout = SETTINGS.timeout_s
    orig_max_turns = SETTINGS.max_tool_turns
    orig_workspace = SETTINGS.workspace_root
    orig_catalog = {tier: list(models) for tier, models in MODEL_CATALOG.items()}
    
    yield
    
    SETTINGS.mode = orig_mode
    SETTINGS.host = orig_host
    SETTINGS.cloud_host = orig_cloud_host
    SETTINGS.api_key = orig_api_key
    SETTINGS.timeout_s = orig_timeout
    SETTINGS.max_tool_turns = orig_max_turns
    SETTINGS.workspace_root = orig_workspace
    MODEL_CATALOG.clear()
    MODEL_CATALOG.update(orig_catalog)

# Fixture to monkeypatch expanduser to prevent test pollution in home directory
@pytest.fixture(autouse=True)
def mock_expanduser(monkeypatch, tmp_path):
    original_expanduser = os.path.expanduser
    def fake_expanduser(path):
        if path.startswith("~"):
            return path.replace("~", str(tmp_path), 1)
        return original_expanduser(path)
    monkeypatch.setattr(os.path, "expanduser", fake_expanduser)

# --- 1. CONFIG LOADING WITH MALFORMED OR CORRUPTED DATA ---

def test_load_config_malformed_json(capsys):
    """Verify that a malformed JSON file raises an exception and prints to stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write("{ invalid json: ")
        tmp_name = tmp.name

    try:
        with pytest.raises(Exception):
            load_config(tmp_name)
        captured = capsys.readouterr()
        assert f"Error loading config file {tmp_name}" in captured.err
    finally:
        os.unlink(tmp_name)

def test_load_config_invalid_type_timeout():
    """Verify that a non-float timeout value in config raises ValueError and is propagated."""
    config_data = {
        "timeout": "not-a-number"
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_name = tmp.name

    try:
        with pytest.raises(ValueError) as exc_info:
            load_config(tmp_name)
        assert "could not convert string to float" in str(exc_info.value)
    finally:
        os.unlink(tmp_name)

def test_load_config_invalid_type_models():
    """Verify that a non-iterable model list in config raises TypeError and is propagated."""
    config_data = {
        "models": {
            "reasoning": 12345
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_name = tmp.name

    try:
        with pytest.raises(TypeError) as exc_info:
            load_config(tmp_name)
        assert "must be a list/sequence" in str(exc_info.value)
    finally:
        os.unlink(tmp_name)

def test_load_config_string_json():
    """Verify that if the JSON config resolves to a string instead of a dict, it raises TypeError and is propagated."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump("mode is daemon", tmp)
        tmp_name = tmp.name

    try:
        with pytest.raises(TypeError) as exc_info:
            load_config(tmp_name)
        assert "must be a dictionary" in str(exc_info.value)
    finally:
        os.unlink(tmp_name)

# --- 2. COMMAND LINE OVERRIDE PRECEDENCE EDGE SCENARIOS ---

def test_workspace_root_override_precedence_bug():
    """Verify that if the config file specifies workspace_root as './workspace',
    the CLI overriding logic preserves this value and does not override it
    with the default '~/teamwork_projects/ollama_developer_assistant'."""
    config_data = {
        "workspace_root": "./workspace"
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_name = tmp.name

    test_argv = ["assistant", "--config", tmp_name, "build", "something"]
    try:
        with patch.object(sys, "argv", test_argv):
            with patch("ollama_swarm.assistant.OllamaBackend"), \
                 patch("ollama_swarm.assistant.Swarm") as mock_swarm_cls:
                
                mock_swarm = MagicMock()
                mock_swarm_cls.return_value = mock_swarm
                mock_result = MagicMock()
                mock_result.history = []
                mock_swarm.run.return_value = mock_result
                
                # We start with workspace_root as the default dataclass value
                SETTINGS.workspace_root = "./workspace"
                
                main()
                
                expected_config_val = os.path.abspath(os.path.expanduser("./workspace"))
                actual_val = SETTINGS.workspace_root
                
                # Verify that the explicit configuration was preserved and not overridden
                assert actual_val == expected_config_val
    finally:
        os.unlink(tmp_name)

# --- 3. WORKSPACE ISOLATION & DIRECTORY TRAVERSAL ---

def test_directory_traversal_absolute_path(tmp_path):
    """Verify that absolute paths outside the workspace are blocked by resolve_safe_path."""
    root = str(tmp_path / "workspace")
    with pytest.raises(ValueError) as exc_info:
        resolve_safe_path(root, "/etc/passwd")
    assert "escapes workspace root" in str(exc_info.value)

def test_directory_traversal_relative_parent(tmp_path):
    """Verify that parent directory traversals ('../') are blocked by resolve_safe_path."""
    root = str(tmp_path / "workspace")
    with pytest.raises(ValueError) as exc_info:
        resolve_safe_path(root, "../../etc/passwd")
    assert "escapes workspace root" in str(exc_info.value)

def test_directory_traversal_via_symlink(tmp_path):
    """Verify that resolve_safe_path resolves symlinks and blocks traversals pointing outside the root."""
    root = tmp_path / "workspace"
    root.mkdir()
    
    # Create external target
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_file = external_dir / "secret.txt"
    external_file.write_text("my-secret")
    
    # Create symlink inside workspace pointing to the external file
    symlink_path = root / "evil_symlink"
    os.symlink(str(external_file), str(symlink_path))
    
    with pytest.raises(ValueError) as exc_info:
        resolve_safe_path(str(root), "evil_symlink")
    assert "escapes workspace root" in str(exc_info.value)

def test_null_byte_in_path(tmp_path):
    """Verify that null bytes in paths are blocked or raise an exception."""
    root = str(tmp_path / "workspace")
    with pytest.raises((ValueError, TypeError)):
        resolve_safe_path(root, "file.txt\x00evil")
