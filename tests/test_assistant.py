from __future__ import annotations

import os
import json
import tempfile
import sys
from unittest.mock import patch, MagicMock
import pytest

from ollama_swarm.config import SETTINGS, MODEL_CATALOG, Tier
from ollama_swarm.assistant import load_config, parse_args, main

# Save original values so we can restore them after tests
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

def test_load_config():
    config_data = {
        "mode": "direct-cloud",
        "host": "http://myhost:11434",
        "api_key": "test_api_key",
        "timeout": 45.0,
        "max_tool_turns": 8,
        "workspace_root": "/tmp/test_workspace_root",
        "models": {
            "reasoning": ["model-a", "model-b"],
            "coding": ["model-c"]
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_name = tmp.name

    try:
        load_config(tmp_name)
        assert SETTINGS.mode == "direct-cloud"
        assert SETTINGS.host == "http://myhost:11434"
        assert SETTINGS.api_key == "test_api_key"
        assert SETTINGS.timeout_s == 45.0
        assert SETTINGS.max_tool_turns == 8
        assert SETTINGS.workspace_root == "/tmp/test_workspace_root"
        
        assert MODEL_CATALOG[Tier.REASONING] == ["model-a", "model-b"]
        assert MODEL_CATALOG[Tier.CODING] == ["model-c"]
    finally:
        os.unlink(tmp_name)

def test_cli_arg_overrides():
    # Precedence: CLI arguments > JSON configuration > Environment variables > Dataclass defaults
    config_data = {
        "mode": "daemon",
        "host": "http://config-host",
        "timeout": 30.0,
        "max_tool_turns": 5,
        "workspace_root": "/tmp/config_workspace"
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_name = tmp.name

    try:
        # Override via CLI
        test_argv = [
            "assistant",
            "--config", tmp_name,
            "--mode", "direct-cloud",
            "--host", "http://cli-host",
            "--api-key", "cli-key",
            "--timeout", "15.0",
            "--max-tool-turns", "3",
            "--output-dir", "/tmp/cli_workspace",
            "build", "a", "website"
        ]
        with patch.object(sys, "argv", test_argv):
            # We mock the entire Swarm.run and OllamaBackend and input/print
            with patch("ollama_swarm.assistant.OllamaBackend") as mock_backend_cls, \
                 patch("ollama_swarm.assistant.Swarm") as mock_swarm_cls:
                
                mock_backend = MagicMock()
                mock_backend_cls.return_value = mock_backend
                
                mock_swarm = MagicMock()
                mock_swarm.return_value = mock_swarm
                mock_swarm_cls.return_value = mock_swarm
                
                # Mock Swarm.run return value
                mock_result = MagicMock()
                mock_result.history = []
                mock_result.governed = True
                mock_result.security_verdict = ""
                mock_result.finops_summary = "FinOps details"
                mock_swarm.run.return_value = mock_result
                
                main()
                
                # Precedence verification: CLI overrides JSON
                assert SETTINGS.mode == "direct-cloud"
                assert SETTINGS.host == "http://cli-host"
                assert SETTINGS.api_key == "cli-key"
                assert SETTINGS.timeout_s == 15.0
                assert SETTINGS.max_tool_turns == 3
                assert SETTINGS.workspace_root == "/tmp/cli_workspace"
                
    finally:
        os.unlink(tmp_name)

def test_workspace_isolation_and_env_vars():
    # If no output-dir is specified, defaults to ~/teamwork_projects/ollama_developer_assistant
    test_argv = ["assistant", "build", "something"]
    with patch.object(sys, "argv", test_argv):
        with patch("ollama_swarm.assistant.OllamaBackend"), \
             patch("ollama_swarm.assistant.Swarm") as mock_swarm_cls:
            
            mock_swarm = MagicMock()
            mock_swarm_cls.return_value = mock_swarm
            mock_result = MagicMock()
            mock_result.history = []
            mock_result.security_verdict = ""
            mock_swarm.run.return_value = mock_result
            
            # Reset workspace root first
            SETTINGS.workspace_root = "./workspace"
            
            main()
            
            expected_default = os.path.abspath(os.path.expanduser("~/teamwork_projects/ollama_developer_assistant"))
            assert SETTINGS.workspace_root == expected_default
            assert os.path.isdir(expected_default)
            # Default dev tools should be enabled
            assert os.environ.get("OLLAMA_SWARM_ENABLE_DEV_TOOLS") == "1"

    # Test when --no-dev-tools is passed
    test_argv_no_dev = ["assistant", "--no-dev-tools", "build", "something"]
    with patch.object(sys, "argv", test_argv_no_dev):
        with patch("ollama_swarm.assistant.OllamaBackend"), \
             patch("ollama_swarm.assistant.Swarm") as mock_swarm_cls:
            
            mock_swarm = MagicMock()
            mock_swarm_cls.return_value = mock_swarm
            mock_result = MagicMock()
            mock_result.history = []
            mock_result.security_verdict = ""
            mock_swarm.run.return_value = mock_result
            
            main()
            assert os.environ.get("OLLAMA_SWARM_ENABLE_DEV_TOOLS") == "0"

def test_interactive_loop():
    # If no goal is passed, it should query reasoning model, prompt for answers, synthesize, and run swarm.
    test_argv = ["assistant"]
    mock_inputs = ["My cool software idea", "Answer 1", "Answer 2", "Answer 3"]
    
    with patch.object(sys, "argv", test_argv):
        with patch("builtins.input", side_effect=mock_inputs) as mock_input_func, \
             patch("ollama_swarm.assistant.OllamaBackend") as mock_backend_cls, \
             patch("ollama_swarm.assistant.Swarm") as mock_swarm_cls:
            
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            
            # Mock the chat calls for questions and synthesis
            mock_responses = [
                # Questions response
                ("glm-5:cloud", {"message": {"content": "1. Question A?\n2. Question B?\n3. Question C?\n"}}),
                # Synthesis response
                ("glm-5:cloud", {"message": {"content": "Synthesized goal details"}})
            ]
            mock_backend.chat_with_fallback.side_effect = mock_responses
            
            mock_swarm = MagicMock()
            mock_swarm_cls.return_value = mock_swarm
            mock_result = MagicMock()
            mock_result.history = []
            mock_result.governed = True
            mock_result.security_verdict = ""
            mock_result.finops_summary = "FinOps summary details"
            mock_swarm.run.return_value = mock_result
            
            main()
            
            # Verify input calls
            assert mock_input_func.call_count == 4
            # Verify chat_with_fallback calls
            assert mock_backend.chat_with_fallback.call_count == 2
            
            # Verify swarm.run is called with the synthesized goal
            mock_swarm.run.assert_called_once_with("Synthesized goal details")

def test_interactive_clarification():
    mock_backend = MagicMock()
    mock_responses = [
        ("glm-5:cloud", {"message": {"content": "1. 2D graphics options?\n2. What is the scope?\n- 3. Bullet question\n"}}),
        ("glm-5:cloud", {"message": {"content": "Synthesized goal details"}})
    ]
    mock_backend.chat_with_fallback.side_effect = mock_responses

    mock_inputs = ["Answer 1", "Answer 2", "Answer 3"]
    with patch("builtins.input", side_effect=mock_inputs) as mock_input_func, \
         patch("builtins.print") as mock_print:
        from ollama_swarm.assistant import interactive_clarification
        res = interactive_clarification(mock_backend, "My game idea", num_questions=3)
        assert res == "Synthesized goal details"
        assert mock_input_func.call_count == 3
        # Check printed questions to verify character stripping was resolved
        # The first question printed should be: "\nQuestion 1: 2D graphics options?"
        mock_print.assert_any_call("\nQuestion 1: 2D graphics options?")
        mock_print.assert_any_call("\nQuestion 2: What is the scope?")
        mock_print.assert_any_call("\nQuestion 3: Bullet question")
