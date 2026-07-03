from __future__ import annotations

from ollama_swarm.agents import Agent, run_agent
from ollama_swarm.config import Tier
from ollama_swarm.router import Router
from ollama_swarm.tools import ToolRegistry


def make_word_count_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def word_count(text: str) -> int:
        """Count words."""
        return len(text.split())

    return registry


def test_run_agent_dispatches_tool_call_then_returns_final_answer(fake_backend) -> None:
    calls = {"n": 0}

    def script(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "word_count", "arguments": {"text": "hello there world"}}}],
                }
            }
        # second call: the tool result should be in the transcript by now
        assert any(m["role"] == "tool" for m in messages)
        return {"message": {"content": "The text has 3 words."}}

    backend = fake_backend(script)
    agent = Agent(name="Builder", system_prompt="test", tier=Tier.CODING, tools=["word_count"])
    result = run_agent(agent, "count words in: hello there world", backend, Router(), make_word_count_registry())

    assert result.tool_calls_made == 1
    assert "3 words" in result.content


def test_run_agent_returns_immediately_when_no_tool_calls(fake_backend) -> None:
    backend = fake_backend(lambda messages: {"message": {"content": "done, no tools needed"}})
    agent = Agent(name="Planner", system_prompt="test", tier=Tier.REASONING)

    result = run_agent(agent, "plan something", backend, Router(), ToolRegistry())

    assert result.tool_calls_made == 0
    assert result.content == "done, no tools needed"


def test_run_agent_stops_at_max_turn_budget(fake_backend, monkeypatch) -> None:
    import ollama_swarm.agents as agents_module

    monkeypatch.setattr(agents_module.SETTINGS, "max_tool_turns", 2)

    def always_calls_tool(messages):
        return {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "word_count", "arguments": {"text": "a"}}}],
            }
        }

    backend = fake_backend(always_calls_tool)
    agent = Agent(name="Builder", system_prompt="test", tier=Tier.CODING, tools=["word_count"])
    result = run_agent(agent, "loop forever", backend, Router(), make_word_count_registry())

    assert "max tool turns" in result.content
    assert result.tool_calls_made == 2
