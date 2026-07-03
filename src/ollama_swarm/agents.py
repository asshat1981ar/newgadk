"""An agent is data (name, prompt, tier, tools), and running one is a loop.

GADK's agents (src/agents/*.py) each carry a conditional `google.adk.agents.Agent`
import gated behind `Config.TEST_MODE`, so every agent file is really two agents:
a real ADK-wrapped one and a mock. Dropping the ADK dependency entirely and talking
to Ollama directly removes that split — one code path, real calls in tests are just
a fake `OllamaBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backend import OllamaBackend
from .config import SETTINGS, Tier
from .router import Router
from .tools import ToolRegistry


@dataclass
class Agent:
    name: str
    system_prompt: str
    tier: Tier
    tools: list[str] = field(default_factory=list)  # names registered in a ToolRegistry


@dataclass
class AgentRunResult:
    model_used: str
    content: str
    transcript: list[dict[str, Any]]
    tool_calls_made: int


def run_agent(
    agent: Agent,
    user_message: str,
    backend: OllamaBackend,
    router: Router,
    registry: ToolRegistry,
    context: str | None = None,
) -> AgentRunResult:
    """Run one agent to completion: call the model, dispatch any tool calls it
    asks for, feed results back, repeat until it answers in plain text or the
    turn budget (`Settings.max_tool_turns`) runs out."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": agent.system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Relevant context:\n{context}"})
    messages.append({"role": "user", "content": user_message})

    tools_schema = registry.schemas(agent.tools) if agent.tools else None
    model_used = ""
    tool_calls_made = 0

    for _ in range(SETTINGS.max_tool_turns):
        chain = router.fallback_chain(agent.tier)

        def _on_fail(model: str, exc: Exception) -> None:
            router.record(model, ok=False)

        model_used, response = backend.chat_with_fallback(
            chain, messages, tools=tools_schema, on_attempt_failed=_on_fail
        )
        router.record(model_used, ok=True)

        message = response["message"]
        messages.append({"role": "assistant", "content": message.get("content", ""),
                          **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {})})

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return AgentRunResult(
                model_used=model_used,
                content=message.get("content", ""),
                transcript=messages,
                tool_calls_made=tool_calls_made,
            )

        for call in tool_calls:
            result = registry.dispatch(call)
            tool_calls_made += 1
            messages.append(
                {
                    "role": "tool",
                    "content": result.error or str(result.result),
                }
            )

    return AgentRunResult(
        model_used=model_used,
        content="[max tool turns reached without a final answer]",
        transcript=messages,
        tool_calls_made=tool_calls_made,
    )
