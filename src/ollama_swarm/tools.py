"""Tool registration and dispatch.

GADK spreads this across `src/tools/dispatcher.py`, `src/capabilities/contracts.py`,
`src/capabilities/registry.py`, and `src/capabilities/service.py` — four files
coordinating a schema, a contract type, a registry, and a dispatch service for
what is, at bottom, "call a Python function with the arguments a model produced."
Ollama's `/api/chat` already speaks OpenAI-style tool schemas natively, so a
registry can build that schema straight from a function's type hints and just
call it — no separate contract layer needed.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        py_type = hints.get(name, str)
        properties[name] = {"type": _PY_TO_JSON_TYPE.get(py_type, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else fn.__name__,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


@dataclass
class ToolCallResult:
    name: str
    result: Any
    error: str | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Use as a decorator: `@registry.register`."""
        self._fns[fn.__name__] = fn
        self._schemas[fn.__name__] = _schema_for(fn)
        return fn

    def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        if names is None:
            return list(self._schemas.values())
        return [self._schemas[n] for n in names if n in self._schemas]

    def dispatch(self, tool_call: Any) -> ToolCallResult:
        """Execute a single tool call object as returned by ollama's `message.tool_calls`."""
        fn_call = tool_call.function if hasattr(tool_call, "function") else tool_call["function"]
        name = fn_call.name if hasattr(fn_call, "name") else fn_call["name"]
        args = fn_call.arguments if hasattr(fn_call, "arguments") else fn_call["arguments"]
        if isinstance(args, str):
            args = json.loads(args)

        fn = self._fns.get(name)
        if fn is None:
            return ToolCallResult(name=name, result=None, error=f"unknown tool: {name}")
        try:
            return ToolCallResult(name=name, result=fn(**args))
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            return ToolCallResult(name=name, result=None, error=str(exc))
