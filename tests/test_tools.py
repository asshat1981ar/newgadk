from __future__ import annotations

from types import SimpleNamespace

from ollama_swarm.tools import ToolRegistry


def sample_tool(city: str, count: int = 1) -> str:
    """Return a greeting for a city."""
    return f"{city}" * count


def test_schema_marks_only_required_params_without_defaults() -> None:
    registry = ToolRegistry()
    registry.register(sample_tool)
    schema = registry.schemas(["sample_tool"])[0]

    assert schema["function"]["name"] == "sample_tool"
    assert schema["function"]["parameters"]["required"] == ["city"]
    assert schema["function"]["parameters"]["properties"]["count"]["type"] == "integer"


def test_dispatch_calls_registered_function_with_dict_args() -> None:
    registry = ToolRegistry()
    registry.register(sample_tool)

    call = SimpleNamespace(function=SimpleNamespace(name="sample_tool", arguments={"city": "NYC", "count": 2}))
    result = registry.dispatch(call)

    assert result.error is None
    assert result.result == "NYCNYC"


def test_dispatch_unknown_tool_reports_error_instead_of_raising() -> None:
    registry = ToolRegistry()
    call = SimpleNamespace(function=SimpleNamespace(name="nope", arguments={}))

    result = registry.dispatch(call)

    assert result.error == "unknown tool: nope"


def test_dispatch_catches_exceptions_from_the_tool_itself() -> None:
    registry = ToolRegistry()

    @registry.register
    def boom() -> str:
        raise ValueError("kaboom")

    call = SimpleNamespace(function=SimpleNamespace(name="boom", arguments={}))
    result = registry.dispatch(call)

    assert result.error == "kaboom"
