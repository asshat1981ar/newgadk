"""A scripted fake backend so the whole suite runs with zero network calls."""

from __future__ import annotations

from typing import Any, Callable, Iterator

import pytest


class FakeBackend:
    """Duck-types OllamaBackend. `script` is a callable that inspects the
    outgoing messages and returns the next response dict to hand back."""

    def __init__(self, script: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> None:
        self.script = script
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(messages)
        return self.script(messages)

    def chat_with_fallback(
        self,
        models: list[str],
        messages: list[dict[str, Any]],
        on_attempt_failed: Callable[[str, Exception], None] | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        return models[0], self.chat(models[0], messages, **kwargs)

    def chat_stream(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        yield self.chat(model, messages)["message"]["content"]

    def embed(self, model: str, text: str) -> list[float]:
        # Deterministic "embedding": character frequency histogram over a-z, so
        # near-duplicate text ends up with near-identical vectors for cosine tests.
        vec = [0.0] * 26
        for ch in text.lower():
            idx = ord(ch) - ord("a")
            if 0 <= idx < 26:
                vec[idx] += 1.0
        return vec


@pytest.fixture
def fake_backend() -> Callable[[Callable[[list[dict[str, Any]]], dict[str, Any]]], FakeBackend]:
    return FakeBackend
