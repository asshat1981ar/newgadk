"""One backend class for every Ollama-shaped call: local daemon or direct cloud.

GADK had `OllamaCloudBackend` (src/services/ollama_cloud_backend.py, cloud-only,
Bearer-auth against https://ollama.com) built alongside a `ModelRouter.get_backend()`
that assumed the same shape but only for `ollama/`-prefixed model strings, plus a
still-unfinished `docs/plans/...-ollama-cloud-agents.md` sketching a *third*,
near-identical `OllamaBackend`. Three overlapping implementations of the same idea.

Here there is one class. Its `mode` decides whether requests go to a local daemon
(which itself transparently proxies `:cloud` models once `ollama signin` has run)
or straight to Ollama's cloud endpoint with a bearer token. Callers never branch
on that — they just call `.chat()`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from .config import SETTINGS, Settings


class OllamaBackend:
    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from ollama import Client

            kwargs: dict[str, Any] = {
                "host": self.settings.active_host(),
                "timeout": self.settings.timeout_s,
            }
            if self.settings.mode == "direct-cloud" and self.settings.api_key:
                kwargs["headers"] = {"Authorization": f"Bearer {self.settings.api_key}"}
            self._client = Client(**kwargs)
        return self._client

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        format: dict[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """One non-streaming call. Returns the raw ollama message dict (has
        `.message.content` and, when tools were offered, `.message.tool_calls`)."""
        opts: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            opts["tools"] = tools
        if format:
            opts["format"] = format
        opts.update(kwargs)
        return self.client.chat(**opts)  # type: ignore[no-any-return]

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield content text chunks as they arrive."""
        stream = self.client.chat(model=model, messages=messages, stream=True, **kwargs)
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece

    def embed(self, model: str, text: str) -> list[float]:
        result = self.client.embeddings(model=model, prompt=text)
        return result["embedding"]  # type: ignore[no-any-return]

    def chat_with_fallback(
        self,
        models: list[str],
        messages: list[dict[str, Any]],
        on_attempt_failed: Callable[[str, Exception], None] | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Try each model in order, return (model_used, response) from the first
        that succeeds. Raises the last error if all fail."""
        last_exc: Exception | None = None
        for model in models:
            try:
                return model, self.chat(model, messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall through
                last_exc = exc
                if on_attempt_failed:
                    on_attempt_failed(model, exc)
        assert last_exc is not None
        raise last_exc
