"""Single source of truth for models, tiers, and connection settings.

GADK kept two independent copies of its model-capability map (`Config.MODEL_CAPABILITY_MAP`
in config.py and `ModelRegistry.DEFAULT_CAPABILITIES` in model_router.py) that could drift.
Here there is exactly one table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    """Capability tiers. Callers pick a tier explicitly instead of the framework
    guessing task type from regex over natural language (GADK's ModelRouter did this
    and would misfile "hi, please refactor this distributed cache" as QUICK because
    it matched the greeting pattern first)."""

    REASONING = "reasoning"   # architecture, review, judgment calls
    CODING = "coding"         # implementation, refactors
    FAST = "fast"             # classification, routing, short answers
    EMBED = "embed"           # embeddings for memory/retrieval


# name -> (tier, ...fallback tiers if the primary model is unavailable)
# Probed live 2026-07 against every model ollama.com/search?c=cloud lists: gpt-oss,
# qwen3-coder (needs a variant tag, unlike qwen3-coder-next), and devstral-2 (previously
# in this catalog's CODING fallback chain) all 404. ministral-3/nemotron-3-nano/
# gemini-3-flash also 404'd in an earlier probe. Everything below resolved.
MODEL_CATALOG: dict[Tier, list[str]] = {
    Tier.REASONING: [
        "glm-5:cloud", "glm-5.2:cloud", "glm-5.1:cloud",
        "deepseek-v4-flash:cloud", "deepseek-v4-pro:cloud",
        "kimi-k2.6:cloud", "nemotron-3-ultra:cloud", "nemotron-3-super:cloud",
    ],
    Tier.CODING: [
        "qwen3-coder-next:cloud", "kimi-k2.7-code:cloud",
        "kimi-k2.6:cloud", "deepseek-v4-pro:cloud",
    ],
    Tier.FAST: [
        "qwen3.5:cloud", "gemma4:cloud", "glm-4.7:cloud",
        "gemini-3-flash-preview:cloud", "kimi-k2.5:cloud",
    ],
    Tier.EMBED: ["nomic-embed-text"],
}


@dataclass
class Settings:
    """Env-driven settings. No pydantic dependency — this project's only
    hard dependency is the `ollama` package itself."""

    # "daemon": talk to a local `ollama serve` process, which transparently
    #           proxies `:cloud` models once `ollama signin` has run.
    # "direct-cloud": talk to https://ollama.com directly with a bearer token —
    #           for daemon-less deploys (containers/serverless) with no local ollama.
    mode: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODE", "daemon"))
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    cloud_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("OLLAMA_API_KEY"))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("OLLAMA_TIMEOUT_S", "60")))
    max_tool_turns: int = field(default_factory=lambda: int(os.environ.get("OLLAMA_MAX_TOOL_TURNS", "6")))
    workspace_root: str = field(default_factory=lambda: os.environ.get("OLLAMA_SWARM_WORKSPACE", "./workspace"))

    def active_host(self) -> str:
        return self.cloud_host if self.mode == "direct-cloud" else self.host


SETTINGS = Settings()


def models_for(tier: Tier) -> list[str]:
    """Ordered fallback chain for a tier. First entry is preferred."""
    return list(MODEL_CATALOG[tier])
