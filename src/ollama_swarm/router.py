"""Tier-based routing with a fallback chain and lightweight performance tracking.

GADK's `ModelRouter` (src/services/model_router.py, ~700 lines) classifies free-text
task descriptions into a capability via regex, then complexity via more regex, then
filters/ranks/cost-trades across ~20 models. In an agent framework the caller always
knows what kind of work it's doing — it is the one constructing the prompt — so
guessing that back out of the text is unnecessary indirection, and the regex approach
is brittle (a task description containing "hi" anywhere routes to the cheap/fast
tier regardless of what else it says). Here, the caller states its tier up front;
the router's only job is picking the best *available* model within it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Tier, models_for


@dataclass
class ModelStats:
    successes: int = 0
    failures: int = 0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 1.0  # optimistic prior


@dataclass
class Router:
    _stats: dict[str, ModelStats] = field(default_factory=dict)

    def record(self, model: str, ok: bool) -> None:
        stats = self._stats.setdefault(model, ModelStats())
        if ok:
            stats.successes += 1
        else:
            stats.failures += 1

    def fallback_chain(self, tier: Tier) -> list[str]:
        """Models for a tier, healthy-first: anything with a success rate below
        50% (and at least a few samples) is pushed to the back instead of dropped —
        it may still be the only model that supports some feature the others lack."""
        candidates = models_for(tier)

        def is_unhealthy(model: str) -> bool:
            stats = self._stats.get(model)
            return bool(stats and stats.successes + stats.failures >= 3 and stats.success_rate < 0.5)

        healthy = [m for m in candidates if not is_unhealthy(m)]
        unhealthy = [m for m in candidates if is_unhealthy(m)]
        return healthy + unhealthy
