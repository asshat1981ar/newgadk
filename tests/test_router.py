from __future__ import annotations

from ollama_swarm.config import Tier, models_for
from ollama_swarm.router import Router


def test_fallback_chain_defaults_to_catalog_order() -> None:
    router = Router()
    assert router.fallback_chain(Tier.CODING) == models_for(Tier.CODING)


def test_unhealthy_model_is_pushed_to_the_back_not_dropped() -> None:
    router = Router()
    chain = models_for(Tier.CODING)
    flaky = chain[0]

    for _ in range(2):
        router.record(flaky, ok=True)
    for _ in range(4):
        router.record(flaky, ok=False)

    result = router.fallback_chain(Tier.CODING)

    assert flaky in result
    assert result[-1] == flaky
    assert result[0] != flaky


def test_healthy_model_keeps_its_position() -> None:
    router = Router()
    chain = models_for(Tier.CODING)
    router.record(chain[0], ok=True)
    router.record(chain[0], ok=True)

    assert router.fallback_chain(Tier.CODING) == chain
