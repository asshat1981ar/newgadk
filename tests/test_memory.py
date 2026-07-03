from __future__ import annotations

from ollama_swarm.memory import Memory


def test_recall_ranks_closer_text_first(tmp_path, fake_backend) -> None:
    backend = fake_backend(lambda messages: {"message": {"content": "unused"}})
    memory = Memory(backend, db_path=str(tmp_path / "mem.db"))

    memory.remember("the deployment pipeline failed on the staging build", tag="incident")
    memory.remember("bananas are a good source of potassium", tag="trivia")

    results = memory.recall("staging deployment pipeline issue", top_k=1)

    assert len(results) == 1
    assert "deployment" in results[0].text


def test_recall_can_filter_by_tag(tmp_path, fake_backend) -> None:
    backend = fake_backend(lambda messages: {"message": {"content": "unused"}})
    memory = Memory(backend, db_path=str(tmp_path / "mem.db"))

    memory.remember("goal alpha succeeded", tag="run_summary")
    memory.remember("unrelated note", tag="note")

    results = memory.recall("alpha", tag="run_summary")

    assert all(r.tag == "run_summary" for r in results)
