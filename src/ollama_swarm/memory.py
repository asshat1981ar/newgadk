"""Cross-run memory: one SQLite file, embeddings from Ollama itself.

GADK's memory stack spans `src/memory/memory_graph.py`, `src/memory/graph_store.py`,
`src/services/vector_index.py`, `src/services/embed_quota.py`, `src/services/embedder.py`
(LiteLLM-wrapped), and an entire optional Memori integration
(docs/plans/2026-04-26-ollama-cloud-agents.md, Implementation 3) with its own triple
schema. That is a lot of surface for "remember what happened and find it again later."
Since the embedding model is just another Ollama model, this rolls it into the same
backend used for chat, stores vectors in SQLite as JSON, and does cosine similarity
in Python — correct at the scale a single swarm's memory actually reaches, and with
zero extra dependencies (no sqlite-vec, no separate vector DB).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass

from .backend import OllamaBackend
from .config import Tier, models_for


@dataclass
class MemoryEntry:
    text: str
    tag: str
    score: float


class Memory:
    def __init__(self, backend: OllamaBackend, db_path: str = "swarm_memory.db") -> None:
        self.backend = backend
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self.conn.commit()

    def _embed(self, text: str) -> list[float]:
        model = models_for(Tier.EMBED)[0]
        return self.backend.embed(model, text)

    def remember(self, text: str, tag: str = "note") -> None:
        vec = self._embed(text)
        self.conn.execute(
            "INSERT INTO memory (tag, text, embedding, created_at) VALUES (?, ?, ?, ?)",
            (tag, text, json.dumps(vec), time.time()),
        )
        self.conn.commit()

    def recall(self, query: str, top_k: int = 5, tag: str | None = None) -> list[MemoryEntry]:
        query_vec = self._embed(query)
        sql = "SELECT tag, text, embedding FROM memory"
        params: tuple[str, ...] = ()
        if tag:
            sql += " WHERE tag = ?"
            params = (tag,)

        scored: list[MemoryEntry] = []
        for row_tag, text, embedding_json in self.conn.execute(sql, params):
            vec = json.loads(embedding_json)
            scored.append(MemoryEntry(text=text, tag=row_tag, score=_cosine(query_vec, vec)))

        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
