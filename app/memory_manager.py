"""
app/memory_manager.py

Short-term Redis-backed chat memory manager + optional long-term memory stored as embeddings
in Qdrant.

Usage:
    mm = MemoryManager(redis_url="redis://localhost:6379", qdrant_client=qdrant, embed_fn=embed_fn)
    mm.push_turn(session_id, user_text="Hi", assistant_text="Hello")
    recent = mm.get_recent(session_id, n=8)
    mm.maybe_summarize(session_id, summarizer_fn=call_llm)  # optional periodic summarization
    mm.upsert_long_term(memory_text, metadata)
    mm.query_long_term(query, top_k=5)
"""

import json
import time
import hashlib
from typing import List, Dict, Optional, Callable

import redis
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# default collection to store long-term memory vectors
LT_MEMORY_COLLECTION = "long_term_memory"

class MemoryManager:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        qdrant_client: Optional[QdrantClient] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        short_term_turns: int = 10,
        long_term_summary_interval: int = 20,  # number of turns before summarization
    ):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.qdrant = qdrant_client
        self.embed_fn = embed_fn
        self.short_term_turns = short_term_turns
        self.long_term_summary_interval = long_term_summary_interval

    def _redis_key(self, session_id: str) -> str:
        return f"session_memory:{session_id}"

    def push_turn(self, session_id: str, user_text: str, assistant_text: str, timestamp: Optional[float] = None):
        """Push a new turn (user + assistant). Keep only last short_term_turns."""
        if timestamp is None:
            timestamp = time.time()
        obj = {"user": user_text, "assistant": assistant_text, "ts": timestamp}
        key = self._redis_key(session_id)
        self.redis.rpush(key, json.dumps(obj))
        # trim
        self.redis.ltrim(key, -self.short_term_turns, -1)

    def get_recent(self, session_id: str, n: int = 5) -> List[Dict]:
        key = self._redis_key(session_id)
        raw = self.redis.lrange(key, -n, -1)
        return [json.loads(x) for x in raw]

    def clear_session(self, session_id: str):
        self.redis.delete(self._redis_key(session_id))

    # ---- Long-term memory (optional) ----
    def ensure_lt_collection(self, dim: int):
        """Create Qdrant collection if missing (idempotent)."""
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not provided")
        from qdrant_client.http.models import VectorParams
        # recreate_collection is used in earlier code; here we try create if not exists
        try:
            self.qdrant.get_collection(LT_MEMORY_COLLECTION)
        except Exception:
            self.qdrant.recreate_collection(
                collection_name=LT_MEMORY_COLLECTION,
                vectors_config=VectorParams(size=dim, distance="Cosine")
            )

    def upsert_long_term(self, texts: List[str], metadatas: List[Dict]):
        """Embed texts with embed_fn and upsert to Qdrant long-term memory collection."""
        if self.qdrant is None or self.embed_fn is None:
            raise RuntimeError("Qdrant client and embed_fn required for upsert_long_term")
        embeddings = self.embed_fn(texts)
        points = []
        for i, (vec, meta, txt) in enumerate(zip(embeddings, metadatas, texts)):
            uid = meta.get("id") or hashlib.sha1(f"{txt}-{time.time()}-{i}".encode()).hexdigest()
            payload = meta.copy()
            payload["text"] = txt
            points.append({"id": uid, "vector": vec, "payload": payload})
        self.qdrant.upsert(collection_name=LT_MEMORY_COLLECTION, points=points)

    def query_long_term(self, query: str, top_k: int = 5, embed_query_fn: Optional[Callable] = None):
        """Return top-k long-term memory hits for `query`."""
        if self.qdrant is None:
            raise RuntimeError("Qdrant client required")
        embed_query_fn = embed_query_fn or self.embed_fn
        if embed_query_fn is None:
            raise RuntimeError("embed_query_fn required")
        qvec = embed_query_fn([query])[0]
        hits = self.qdrant.search(collection_name=LT_MEMORY_COLLECTION, query_vector=qvec, limit=top_k)
        return [{"id": h.id, "payload": h.payload, "score": h.score} for h in hits]

    # ---- Summarization hook ----
    def maybe_summarize(self, session_id: str, summarizer_fn: Callable[[str], str], force: bool = False):
        """
        If session length exceeds long_term_summary_interval, summarize last N turns and upsert to long-term memory.
        summarizer_fn: a function that takes text and returns summary string (e.g. call_llm)
        """
        key = self._redis_key(session_id)
        length = self.redis.llen(key)
        if not force and length < self.long_term_summary_interval:
            return None
        turns = self.get_recent(session_id, n=length)
        combined = "\n".join([f"User: {t['user']}\nAssistant: {t['assistant']}" for t in turns])
        summary = summarizer_fn(combined)
        # upsert summary into long-term memory with metadata
        meta = {"session_id": session_id, "summary_ts": time.time(), "type": "session_summary"}
        if self.embed_fn and self.qdrant:
            self.upsert_long_term([summary], [meta])
        # optional: clear short-term memory after summarization to reduce token costs
        self.clear_session(session_id)
        return summary
