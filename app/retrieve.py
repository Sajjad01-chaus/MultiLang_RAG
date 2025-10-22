import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client.http import models as qmodels
from app.db import get_qdrant, get_model, COLLECTION
from app.bm25_search import get_bm25_index
from app.reranker import get_reranker

class HybridRetriever:
    def __init__(self):
        self.qdrant = get_qdrant()
        self.model = get_model()
        self.bm25 = get_bm25_index()
        self.collection = COLLECTION
        self.reranker = get_reranker()

    def semantic_search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        qvec = self.model.encode(query, convert_to_numpy=True).astype(np.float32).tolist()
        qfilter = None
        if metadata_filter:
            qfilter = qmodels.Filter(
                must=[qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in metadata_filter.items()]
            )

        hits = self.qdrant.search(collection_name=self.collection, query_vector=qvec, limit=top_k, query_filter=qfilter)
        return [
            {
                "chunk_id": h.payload.get("id"),
                "doc_id": h.payload.get("doc_id"),
                "page": h.payload.get("page"),
                "text": h.payload.get("text"),
                "score": h.score,
                "language": h.payload.get("language"),
                "source": h.payload.get("source"),
            } for h in hits
        ]

    def keyword_search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None):
        return self.bm25.search(query, top_k=top_k)

    def hybrid_retrieve(self, query: str, top_k: int = 5, semantic_weight: float = 0.7, metadata_filter: Optional[Dict] = None):
        sem = self.semantic_search(query, top_k=top_k * 2, metadata_filter=metadata_filter)
        kw = self.keyword_search(query, top_k=top_k * 2, metadata_filter=metadata_filter)
        combined = {}

        for r in sem:
            cid = r.get("chunk_id")
            if cid:
                combined[cid] = {**r, "hybrid_score": r.get("score", 0) * semantic_weight}

        for r in kw:
            cid = r.get("id")
            if not cid:
                continue
            if cid in combined:
                combined[cid]["hybrid_score"] += r.get("score", 0) * (1 - semantic_weight)
            else:
                combined[cid] = {
                    "chunk_id": cid,
                    "doc_id": r.get("doc_id"),
                    "page": r.get("page"),
                    "text": r.get("text"),
                    "score": r.get("score"),
                    "language": r.get("language"),
                    "source": r.get("source"),
                    "hybrid_score": r.get("score", 0) * (1 - semantic_weight),
                }

        docs = list(combined.values())
        passages = [d["text"] for d in docs]
        scores = self.reranker.rerank(query, passages)
        for d, s in zip(docs, scores):
            d["hybrid_score"] = (d["hybrid_score"] + s) / 2.0

        return sorted(docs, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]


_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
