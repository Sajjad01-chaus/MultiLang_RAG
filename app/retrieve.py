# """
# app/retrieve.py

# Hybrid retrieval with metadata filter support and cross-encoder reranking.
# This file focuses on clarity and integration points so you can plug your models/clients easily.
# """

# from typing import List, Dict, Any, Optional, Tuple
# import numpy as np
# import time

# # placeholder imports - replace with your actual model instances or pass in functions
# # from sentence_transformers import SentenceTransformer
# # from cross_encoder import CrossEncoder
# # from qdrant_client import QdrantClient

# # Example signature expectations:
# # semantic_embed_fn(texts: List[str]) -> List[List[float]]
# # cross_rerank_fn(pairs: List[Tuple[str, str]]) -> List[float]
# # bm25_search_fn(query: str, top_k: int) -> List[Dict]  # returns [{'id': id, 'text': text}]
# # qdrant_client.search(collection_name, query_vector, limit, query_filter=...) -> hits

# def normalize_scores(scores: List[float]) -> np.ndarray:
#     arr = np.array(scores, dtype=float)
#     if arr.max() == arr.min():
#         return np.ones_like(arr)
#     # scale 0-1
#     return (arr - arr.min()) / (arr.max() - arr.min())

# class Retriever:
#     def __init__(
#         self,
#         qdrant_client,
#         semantic_embed_fn,
#         bm25_search_fn=None,
#         cross_rerank_fn=None,
#         collection: str = "pdf_chunks",
#     ):
#         self.qdrant = qdrant_client
#         self.semantic_embed_fn = semantic_embed_fn
#         self.bm25_search_fn = bm25_search_fn
#         self.cross_rerank_fn = cross_rerank_fn
#         self.collection = collection

#     def _qdrant_filter_from_meta(self, metadata_filter: Optional[Dict[str, Any]]):
#         """
#         Convert a simple metadata_filter dict into Qdrant Filter model.
#         Example metadata_filter:
#             {"language": "hi", "source": "policy", "date_after": "2020-01-01"}
#         This function should be adapted to your payload schema.
#         """
#         if not metadata_filter:
#             return None
#         # simple equality matches
#         must_conditions = []
#         for k, v in metadata_filter.items():
#             if k.endswith("_after") or k.endswith("_before"):
#                 # date handling not fully implemented here — adapt based on how you store dates in payload
#                 continue
#             must_conditions.append(
#                 {"key": k, "match": {"value": v}}
#             )
#         # Qdrant client accepts a Filter in typed models; easiest is to use dict if client supports it,
#         # else construct FieldCondition objects. We'll return dict-friendly structure.
#         if must_conditions:
#             return {"must": [{"key": c["key"], "match": {"value": c["match"]["value"]}} for c in must_conditions]}
#         return None

#     def semantic_search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None):
#         qvec = self.semantic_embed_fn([query])[0]
#         qfilter = self._qdrant_filter_from_meta(metadata_filter)
#         # Qdrant python client: qdrant.search(collection_name, query_vector=..., limit=top_k, query_filter=qfilter)
#         hits = self.qdrant.search(collection_name=self.collection, query_vector=qvec, limit=top_k, query_filter=qfilter)
#         # hits: objects with .id, .payload, .score
#         results = [{"id": h.id, "payload": h.payload, "score": h.score} for h in hits]
#         return results

#     def keyword_search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None):
#         if self.bm25_search_fn is None:
#             return []
#         # bm25_search_fn should return list of dicts with 'id', 'text', optionally 'score'
#         return self.bm25_search_fn(query, top_k=top_k, metadata_filter=metadata_filter)

#     def hybrid_retrieve(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None, rerank_top_k: int = 20):
#         """
#         1. Run semantic search (Qdrant)
#         2. Run keyword search (BM25) if available
#         3. Merge candidates (dedupe by id)
#         4. Rerank using cross encoder if available
#         """
#         sem = self.semantic_search(query, top_k=top_k, metadata_filter=metadata_filter)
#         kw = self.keyword_search(query, top_k=top_k, metadata_filter=metadata_filter)

#         candidates = {}
#         # add semantic candidates
#         for s in sem:
#             candidates[s["id"]] = {"id": s["id"], "payload": s["payload"], "sem_score": s["score"], "text": s["payload"].get("text","")}
#         # add keyword candidates
#         for k in kw:
#             cid = k.get("id")
#             if cid not in candidates:
#                 candidates[cid] = {"id": cid, "payload": k.get("payload", {}), "kw_score": k.get("score", 0.0), "text": k.get("text", k.get("payload",{}).get("text",""))}
#             else:
#                 candidates[cid]["kw_score"] = k.get("score", 0.0)

#         # build list for reranking
#         cand_list = list(candidates.values())
#         # if cross-ranker exists, create pairs: (query, passage_text)
#         if self.cross_rerank_fn and cand_list:
#             # limit to rerank_top_k most promising candidates by combined heuristic
#             # compute combined initial score
#             for c in cand_list:
#                 sem_s = c.get("sem_score", 0.0) or 0.0
#                 kw_s = c.get("kw_score", 0.0) or 0.0
#                 c["initial_score"] = sem_s + kw_s
#             cand_list = sorted(cand_list, key=lambda x: x["initial_score"], reverse=True)[:rerank_top_k]

#             pair_list = [(query, c.get("text","")) for c in cand_list]
#             rerank_scores = self.cross_rerank_fn(pair_list)  # returns list of floats (higher = better)
#             for c, score in zip(cand_list, rerank_scores):
#                 c["rerank_score"] = float(score)
#             # final sort uses rerank_score primarily, fallback to initial_score
#             ranked = sorted(cand_list, key=lambda x: (x.get("rerank_score", 0.0), x.get("initial_score", 0.0)), reverse=True)
#         else:
#             # fallback: sort by sem_score + kw_score
#             for c in cand_list:
#                 c["initial_score"] = c.get("sem_score", 0.0) + c.get("kw_score", 0.0)
#             ranked = sorted(cand_list, key=lambda x: x.get("initial_score", 0.0), reverse=True)

#         # return top_k
#         return ranked[:top_k]


# app/retrieve.py
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client.http import models as qmodels

# Import the shared instances from your db file
from app.db import get_qdrant, get_model, COLLECTION
from app.bm25_search import get_bm25_index

class HybridRetriever:
    def __init__(self):
        """
        Initializes the retriever using shared instances from app.db
        """
        self.qdrant = get_qdrant()
        self.model = get_model()
        self.bm25 = get_bm25_index()
        self.collection = COLLECTION

    def semantic_search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Performs semantic search using vector embeddings."""
        qvec = self.model.encode(query, convert_to_numpy=True).astype(np.float32).tolist()

        qfilter = None
        if metadata_filter:
            qfilter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                    for k, v in metadata_filter.items()
                ]
            )

        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=top_k,
            query_filter=qfilter
        )
        
        # Format results to always include 'chunk_id'
        results = [
            {
                "chunk_id": hit.payload.get("id"),
                "doc_id": hit.payload.get("doc_id"),
                "page": hit.payload.get("page"),
                "text": hit.payload.get("text"),
                "score": hit.score,
                "language": hit.payload.get("language"),
                "source": hit.payload.get("source"),
            }
            for hit in hits
        ]
        return results

    def keyword_search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Performs BM25 keyword search."""
        return self.bm25.search(query, top_k=top_k, metadata_filter=metadata_filter)

    def hybrid_retrieve(self, query: str, top_k: int = 5, semantic_weight: float = 0.7, metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Combines semantic and keyword search results with consistent data structure."""
        sem = self.semantic_search(query, top_k=top_k * 2, metadata_filter=metadata_filter)
        kw = self.keyword_search(query, top_k=top_k * 2, metadata_filter=metadata_filter)
        
        combined = {}
        
        # Add semantic results
        for result in sem:
            chunk_id = result.get("chunk_id")
            if not chunk_id: continue
            
            combined[chunk_id] = {
                **result,
                "hybrid_score": result.get("score", 0.0) * semantic_weight
            }
        
        # Add keyword results
        keyword_weight = 1.0 - semantic_weight
        for result in kw:
            chunk_id = result.get("id") # Keyword search uses 'id'
            if not chunk_id: continue
            
            keyword_score = result.get("score", 0.0)
            
            if chunk_id in combined:
                combined[chunk_id]["hybrid_score"] += keyword_score * keyword_weight
            else:
                # ✅ THIS IS THE FIX: Ensure the dictionary has the 'chunk_id' key
                combined[chunk_id] = {
                    "chunk_id": chunk_id, # Explicitly add the required key
                    "doc_id": result.get("doc_id"),
                    "page": result.get("page"),
                    "text": result.get("text"),
                    "score": keyword_score,
                    "language": result.get("language"),
                    "source": result.get("source"),
                    "hybrid_score": keyword_score * keyword_weight
                }
        
        ranked = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_k]


# Singleton instance to avoid re-initializing
_retriever = None

def get_retriever() -> HybridRetriever:
    """Gets the singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever