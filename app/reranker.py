"""
app/reranker.py

Cross-encoder reranking for improved retrieval relevance.
Uses small multilingual model for efficiency.
"""
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import os

# Use small cross-encoder model (only ~120MB)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        Rerank passages given a query.
        Returns list of scores (higher = more relevant)
        """
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs)
        return scores.tolist()
    
    def rerank_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Rerank pre-formed (query, passage) pairs.
        """
        scores = self.model.predict(pairs)
        return scores.tolist()

# Global reranker instance
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker