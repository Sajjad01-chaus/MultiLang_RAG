"""
app/bm25_search.py

Implements BM25 keyword search for hybrid retrieval.
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import numpy as np

class BM25Index:
    def __init__(self):
        self.corpus = []  # List of tokenized documents
        self.doc_metadata = []  # Metadata for each document
        self.bm25 = None
        
    def index_documents(self, texts: List[str], metadatas: List[Dict]):
        """Index documents for BM25 search"""
        self.corpus = [text.lower().split() for text in texts]
        self.doc_metadata = metadatas
        self.bm25 = BM25Okapi(self.corpus)
        
    def search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """Search using BM25 and return top results"""
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # Get more for filtering
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
                
            meta = self.doc_metadata[idx]
            
            # Apply metadata filter if provided
            if metadata_filter:
                match = all(meta.get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue
                    
            results.append({
                "id": meta.get("id"),
                "text": meta.get("text", ""),
                "payload": meta,
                "score": float(scores[idx])
            })
            
        return results
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Incrementally add documents to index"""
        new_corpus = [text.lower().split() for text in texts]
        self.corpus.extend(new_corpus)
        self.doc_metadata.extend(metadatas)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.corpus)

# Global BM25 index instance
_bm25_index = BM25Index()

def get_bm25_index():
    return _bm25_index