"""
app/evaluation.py

Evaluation metrics for RAG system:
1. Query Relevance (using semantic similarity)
2. Retrieval Accuracy (precision@k, recall@k)
3. Latency tracking
4. Fluency (readability scores)
"""

import time
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
import textstat

class RAGEvaluator:
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.metrics_history = []
        
    def evaluate_query_relevance(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Measure semantic similarity between query and retrieved documents.
        Returns average cosine similarity score.
        """
        if not retrieved_docs:
            return 0.0
            
        query_emb = self.model.encode(query, convert_to_tensor=True)
        doc_embs = self.model.encode(retrieved_docs, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        return float(similarities.mean())
    
    def evaluate_retrieval_accuracy(
        self, 
        retrieved_ids: List[str], 
        ground_truth_ids: List[str], 
        k: int = 5
    ) -> Dict[str, float]:
        """
        Calculate precision@k and recall@k
        """
        retrieved_set = set(retrieved_ids[:k])
        ground_truth_set = set(ground_truth_ids)
        
        if not ground_truth_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        true_positives = len(retrieved_set & ground_truth_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(ground_truth_set)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    
    def evaluate_fluency(self, text: str) -> Dict[str, float]:
        """
        Evaluate fluency using readability metrics
        """
        return {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
            "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
            "automated_readability_index": round(textstat.automated_readability_index(text), 2)
        }
    
    def measure_latency(self, func, *args, **kwargs) -> tuple:
        """
        Measure execution time of a function
        Returns (result, latency_ms)
        """
        start = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start) * 1000
        return result, round(latency, 2)
    
    def log_metrics(self, query: str, response: Dict[str, Any], latency_ms: float):
        """
        Log comprehensive metrics for a query
        """
        retrieved_texts = [p.get("text", "") for p in response.get("provenance", [])]
        
        metrics = {
            "timestamp": time.time(),
            "query": query,
            "latency_ms": latency_ms,
            "query_relevance": self.evaluate_query_relevance(query, retrieved_texts),
            "fluency": self.evaluate_fluency(response.get("answer", "")),
            "num_retrieved": len(retrieved_texts),
            "answer_length": len(response.get("answer", ""))
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from all logged metrics
        """
        if not self.metrics_history:
            return {"message": "No metrics logged yet"}
        
        latencies = [m["latency_ms"] for m in self.metrics_history]
        relevances = [m["query_relevance"] for m in self.metrics_history]
        
        return {
            "total_queries": len(self.metrics_history),
            "avg_latency_ms": round(np.mean(latencies), 2),
            "p50_latency_ms": round(np.percentile(latencies, 50), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "avg_query_relevance": round(np.mean(relevances), 4),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2)
        }

# Global evaluator instance
_evaluator = None

def get_evaluator(model: SentenceTransformer):
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator(model)
    return _evaluator