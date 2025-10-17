import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, tempfile
from app.evaluation import get_evaluator
from app.ingest import ingest_pdf
# from app.retrieve import Retriever # This seems to be your custom class, let's assume it's correctly defined
from app.query_decomposer import QueryDecomposer
from app.llm_utils import call_llm
from app.indexer import ensure_collection
from app.bm25_search import get_bm25_index
from app.reranker import get_reranker
# from sentence_transformers import SentenceTransformer  # ✅ CHANGE: No longer needed here
# from qdrant_client import QdrantClient                # ✅ CHANGE: No longer needed here
from app.retrieve import get_retriever # This should point to your updated retrieve.py
from app.models import QueryRequest, QueryResponse, IngestDocResponse

# ✅ CHANGE: Import the shared instances from app/db.py
from app.db import get_qdrant, get_model, COLLECTION

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Multilingual PDF RAG System", version="1.0")

# --- Configuration is now handled by app/db.py ---
#EMBED_MODEL_NAME = os.getenv(...) # No longer needed
# QDRANT_URL = os.getenv(...)       # No longer needed

# ✅ CHANGE: Initialize models using the shared instances from app/db.py
print("Initializing shared models from app/db...")
embedder = get_model()
qdrant = get_qdrant()
print("Models initialized successfully.")

bm25_index = get_bm25_index()
reranker = get_reranker()

# Ensure collection exists in the in-memory DB
# NOTE: It's important that this runs after qdrant is initialized
try:
    dim = embedder.get_sentence_embedding_dimension()
    ensure_collection(dim)
    print(f"Qdrant collection '{COLLECTION}' ensured with dimension {dim}.")
except Exception as e:
    print(f"Error ensuring collection: {e}")


# ✅ CHANGE: Use the new get_retriever() function from your updated retrieve.py
# This ensures the retriever uses the same in-memory qdrant instance.
retriever = get_retriever()

decomposer = QueryDecomposer(llm_fn=call_llm)


@app.post("/ingest", response_model=IngestDocResponse)
async def ingest_endpoint(file: UploadFile = File(...), language: str = Form(None)):
    """
    Upload and index a PDF file.
    Supports multilingual documents including Hindi, Bengali, Chinese, etc.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Process and index
        result = ingest_pdf(tmp_path, lang_hint=language)
        
        os.remove(tmp_path)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    RAG query endpoint with advanced features:
    - Query decomposition for complex questions
    - Hybrid search (semantic + keyword BM25)
    - Cross-encoder reranking
    - Metadata filtering support
    """
    start = time.time()
    query = payload.query
    metadata_filter = getattr(payload, 'metadata_filter', None)

    # Step 1: Decompose complex query into sub-queries
    subqueries = decomposer.decompose(query)

    # Step 2: Retrieve and answer each sub-query
    answers = []
    all_docs = []
    
    for sq in subqueries:
        # Hybrid retrieval
        docs = retriever.hybrid_retrieve(
            query=sq, 
            top_k=5,
            metadata_filter=metadata_filter
        )
        all_docs.extend(docs) # Collect docs for context

    # Remove duplicates
    unique_docs = {doc['chunk_id']: doc for doc in all_docs}.values()

    # Build context from top documents
    context = "\n\n".join([
        f"[Source: {d.get('source', 'unknown')}, Page: {d.get('page', '?')}]\n{d.get('text', '')}"
        for d in unique_docs
    ])
    
    provenance = [
        {
            "doc_id": d.get("doc_id"),
            "source": d.get("source"),
            "page": d.get("page"),
            "score": d.get("hybrid_score", 0)
        } 
        for d in unique_docs
    ]
    
    # Generate answer using LLM
    prompt = f"""You are a helpful assistant. Answer the question concisely based ONLY on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    llm_answer = call_llm(prompt) # Assuming call_llm is defined elsewhere
    
    final_answer = {
        "answer": llm_answer,
        "provenance": provenance,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "subqueries": subqueries
    }
    
    return final_answer


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "ok": True, 
        "msg": "Multilingual PDF RAG System running",
        "models": {
            "embedding_model_name": embedder.get_model_info().get('name'),
            "collection": COLLECTION
        }
    }


@app.get("/stats")
def get_stats():
    """Get system statistics"""
    try:
        collection_info = qdrant.get_collection(COLLECTION)
        return {
            "ok": True,
            "qdrant_points": collection_info.points_count,
            "bm25_docs": len(bm25_index.corpus),
            "collection": COLLECTION
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# This part for evaluation seems fine as is
evaluator = get_evaluator(embedder)

@app.get("/metrics")
def get_metrics():
    """Get performance metrics and statistics"""
    return evaluator.get_summary_stats()