import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, tempfile
from dotenv import load_dotenv
from langdetect import detect

# Load environment variables FIRST
load_dotenv()

from app.evaluation import get_evaluator
from app.ingest import ingest_pdf
from app.query_decomposer import QueryDecomposer
from app.llm_utils import call_llm
from app.indexer import ensure_collection
from app.bm25_search import get_bm25_index
from app.reranker import get_reranker
from app.retrieve import get_retriever
from app.models import QueryRequest, QueryResponse, IngestDocResponse
from app.db import get_qdrant, get_model, COLLECTION

app = FastAPI(title="Multilingual PDF RAG System", version="1.1")

print("Initializing shared models...")
embedder = get_model()
qdrant = get_qdrant()
bm25_index = get_bm25_index()
reranker = get_reranker()
retriever = get_retriever()
decomposer = QueryDecomposer(llm_fn=call_llm)
evaluator = get_evaluator(embedder)
print("✅ Models initialized successfully.")

try:
    dim = embedder.get_sentence_embedding_dimension()
    ensure_collection(dim)
    print(f"Qdrant collection '{COLLECTION}' ready (dim={dim})")
except Exception as e:
    print(f"⚠️ Error ensuring collection: {e}")


@app.post("/ingest", response_model=IngestDocResponse)
async def ingest_endpoint(file: UploadFile = File(...), language: str = Form(None)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ✅ Use filename-based doc_id so evaluation can match it deterministically
        custom_doc_id = os.path.splitext(file.filename)[0]
        result = ingest_pdf(tmp_path, doc_id=custom_doc_id, lang_hint=language)
        os.remove(tmp_path)
        return result

    except ValueError as e:
        print(f"[USER ERROR] {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e), "doc_id": "N/A", "chunks": 0}
        )
    except Exception as e:
        print(f"[UNEXPECTED INGEST ERROR] {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected error: {str(e)}", "doc_id": "N/A", "chunks": 0}
        )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    RAG query endpoint with multilingual-aware retrieval and provenance tracking.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    start = time.time()
    query = payload.query
    metadata_filter = getattr(payload, "metadata_filter", None)

    # --- Step 1: Detect query language ---
    try:
        detected_lang = detect(query)
        print(f"[LANG DETECT] Query language detected: {detected_lang}")
    except Exception:
        detected_lang = None

    # --- Step 2: Add language filter ---
    q_filter = None
    if detected_lang:
        q_filter = Filter(
            must=[
                FieldCondition(
                    key="language",
                    match=MatchValue(value=detected_lang)
                )
            ]
        )

    # --- Step 3: Encode query with multilingual E5 (normalized) ---
    query_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    # --- Step 4: Retrieve from Qdrant (semantic) ---
    try:
        qdrant_hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_emb,
            limit=8,
            with_payload=True,
            query_filter=q_filter
        )
    except Exception as e:
        print(f"[QDRANT SEARCH ERROR] {e}")
        qdrant_hits = []

    # --- Step 5: BM25 fallback retrieval ---
    # ✅ FIXED: removed unsupported language_filter argument
    bm25_docs = bm25_index.search(query, top_k=8)

    # --- Step 6: Merge and deduplicate ---
    all_docs = []
    for h in qdrant_hits:
        p = h.payload
        p["score"] = h.score
        p["source"] = "qdrant"
        all_docs.append(p)
    for b in bm25_docs:
        b["source"] = "bm25"
        all_docs.append(b)

    unique_docs = list({d.get("chunk_id", id(d)): d for d in all_docs}.values())
    unique_docs = sorted(unique_docs, key=lambda x: x.get("score", 0), reverse=True)[:5]

   # --- Step 7: Context + Provenance ---
    context = "\n\n".join([f"[Doc: {d.get('doc_id') or d.get('id') or d.get('source')}] {d.get('text')}" for d in unique_docs])

    provenance = []
    for d in unique_docs:
        prov = {
            "doc_id": d.get("doc_id") or d.get("id") or d.get("source") or "unknown",
            "chunk_id": d.get("chunk_id"),
            "source": d.get("source"),
            "page": d.get("page"),
            "score": d.get("score"),
            "language": d.get("language")
        }
        provenance.append(prov)


    if not context.strip():
        context = "(No relevant context found.)"

#---step 8--------
    prompt = f"""
You are a multilingual assistant that can understand and translate Bengali, Urdu, and Chinese.
Answer based ONLY on the context below, even if it's in another language.
If the answer is not present, respond with "Not found in document."

---
Question Language: {detected_lang or 'unknown'}
Query: {query}

Context:
{context}

Answer (in the same language as the query):
"""

    llm_answer = call_llm(prompt)
    latency = round((time.time() - start) * 1000, 2)

    return {
        "answer": llm_answer,
        "provenance": provenance,
        "latency_ms": latency,
        "detected_language": detected_lang
    }




@app.get("/health")
def health_check():
    return {"ok": True, "msg": "Multilingual PDF RAG System is running"}

@app.get("/stats")
def get_stats():
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

@app.get("/metrics")
def get_metrics():
    return evaluator.get_summary_stats()
