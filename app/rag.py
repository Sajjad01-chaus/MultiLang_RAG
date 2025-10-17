# app/rag.py
from app.retrieve import Retriever
from app.llm_utils import call_llm

def build_context(chunks):
    return "\n\n".join([f"[p{c['payload'].get('page','?')}] {c['payload'].get('text','')}" for c in chunks])

def generate_answer(query: str, retriever: Retriever, top_k: int = 5) -> dict:
    hits = retriever.hybrid_retrieve(query, top_k=top_k)
    if not hits:
        return {"answer": "No relevant context found.", "provenance": []}

    context = build_context(hits)
    prompt = (
        "You are an expert assistant. Use the provided context to answer clearly and concisely.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    answer = call_llm(prompt)
    provenance = [{"doc_id": h['payload'].get('doc_id'), "page": h['payload'].get('page')} for h in hits]
    return {"answer": answer, "provenance": provenance}
