import os
import uuid
import time
from typing import Dict, Any, List

from app.extract import extract_text_from_pdf
from app.chunker import smart_chunk_text
from app.indexer import ensure_collection, embed_and_upsert
from app.bm25_search import get_bm25_index
from app.db import get_model, get_qdrant

def ingest_pdf(path: str, doc_id: str = None, lang_hint: str = None) -> Dict[str, Any]:
    """End-to-end ingestion with corrected error handling for scanned documents."""
    start = time.time()
    if not doc_id:
        doc_id = str(uuid.uuid4())

    print(f"\n[INGEST] Starting ingestion for: {path}")
    print(f"[INGEST] Using language hint: {lang_hint or 'auto'}")

    # Get shared instances
    model = get_model()
    bm25 = get_bm25_index()

    # 1. Extract text
    try:
        pages = extract_text_from_pdf(path, lang_hint=lang_hint)
        if not pages:
            print("[ERROR] extract_text_from_pdf returned an empty dictionary.")
            return {
                "status": "error", "message": "No text could be extracted from the document.",
                "doc_id": doc_id, "chunks": 0, "pages_processed": 0,
                "seconds": round(time.time() - start, 2)
            }
        print(f"[EXTRACT] Extracted {len(pages)} pages.")

    # ✅ THIS IS THE FIX:
    # We specifically catch the ValueError from extract.py and immediately re-raise it.
    # This allows the error to "pass through" this function and be caught by the API endpoint in app.py.
    except ValueError as e:
        # Re-raise the specific error so the frontend can get the correct message.
        raise e
    except Exception as e:
        # This will now only catch other, unexpected extraction errors.
        print(f"[FATAL INGEST ERROR] {e}")
        return {
            "status": "error", "message": f"An unexpected error occurred during extraction: {str(e)}",
            "doc_id": doc_id, "chunks": 0, "pages_processed": 0,
            "seconds": round(time.time() - start, 2)
        }

    # 2. Chunk text
    all_chunks: List[Dict[str, Any]] = []
    try:
        for page_num, meta in pages.items():
            text = meta.get("text", "")
            if not text or not text.strip():
                continue

            chunks = smart_chunk_text(
                doc_id=doc_id,
                page_num=page_num,
                text=text,
                max_tokens=8000, # Increased for bge-m3 model
                overlap_tokens=100
            )
            if not chunks:
                continue

            for c in chunks:
                c["language"] = meta.get("lang", lang_hint or "und")
                c["source"] = os.path.basename(path)
                c["is_scanned"] = meta.get("is_scanned", False)
                all_chunks.append(c)
        print(f"[CHUNK] Created {len(all_chunks)} chunks total.")
    except Exception as e:
        print(f"[CHUNK ERROR] {e}")
        return {
            "status": "error", "message": f"Chunking failed: {e}",
            "doc_id": doc_id, "chunks": 0, "pages_processed": len(pages),
            "seconds": round(time.time() - start, 2)
        }

    if not all_chunks:
        return {
            "status": "warning", "message": "No valid text chunks were created from the document.",
            "doc_id": doc_id, "chunks": 0, "pages_processed": len(pages),
            "seconds": round(time.time() - start, 2)
        }

    # 3. Index
    try:
        dim = model.get_sentence_embedding_dimension()
        ensure_collection(dim)

        chunk_texts = [c["text"] for c in all_chunks]
        metadata_list = [
            {
                "id": c["chunk_id"], "doc_id": c["doc_id"], "page": c["page"],
                "language": c["language"], "source": c["source"],
                "is_scanned": c["is_scanned"], "text": c["text"]
            }
            for c in all_chunks
        ]

        embed_and_upsert(chunk_texts, metadata_list)
        bm25.add_documents(chunk_texts, metadata_list)
        print(f"[INDEX] Indexed {len(all_chunks)} chunks in Qdrant + BM25.")
    except Exception as e:
        print(f"[INDEX ERROR] {e}")
        return {
            "status": "error", "message": f"Indexing failed: {e}",
            "doc_id": doc_id, "chunks": 0, "pages_processed": len(pages),
            "seconds": round(time.time() - start, 2)
        }

    elapsed = time.time() - start
    print(f"[DONE] Completed ingestion in {elapsed:.2f}s ✅")

    return {
        "status": "success", "doc_id": doc_id, "chunks": len(all_chunks),
        "pages_processed": len(pages), "seconds": round(elapsed, 2),
        "message": f"Indexed {len(all_chunks)} chunks from {len(pages)} pages."
    }

