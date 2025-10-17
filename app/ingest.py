# app/ingest.py
import os
import uuid
import time
from typing import Dict, Any, List

from app.extract import extract_text_from_pdf
from app.chunker import smart_chunk_text
from app.indexer import ensure_collection, embed_and_upsert
from app.bm25_search import get_bm25_index
from app.db import get_model, get_qdrant  # ✅ Import shared instances


def ingest_pdf(path: str, doc_id: str = None, lang_hint: str = None) -> Dict[str, Any]:
    """End-to-end ingestion with debugging output."""
    start = time.time()
    if not doc_id:
        doc_id = str(uuid.uuid4())

    print(f"\n[INGEST] Starting ingestion for: {path}")
    print(f"[INGEST] Using language hint: {lang_hint or 'auto'}")

    # Get shared instances
    model = get_model()
    qdrant = get_qdrant()
    bm25 = get_bm25_index()

    # 1️⃣ Extract text
    try:
        pages = extract_text_from_pdf(path, lang_hint=lang_hint)
        if not pages:
            print("[ERROR] extract_text_from_pdf returned None or empty dict.")
            return {
                "status": "error",
                "message": "No text/pages found during extraction.",
                "doc_id": doc_id,
                "chunks": 0,
                "pages_processed": 0,
                "seconds": round(time.time() - start, 2)
            }
        if not isinstance(pages, dict):
            print(f"[ERROR] extract_text_from_pdf returned type: {type(pages)}")
            return {
                "status": "error", 
                "message": "Invalid extraction output type.",
                "doc_id": doc_id,
                "chunks": 0,
                "pages_processed": 0,
                "seconds": round(time.time() - start, 2)
            }
        print(f"[EXTRACT] Extracted {len(pages)} pages.")
    except Exception as e:
        print(f"[EXTRACT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"Extraction failed: {str(e)}",
            "doc_id": doc_id,
            "chunks": 0,
            "pages_processed": 0,
            "seconds": round(time.time() - start, 2)
        }

    # 2️⃣ Chunk text
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
                max_tokens=400,
                overlap_tokens=50
            )
            if chunks is None:
                print(f"[ERROR] smart_chunk_text returned None for page {page_num}")
                continue

            for c in chunks:
                c["language"] = meta.get("lang", lang_hint or "und")
                c["source"] = os.path.basename(path)
                c["is_scanned"] = meta.get("is_scanned", False)
                all_chunks.append(c)
        print(f"[CHUNK] Created {len(all_chunks)} chunks total.")
    except Exception as e:
        print(f"[CHUNK ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"Chunking failed: {str(e)}",
            "doc_id": doc_id,
            "chunks": 0,
            "pages_processed": len(pages),
            "seconds": round(time.time() - start, 2)
        }

    if not all_chunks:
        return {
            "status": "warning", 
            "message": "No valid text chunks created.", 
            "doc_id": doc_id,
            "chunks": 0,
            "pages_processed": len(pages),
            "seconds": round(time.time() - start, 2)
        }

    # 3️⃣ Index
    try:
        dim = model.get_sentence_embedding_dimension()
        ensure_collection(dim)

        chunk_texts = [c["text"] for c in all_chunks]
        metadata_list = [
            {
                "id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "page": c["page"],
                "language": c["language"],
                "source": c["source"],
                "is_scanned": c["is_scanned"],
                "text": c["text"]
            }
            for c in all_chunks
        ]

        embed_and_upsert(chunk_texts, metadata_list)
        bm25.add_documents(chunk_texts, metadata_list)
        print(f"[INDEX] Indexed {len(all_chunks)} chunks in Qdrant + BM25.")
    except Exception as e:
        print(f"[INDEX ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"Indexing failed: {str(e)}",
            "doc_id": doc_id,
            "chunks": 0,
            "pages_processed": len(pages),
            "seconds": round(time.time() - start, 2)
        }

    elapsed = time.time() - start
    print(f"[DONE] Completed ingestion in {elapsed:.2f}s ✅")

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks": len(all_chunks),
        "pages_processed": len(pages),
        "seconds": round(elapsed, 2),
        "message": f"Indexed {len(all_chunks)} chunks from {len(pages)} pages."
    }