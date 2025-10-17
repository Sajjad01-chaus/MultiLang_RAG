# app/chunker.py
import hashlib
import math
from typing import List, Dict, Optional
import nltk
from transformers import AutoTokenizer
import numpy as np

# Lazy download - first run will download punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Default tokenizer/model for token counting 
DEFAULT_TOKENIZER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _make_chunk_id(doc_id: str, page: int, chunk_idx: int) -> str:
    return f"{doc_id}::p{page}::c{chunk_idx}"


def token_count(text: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    toks = tokenizer.encode(text, add_special_tokens=False)
    return len(toks)


def sentence_tokenize(text: str) -> List[str]:
    # nltk sentence tokenizer works reasonably across languages for basic segmentation
    return nltk.tokenize.sent_tokenize(text)


def smart_chunk_text(
    doc_id: str,
    page_num: int,
    text: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    min_chunk_tokens: int = 40,
    dedupe_threshold: Optional[float] = 0.93,
    vectorizer=None,  # optional function that maps text -> embedding vector (numpy)
) -> List[Dict]:
    """
    Convert a large text into chunks that respect sentence boundaries and token budget.

    Returns list of dicts:
      {
        "chunk_id": str,
        "doc_id": str,
        "page": int,
        "text": str,
        "token_count": int
      }
    """
    if not text or not text.strip():
        print(f"[CHUNKER] Empty text for page {page_num}")
        return []

    try:
        sentences = sentence_tokenize(text)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        chunks = []
        cur_sentences = []
        cur_tokens = 0
        chunk_idx = 0

        # Helper to flush current buffer into chunk
        def flush_chunk(keep_overlap_sentences: List[str] = None):
            nonlocal cur_sentences, cur_tokens, chunk_idx, chunks
            if not cur_sentences:
                return
            chunk_text = " ".join(cur_sentences).strip()
            tcount = len(tokenizer.encode(chunk_text, add_special_tokens=False, truncation=False))
            chunk = {
                "chunk_id": _make_chunk_id(doc_id, page_num, chunk_idx),
                "doc_id": doc_id,
                "page": page_num,
                "text": chunk_text,
                "token_count": tcount,
            }
            chunks.append(chunk)
            chunk_idx += 1
            # overlap for next chunk
            cur_sentences = keep_overlap_sentences[:] if keep_overlap_sentences else []
            cur_tokens = len(tokenizer.encode(" ".join(cur_sentences), add_special_tokens=False, truncation=False)) if cur_sentences else 0

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s_tokens = len(tokenizer.encode(s, add_special_tokens=False, truncation=False))
            if cur_tokens + s_tokens <= max_tokens:
                cur_sentences.append(s)
                cur_tokens += s_tokens
            else:
                # flush keeping overlap sentences (approx overlap_tokens worth)
                # naive strategy: keep last sentences until overlap_token budget is roughly met
                keep = []
                keep_tokens = 0
                while cur_sentences and keep_tokens < overlap_tokens:
                    last = cur_sentences[-1]
                    ltoks = len(tokenizer.encode(last, add_special_tokens=False, truncation=False))
                    keep.insert(0, last)
                    keep_tokens += ltoks
                    cur_sentences.pop()
                # When overlap would be tiny and current buffer empty, force include current sentence to avoid infinite loop
                if not cur_sentences and s_tokens > max_tokens:
                    # The single sentence is larger than max_tokens; we must hard-split sentence by token chunks
                    # fallback: split sentence into approximate token slices using tokenizer
                    token_ids = tokenizer.encode(s, add_special_tokens=False, truncation=False)
                    start = 0
                    while start < len(token_ids):
                        end = min(start + max_tokens, len(token_ids))
                        sub_tokens = token_ids[start:end]
                        sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        chunk = {
                            "chunk_id": _make_chunk_id(doc_id, page_num, chunk_idx),
                            "doc_id": doc_id,
                            "page": page_num,
                            "text": sub_text,
                            "token_count": len(sub_tokens),
                        }
                        chunks.append(chunk)
                        chunk_idx += 1
                        start = end - overlap_tokens if end < len(token_ids) else end
                    cur_sentences = []
                    cur_tokens = 0
                else:
                    # flush current buffer and start fresh with kept overlap + current sentence
                    flush_chunk(keep_overlap_sentences=keep)
                    cur_sentences = keep[:]  # already copied by flush_chunk but safe
                    cur_tokens = len(tokenizer.encode(" ".join(cur_sentences), add_special_tokens=False, truncation=False)) if cur_sentences else 0
                    # now try to add current sentence again
                    if len(tokenizer.encode(s, add_special_tokens=False, truncation=False)) + cur_tokens <= max_tokens:
                        cur_sentences.append(s)
                        cur_tokens += s_tokens
                    else:
                        # if still doesn't fit (rare), push as its own chunk
                        flush_chunk(keep_overlap_sentences=[])
                        cur_sentences = [s]
                        cur_tokens = s_tokens

        # final flush - THIS WAS MISSING!
        flush_chunk()

        # Filter out chunks that are too small
        chunks = [c for c in chunks if c["token_count"] >= min_chunk_tokens]
        
        print(f"[CHUNKER] Created {len(chunks)} chunks for page {page_num}")
        return chunks

    except Exception as e:
        print(f"[CHUNKER ERROR] Failed to chunk page {page_num}: {e}")
        import traceback
        traceback.print_exc()
        return []  # ✅ Return empty list instead of None