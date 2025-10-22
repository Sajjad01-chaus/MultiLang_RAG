# app/db.py
import os
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "pdf_chunks"

# --- Shared global instances ---
_qdrant = None
_model = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        qdrant_path = os.path.join(os.getcwd(), "qdrant_storage")
        os.makedirs(qdrant_path, exist_ok=True)
        _qdrant = QdrantClient(path=qdrant_path)
    return _qdrant


def get_model():
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(model, texts):
    # Always normalize embeddings for consistent cosine similarity
    emb = model.encode(texts, normalize_embeddings=True)
    return np.array(emb)
