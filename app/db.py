# app/db.py
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Singleton instances
_qdrant = QdrantClient(":memory:")
_model = SentenceTransformer(EMBED_MODEL)

def get_qdrant():
    return _qdrant

def get_model():
    return _model