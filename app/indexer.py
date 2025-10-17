# app/indexer.py
import numpy as np
from typing import List, Dict, Any
from qdrant_client.http import models as qmodels
from app.db import get_qdrant, get_model, COLLECTION

def ensure_collection(dim: int):
    """Ensure Qdrant collection exists with correct dimensionality."""
    qdrant = get_qdrant()
    try:
        qdrant.get_collection(COLLECTION)
    except Exception:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

def embed_and_upsert(texts: List[str], metadatas: List[Dict[str, Any]]):
    """Embed list of texts and upsert into Qdrant with metadata payload."""
    qdrant = get_qdrant()
    model = get_model()
    
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    points = [
        qmodels.PointStruct(
            id=hash(meta["id"]) % (2 ** 63),
            vector=vec.tolist(),
            payload=meta,
        )
        for vec, meta in zip(embeddings, metadatas)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(points)