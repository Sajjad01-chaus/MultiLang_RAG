"""
workers/tasks.py

Background ingestion tasks for large PDFs.
"""

from celery import shared_task
from app.ingest import ingest_pdf

@shared_task(bind=True)
def ingest_pdf_task(self, path: str, doc_id: str = None, lang_hint: str = None):
    """
    Background ingestion using Celery.
    """
    try:
        result = ingest_pdf(path, doc_id, lang_hint)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
