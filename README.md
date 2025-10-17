🧠 Multilingual RAG PDF Question Answering SystemAn intelligent document retrieval system that uses Retrieval-Augmented Generation (RAG) to answer questions from PDF documents in 50+ languages, running entirely on your local machine.✨ Core Features📄 PDF Ingestion: Upload and process both digital and scanned PDFs with automatic OCR.🌍 Multilingual: Natively handles 50+ languages (e.g., English, Hindi, Bengali, Chinese).🔍 Hybrid Search: Combines semantic (vector) and keyword (BM25) search for high accuracy.🤖 Local & Private: Uses a local LLM (flan-t5-base) for answer generation. No API keys or data ever leave your machine.🚀 Quick Start1. PrerequisitesPython 3.8+Tesseract OCR (see installation guide and ensure it's in your system's PATH)2. Setup# Clone the repository
git clone [https://github.com/Sajjad01-chaus/MultiLang_RAG.git](https://github.com/Sajjad01-chaus/MultiLang_RAG.git)
cd MultiLang_RAG

# Create and activate a virtual environment
python -m venv myvenv
# Windows:
myvenv\Scripts\activate
# Mac/Linux:
source myvenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download necessary NLTK data
python -c "import nltk; nltk.download('punkt')"
3. ConfigurationCreate a .env file in the root directory and add the following configuration. This is the recommended setup for local use.# .env
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SMALL_LLM_MODEL=google/flan-t5-base
QDRANT_COLLECTION=pdf_chunks
4. Run the ApplicationStart the FastAPI backend server. The first time you run this, it will download the language models (this may take a few minutes).uvicorn app.app:app --reload
The API will be available at http://127.0.0.1:8000.📖 Basic Usage (API)1. Ingest a PDFSend a POST request to /ingest with your PDF file.Example:curl -X POST "[http://127.0.0.1:8000/ingest](http://127.0.0.1:8000/ingest)" -F "file=@/path/to/your/document.pdf"
Response:{
  "status": "success",
  "doc_id": "c14b9884-1b10-4090-9ac5-37584cbca3a7",
  "chunks": 4,
  ...
}
2. Ask a QuestionSend a POST request to /query with your question.Example:curl -X POST "[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic of this document?"}'
Response:{
    "answer": "The document is a No Objection Certificate (NOC) for obtaining an e-passport...",
    "provenance": [ ... ],
    ...
}
