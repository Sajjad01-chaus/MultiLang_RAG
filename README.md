# Mini RAG Document QA System

A FastAPI-based RAG (Retrieval-Augmented Generation) system for PDF document question-answering using HuggingFace API.

## 🚀 Features

- **PDF Upload & Processing**: Upload PDF documents with OCR support
- **RAG Pipeline**: Advanced retrieval and generation using vector search
- **HuggingFace Integration**: Uses HuggingFace Inference API for LLM responses
- **Web Interface**: Streamlit-based UI for easy interaction
- **Docker Support**: Complete containerization with Docker Compose

## 🏗️ Architecture

- **FastAPI Backend**: REST API for document processing and queries
- **Qdrant Vector DB**: Vector storage and similarity search
- **Redis + Celery**: Background task processing
- **Streamlit UI**: Web interface for user interaction
- **HuggingFace API**: LLM inference for answer generation

## 📋 Prerequisites

- Docker & Docker Compose
- HuggingFace API Key

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Cerebralzip
```

### 2. Configure Environment Variables
```bash
# Copy the environment template
cp .env.example .env

# Edit .env file and add your HuggingFace API key
HF_API_KEY=your_huggingface_api_key_here
```

### 3. Build and Run
```bash
docker-compose up --build
```

## 🌐 Access Points

- **FastAPI API**: http://localhost:8000
- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## 📚 API Endpoints

### Upload Document
```bash
POST /ingest
Content-Type: multipart/form-data

# Upload a PDF file
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@document.pdf" \
     -F "language=en"
```

### Query Document
```bash
POST /query
Content-Type: application/json

{
  "query": "What is the main topic of this document?"
}
```

### Health Check
```bash
GET /health
```

## 🔧 Configuration

### Environment Variables
- `HF_API_KEY`: Your HuggingFace API key (required)
- `SMALL_LLM_MODEL`: HuggingFace model to use (default: microsoft/DialoGPT-medium)
- `QDRANT_URL`: Qdrant vector database URL
- `REDIS_URL`: Redis URL for task queue
- `EMBED_MODEL`: Sentence transformer model for embeddings

### Models Used
- **LLM**: microsoft/DialoGPT-medium (via HuggingFace API)
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Vector DB**: Qdrant

## 🔒 Security Notes

- `.env` file is gitignored and contains sensitive API keys
- Never commit API keys to version control
- Use environment variables for all sensitive configuration

## 📁 Project Structure

```
├── app/                    # FastAPI application
│   ├── app.py             # Main API endpoints
│   ├── llm_utils.py       # HuggingFace API integration
│   ├── ingest.py          # Document processing
│   ├── retrieve.py        # Vector search
│   └── ...
├── UI/                    # Streamlit interface
│   └── app.py
├── workers/               # Celery background tasks
├── docker-compose.yaml    # Container orchestration
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── .env.example         # Environment template
```

## 🚀 Usage

1. **Upload a PDF**: Use the Streamlit UI or API to upload documents
2. **Ask Questions**: Query the uploaded documents using natural language
3. **Get Answers**: Receive contextual answers with source citations

## 🔧 Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn app.app:app --reload

# Run Streamlit
streamlit run UI/app.py
```

### Docker Development
```bash
# Build and run all services
docker-compose up --build

# Run specific service
docker-compose up fastapi
```

## 📝 License

This project is for educational/assignment purposes.

## 🤝 Contributing

This is an assignment submission. Please follow the assignment guidelines for any modifications.
