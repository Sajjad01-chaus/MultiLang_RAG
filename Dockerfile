# ---- Base image ----
    FROM python:3.10-slim

    # ---- System dependencies ----
    RUN apt-get update && apt-get install -y \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Working directory ----
    WORKDIR /code
    
    # ---- Upgrade pip and install dependencies ----
    COPY requirements.txt /code/
    RUN pip install --upgrade pip setuptools wheel && \
        pip install --no-cache-dir -r requirements.txt
    
    # Copy the rest of the project
    COPY . /code/
    
    # ---- Default command (override in docker-compose) ----
    CMD ["bash"]