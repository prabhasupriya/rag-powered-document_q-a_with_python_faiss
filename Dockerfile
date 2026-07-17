FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / pdfplumber's image backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create dirs that will be volume-mounted
RUN mkdir -p /app/data /app/index_store

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
