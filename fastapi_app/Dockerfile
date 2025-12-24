FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Copy application code
COPY . /app
# Ensure embeddings are present (mount or bake)
VOLUME ["/app/data/embeddings", "/app/data/embedding_json"]

EXPOSE 8000

# SentenceTransformer will download weights on first run; mount a volume to cache if needed.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
