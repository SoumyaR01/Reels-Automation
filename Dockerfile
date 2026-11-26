# ReelForge Movie RAG System Dockerfile
FROM python:3.11-slim

# Install system dependencies including FFmpeg and font libraries for Devanagari subtitles
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    fonts-indic \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY reelforge/ reelforge/
COPY auto_reels.py .

ENV PYTHONPATH=/app
ENV REELFORGE_WORK_DIR=/app/workdir
ENV REELFORGE_OUTPUT_DIR=/app/reels

EXPOSE 8000

# Default command starts FastAPI Studio
CMD ["python", "-m", "reelforge.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
