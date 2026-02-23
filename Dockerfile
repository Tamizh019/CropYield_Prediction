# ─── Base Image ───────────────────────────────────────────────
FROM python:3.10-slim

# ─── Metadata ─────────────────────────────────────────────────
LABEL maintainer="Tamizharasan"
LABEL description="AgriVision - YieldMax Precision Model for Crop Yield Prediction"

# ─── Environment Setup ────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# ─── System Dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ────────────────────────────────────────
WORKDIR /app

# ─── Copy Requirements First (layer caching) ─────────────────
COPY requirements.txt .

# ─── Install Python Dependencies ─────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ─── Copy Application Code ────────────────────────────────────
COPY . .

# ─── Create Required Directories ─────────────────────────────
RUN mkdir -p models data logs static

# ─── Expose Port (Hugging Face uses 7860) ─────────────────────
EXPOSE 7860

# ─── Health Check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# ─── Run Application ──────────────────────────────────────────
# Uses gunicorn for production; Hugging Face requires port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "120", "--preload", "app:app"]
