# ── Base image ──────────────────────────────────────────────────
FROM python:3.10-slim

# ── Set working directory ──────────────────────────────────────
WORKDIR /app

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy all application code ─────────────────────────────
# Required because requirements.txt contains "-e ." which needs
# setup.py and the networksecurity package present to install
COPY . .

# ── Install Python dependencies ─────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Create non-root user for security ───────────────────────────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ── Expose port ─────────────────────────────────────────────────
EXPOSE 8080

# ── Environment variables ───────────────────────────────────────
ENV PYTHONUNBUFFERED=1

# ── Entrypoint + CMD ──────────────────────────────────────────
ENTRYPOINT ["python"]
CMD ["app.py"]