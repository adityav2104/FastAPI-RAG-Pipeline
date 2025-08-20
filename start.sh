#!/usr/bin/env bash
set -e

# 1) Ensure FAISS index exists (idempotent)
if [ ! -f "faiss_db/index.faiss" ] || [ ! -f "faiss_db/index.pkl" ]; then
  echo "[start.sh] FAISS index missing. Building..."
  python prepare_db.py
else
  echo "[start.sh] FAISS index found. Skipping build."
fi

# 2) Start FastAPI on the port Render provides
exec uvicorn main:app --host 0.0.0.0 --port $PORT
