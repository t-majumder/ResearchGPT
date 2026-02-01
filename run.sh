#!/usr/bin/env bash

echo "🚀 Starting ResearchGPT (Backend + Frontend)"

# -----------------------------
# Resolve project root
# -----------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------
# Activate Python venv
# -----------------------------
echo "🐍 Activating virtual environment"
source "$PROJECT_ROOT/gpt/bin/activate"

# -----------------------------
# Start Backend
# -----------------------------
echo "⚙️ Starting Backend (FastAPI)"
cd "$PROJECT_ROOT/backend" || exit 1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# -----------------------------
# Start Frontend
# -----------------------------
echo "🌐 Starting Frontend"
cd "$PROJECT_ROOT/frontend" || exit 1
python -m http.server 5173 &
FRONTEND_PID=$!

# -----------------------------
# Graceful shutdown
# -----------------------------
trap "echo '🛑 Stopping servers'; kill $BACKEND_PID $FRONTEND_PID" SIGINT SIGTERM

echo "✅ Backend:  http://localhost:8000"
echo "✅ Frontend: http://localhost:5173"

wait
