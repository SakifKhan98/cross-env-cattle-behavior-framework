#!/usr/bin/env bash
# Start the Cattle Vision Framework web app (FastAPI + built React frontend).
# Usage: bash scripts/start_app.sh [--port PORT]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$REPO_ROOT/ui/dist"
PORT="${PORT:-8000}"

# Parse optional --port argument
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Check the built frontend exists
if [[ ! -d "$DIST_DIR" || ! -f "$DIST_DIR/index.html" ]]; then
  echo ""
  echo "ERROR: ui/dist/ not found or incomplete."
  echo ""
  echo "Build the frontend first:"
  echo "  cd ui && npm run build"
  echo ""
  exit 1
fi

# Activate conda environment
CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_INIT" ]]; then
  CONDA_INIT="/home/zxs12/miniconda3/etc/profile.d/conda.sh"
fi
if [[ ! -f "$CONDA_INIT" ]]; then
  echo "ERROR: conda init script not found. Activate cattletransformer manually."
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_INIT"
conda activate cattletransformer

URL="http://localhost:${PORT}"
echo ""
echo "Starting Cattle Vision Framework at ${URL}"
echo "Press Ctrl+C to stop."
echo ""

# Open browser after a short delay so the server is ready
(sleep 2 && \
  if command -v xdg-open &>/dev/null; then xdg-open "$URL"; \
  elif command -v open &>/dev/null; then open "$URL"; \
  fi) &

# Start FastAPI (replaces this shell process)
cd "$REPO_ROOT"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
