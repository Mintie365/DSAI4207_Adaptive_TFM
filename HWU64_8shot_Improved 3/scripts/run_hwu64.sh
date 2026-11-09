#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$PROJECT_DIR/.venv312/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv312/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found. Create .venv312 and install requirements.txt first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}"

STAGE="${1:-all}"
echo "Project: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "Stage: $STAGE"
echo "HF metadata timeout: ${HF_HUB_ETAG_TIMEOUT}s"
echo "HF download timeout: ${HF_HUB_DOWNLOAD_TIMEOUT}s"
exec "$PYTHON_BIN" "$PROJECT_DIR/run.py" "$STAGE"
