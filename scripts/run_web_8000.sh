#!/usr/bin/env bash
# 先关闭占用 8000 的进程，再在 0.0.0.0:8000 启动 API + Web 看板。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

pkill -f "uvicorn app.main:app" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8000/tcp 2>/dev/null || true
else
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti :8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  fi
fi
sleep 0.5

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
