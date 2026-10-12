#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SPECTRUMCLAW_PROJECT_ROOT:-/workspace/YiFan/SpectrumClaw}"
cd "$PROJECT_ROOT"
mkdir -p logs/ingest

LOG="logs/ingest/parallel_mineru_chain_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/ingest/server_parallel_chain.sh > "$LOG" 2>&1 &
PID=$!

echo "$PID" > runs/ingest.pid
echo "PID=$PID"
echo "LOG=$LOG"
