#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cavadalab/Documents/scsv/covision/cosmos_cds
MILVUS_HEALTH=http://127.0.0.1:9091/healthz
BACKEND_HEALTH=http://127.0.0.1:5000/health
VIEWER_URL=http://127.0.0.1:5000/viewer/
WAIT_SECONDS=120

healthy() {
    curl -fsS "$1" >/dev/null
}

wait_for() {
    local url="$1"
    local name="$2"
    for _ in $(seq 1 "$WAIT_SECONDS"); do
        if healthy "$url"; then
            echo "$name ready"
            return
        fi
        sleep 1
    done
    echo "$name did not become ready"
    exit 1
}

cd "$ROOT"

if healthy "$MILVUS_HEALTH"; then
    echo "Milvus already running"
else
    echo "Starting Milvus"
    bash database/standalone_embed.sh start
    wait_for "$MILVUS_HEALTH" "Milvus"
fi

if healthy "$BACKEND_HEALTH"; then
    echo "Backend already running: $VIEWER_URL"
else
    echo "Starting backend: $VIEWER_URL"
    cd "$ROOT/backend"
    exec "$ROOT/.venv/bin/python" app.py
fi
