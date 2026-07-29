#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cavadalab/Documents/scsv/covision/cosmos_cds
MILVUS_CONTAINER=milvus-standalone
MILVUS_HEALTH=http://127.0.0.1:9091/healthz
BACKEND_HEALTH=http://127.0.0.1:5000/health
VIEWER_URL=http://127.0.0.1:5000/viewer/
MILVUS_WAIT_SECONDS=120
BACKEND_WAIT_SECONDS=300
LOG_LINES=100

healthy() {
    curl -fsS "$1" >/dev/null 2>&1
}

wait_for_milvus() {
    for _ in $(seq 1 "$MILVUS_WAIT_SECONDS"); do
        if healthy "$MILVUS_HEALTH"; then
            echo "Milvus ready"
            return
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "$MILVUS_CONTAINER")" != "true" ]]; then
            echo "Milvus exited during startup"
            docker logs --tail "$LOG_LINES" "$MILVUS_CONTAINER"
            exit 1
        fi
        sleep 1
    done
    echo "Milvus did not become ready"
    docker logs --tail "$LOG_LINES" "$MILVUS_CONTAINER"
    exit 1
}

cd "$ROOT"

if healthy "$MILVUS_HEALTH"; then
    echo "Milvus already running"
else
    echo "Starting Milvus"
    bash database/standalone_embed.sh start
    wait_for_milvus
fi

if healthy "$BACKEND_HEALTH"; then
    echo "Backend already running: $VIEWER_URL"
else
    echo "Starting backend: $VIEWER_URL"
    cd "$ROOT/backend"
    "$ROOT/.venv/bin/python" app.py &
    BACKEND_PID=$!
    trap 'kill "$BACKEND_PID" 2>/dev/null' INT TERM
    for _ in $(seq 1 "$BACKEND_WAIT_SECONDS"); do
        if healthy "$BACKEND_HEALTH"; then
            echo "Backend ready: $VIEWER_URL"
            wait "$BACKEND_PID"
            exit
        fi
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo "Backend exited during startup"
            wait "$BACKEND_PID"
        fi
        sleep 1
    done
    echo "Backend did not become ready"
    kill "$BACKEND_PID"
    wait "$BACKEND_PID"
fi
