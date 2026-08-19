#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BACKEND_PATTERN="$ROOT/.venv/bin/python -m cosmos_cds.backend.app"
DOWNLOAD_PATTERN="$ROOT/.venv/bin/python -m cosmos_cds.backend.download_video_list"
WAIT_SECONDS=10

stop_processes() {
    local name="$1"
    local pattern="$2"
    local pids

    if pids="$(pgrep -f "$pattern")"; then
        echo "Stopping $name: $pids"
        kill $pids
        for _ in $(seq 1 "$WAIT_SECONDS"); do
            if ! pgrep -f "$pattern" >/dev/null; then
                echo "$name stopped"
                return
            fi
            sleep 1
        done
        pids="$(pgrep -f "$pattern")"
        echo "Force stopping $name: $pids"
        kill -9 $pids
    else
        echo "$name not running"
    fi
}

cd "$ROOT"

stop_processes "backend" "$BACKEND_PATTERN"
stop_processes "download worker" "$DOWNLOAD_PATTERN"

echo "Stopping Milvus"
bash database/standalone_embed.sh stop
echo "All services stopped"
