#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cavadalab/Documents/scsv/covision/cosmos_cds
BACKEND_PATTERN="$ROOT/.venv/bin/python app.py"
DOWNLOAD_PATTERN="$ROOT/backend/.venv/bin/python $ROOT/backend/download_video_list.py"
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
