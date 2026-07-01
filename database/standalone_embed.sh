#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "${1:-start}" in
    start)
        docker compose up -d
        ;;
    stop)
        docker compose down
        ;;
    restart)
        docker compose down
        docker compose up -d
        ;;
    upgrade)
        docker compose pull
        docker compose up -d
        ;;
    delete)
        read -r -p "Delete Milvus container and data? Type DELETE: " answer
        [[ "$answer" == "DELETE" ]]
        docker compose down -v
        rm -rf volumes
        ;;
    *)
        echo "usage: bash standalone_embed.sh start|stop|restart|upgrade|delete"
        exit 1
        ;;
esac
