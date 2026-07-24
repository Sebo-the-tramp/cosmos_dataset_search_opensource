#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=milvus-standalone
MILVUS_GID=999
MILVUS_IMAGE=milvusdb/milvus:v3.0-beta
MILVUS_UID=999

cd "$(dirname "$0")"

VOLUME_DIR="$PWD/volumes/milvus"

container_exists() {
    docker inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]
}

prepare_volume() {
    mkdir -p "$VOLUME_DIR"
    if [[ "$(stat -c '%u:%g' "$VOLUME_DIR")" != "$MILVUS_UID:$MILVUS_GID" ]]; then
        docker run --rm --user root -v "$VOLUME_DIR:/var/lib/milvus" --entrypoint chown "$MILVUS_IMAGE" -R "$MILVUS_UID:$MILVUS_GID" /var/lib/milvus
    fi
}

case "${1:-start}" in
    start)
        prepare_volume
        if container_exists; then
            if container_running; then
                echo "$CONTAINER_NAME already running"
            else
                docker start "$CONTAINER_NAME"
            fi
        else
            docker compose up -d
        fi
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
