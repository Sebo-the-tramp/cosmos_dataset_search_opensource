#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=milvus-standalone
MILVUS_GID=999
MILVUS_IMAGE=milvusdb/milvus:v3.0-beta@sha256:0a030f267a16358901c662adac02813027ace9e23bfbe6db66528e4fd1c28168
MILVUS_UID=999
STOP_SECONDS=30

cd "$(dirname "$0")"

VOLUME_DIR="$PWD/volumes/milvus"
INSERT_LOG="$VOLUME_DIR/data/insert_log"
LEGACY_INSERT_LOG="$VOLUME_DIR/data/var/lib/milvus/data/insert_log"

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
    if [[ -d "$LEGACY_INSERT_LOG" && ! -e "$INSERT_LOG" ]]; then
        docker run --rm --user root -v "$VOLUME_DIR:/var/lib/milvus" --entrypoint ln "$MILVUS_IMAGE" -s var/lib/milvus/data/insert_log /var/lib/milvus/data/insert_log
        echo "Repaired legacy Milvus storage path"
    fi
}

case "${1:-start}" in
    start)
        prepare_volume
        docker compose up -d
        ;;
    stop)
        if container_exists && container_running; then
            docker stop --timeout "$STOP_SECONDS" "$CONTAINER_NAME"
        else
            echo "$CONTAINER_NAME already stopped"
        fi
        ;;
    restart)
        prepare_volume
        if container_exists && container_running; then
            docker stop --timeout "$STOP_SECONDS" "$CONTAINER_NAME"
        fi
        docker compose up -d
        ;;
    upgrade)
        docker compose pull
        docker compose up -d --force-recreate
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
