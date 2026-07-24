<div align="center">
  <h1>COSMOS Dataset Search</h1>
  <h3>Local Cosmos Dataset Search-style video retrieval over NVIDIA PhysicalAI autonomous-driving clips.</h3>
  <p><sub>Developed during an internship at <strong>CovisionLAB</strong></sub></p>
</div>

---


Local Cosmos Dataset Search-style video retrieval over NVIDIA PhysicalAI autonomous-driving clips.

This repo uses `nvidia/Cosmos-Embed1-448p` to embed videos and text, stores video embeddings in Milvus, serves search through a Flask backend, and serves the browser UI from the same backend.

Architecture:

![Cosmos CDS architecture](architecture.png)

```text
Hugging Face zips -> mp4 extraction -> Cosmos video embeddings -> Parquet -> Milvus -> Flask backend + viewer
```

## Status

This is a research/internal pipeline with hardcoded constants at the top of the scripts. It is intentionally simple and fail-fast, not a packaged product.

The repository code is licensed under Apache-2.0. NVIDIA model weights, NVIDIA dataset files, videos, generated embeddings, and Milvus volumes containing generated embeddings are not redistributed by this project.

## Requirements

| Requirement | Purpose |
| --- | --- |
| Linux workstation | The scripts use local paths, Docker, and CUDA tooling. |
| NVIDIA GPU with CUDA | Cosmos embedding inference. |
| Docker | Runs standalone Milvus. |
| `uv` | Creates Python environments and installs packages. |
| Python 3.10 | Main backend / embedding environment. |
| Python 3.12 | `physical-ai-av` downloader environment. |
| Hugging Face access | Downloads dataset zips and loads `nvidia/Cosmos-Embed1-448p`. |
| `ffmpeg` / `ffprobe` with NVENC | Downloads and serves browser-playable H.264 MP4s. |
| `/mnt/ramcds` with about 64 GB | Temporary zip/download/extract workspace. |
| Milvus disk space | Stores the vector database under `database/volumes/`. |

Runtime services:

| Service | Port | Purpose |
| --- | ---: | --- |
| Milvus standalone | `19530` | Vector database API. |
| Milvus health | `9091` | Container health check. |
| Embedded etcd | `2379` | Milvus internal metadata. |
| Flask backend/viewer | `5000` | API, web UI, and local MP4 serving. |

## Install

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds

uv venv --python 3.10
uv pip install flask pymilvus pyarrow rich tqdm numpy transformers accelerate huggingface-hub decord torch

uv venv --python 3.12 backend/.venv
uv pip install --python backend/.venv/bin/python physical-ai-av==0.2.2 tqdm
```

Login if the model or dataset are not already cached:

```bash
.venv/bin/huggingface-cli login
```

`backend/app.py` uses `LOCAL_FILES_ONLY = True`, so `nvidia/Cosmos-Embed1-448p` must already exist in the Hugging Face cache before backend startup.

## Structure

| Path | Purpose |
| --- | --- |
| `buffered_pipeline/downloader.py` | Downloads NVIDIA PhysicalAI AV zips, extracts only `.mp4`, deletes zips, and writes `/mnt/ramcds/processed.json`. |
| `buffered_pipeline/embedder.py` | Embeds extracted videos with `nvidia/Cosmos-Embed1-448p`. |
| `buffered_pipeline/orchestrator.py` | Streams download/extract/embed with multiple download workers and one embedder process per GPU. |
| `buffered_pipeline/merge_448_embeddings.py` | Merges 448p Parquet outputs. |
| `run.sh` | Starts Milvus if needed, then starts the backend/viewer if needed. |
| `database/compose.yaml` | Docker Compose config for standalone Milvus. |
| `database/standalone_embed.sh` | Starts/stops/deletes standalone Milvus in Docker. |
| `database/ingest.py` | Drops existing collections after confirmation, creates `cosmos_cds_test_00`, and ingests the merged Parquet file. |
| `backend/app.py` | Unified Flask app: serves `/viewer/`, embeds queries, searches Milvus, starts downloads, serves MP4s. |
| `backend/download_video_list.py` | Worker script used by `/download`; extracts result clips through `physical-ai-av` and transcodes them. |
| `web_viewer/` | Browser UI assets served by `backend/app.py`. |

## Generated Data

These paths are generated and should not be committed:

| Path | Meaning |
| --- | --- |
| `/mnt/ramcds` | Temporary zip and extraction workspace. |
| `buffered_pipeline/data/` | Parquet embeddings, reports, shards, logs. |
| `database/volumes/` | Docker Milvus persistent data. |
| `buffered_pipeline/test_embeddings/` | Local test clips, reports, and test embeddings. |
| `/data0/sebastian.cavada/datasets/cosmos-cds/data/` | Downloaded query manifests plus shared clips in `clips/`. |
| `/tmp/cosmos_cds_download_requests/` | Backend download job request/progress files. |
| `/tmp/cosmos_cds_video_queries/` | Uploaded video-query files. |

Generated embeddings from gated or restricted datasets should stay local unless the dataset owner explicitly permits redistribution.

## Build Embeddings

Prepare temp storage:

```bash
sudo mkdir -p /mnt/ramcds
sudo chown "$USER:$USER" /mnt/ramcds
```

Edit the constants at the top of `buffered_pipeline/orchestrator.py` before a run:

| Constant | Meaning |
| --- | --- |
| `START_ZIP` | First dataset zip index. |
| `ZIP_LIMIT` | Number of zips to process. |
| `GPU_IDS` | GPUs used for embedding. |
| `DOWNLOAD_WORKERS` | Parallel zip download/extract workers. |
| `MAX_READY_ZIPS` | Maximum staged zips waiting for embedding. |
| `MAX_RAM_GB` | Maximum intended `/mnt/ramcds` usage. |

Run the streaming pipeline:

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds/buffered_pipeline
../.venv/bin/python orchestrator.py
```

The current merged 448p embedding artifact expected by ingest is:

```text
buffered_pipeline/data/embeddings_0000_3145.448.parquet
```

## Start Milvus

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds/database
bash standalone_embed.sh start
```

Check:

```bash
docker ps
curl http://127.0.0.1:9091/healthz
```

Stop or delete:

```bash
bash standalone_embed.sh stop
bash standalone_embed.sh delete
```

## Ingest

`database/ingest.py` is destructive by design: it asks for `DELETE`, drops all existing Milvus collections, creates `cosmos_cds_test_00`, and inserts the configured Parquet embeddings.

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds/database
../.venv/bin/python ingest.py
```

Milvus collection settings:

| Setting | Value |
| --- | --- |
| Collection | `cosmos_cds_test_00` |
| Vector field | `embedding` |
| Dimension | `768` |
| Metric | `COSINE` |
| Output fields | `video_path`, `chunk` |

## Run App

Start the unified backend/viewer:

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds
./run.sh
```

`run.sh` starts Milvus through Docker Compose when needed, then starts the Flask backend in the foreground.

Open:

```text
http://127.0.0.1:5000/viewer/
```

The viewer supports:

| Action | Meaning |
| --- | --- |
| Word query | Embeds text and searches Milvus. |
| Video query | Uploads a local video, embeds it, and searches Milvus. |
| Download data | Saves query manifests under `data/{query}/` and deduplicated MP4s under `data/clips/`. |
| History | Lists previously downloaded query folders and re-runs them. |
| Copy path | Copies the displayed local or source video path from a result card. |
| Video playback | Serves downloaded clips through `/video/{query}/{clip_id}.mp4`. |

API:

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | Backend status. |
| `GET /viewer/` | Web UI. |
| `GET /history` | Downloaded query folders. |
| `DELETE /history/<query_name>` | Deletes one query manifest and prunes unreferenced shared clips. |
| `GET /search?word=pickup&quantity=5` | Text-to-video search. |
| `POST /search_video` | Video-to-video search. |
| `POST /download` | Starts a background clip download job. |
| `GET /download/<job_id>` | Polls download job progress. |
| `GET /video/<query_name>/<filename>` | Serves a downloaded MP4. |

## Normal Startup

If embeddings are already loaded into Milvus:

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds
./run.sh
```

Then open `http://127.0.0.1:5000/viewer/`.

## Known Limitations

- Scripts use hardcoded local paths and constants rather than CLI arguments.
- `backend/app.py` loads one Cosmos model on startup and expects Milvus to already contain the collection.
- `/download` needs `backend/.venv` with `physical-ai-av` and working NVIDIA dataset access.
- MP4 download/transcode expects `ffmpeg`, `ffprobe`, and NVENC support.
- The repository does not yet include a `pyproject.toml` or pinned lockfile.

## License

The repository code and documentation are licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE.md`.
