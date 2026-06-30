# Cosmos CDS

Local text-to-video search over NVIDIA PhysicalAI autonomous-driving clips using Cosmos video embeddings, Milvus vector search, a Flask API, and a static web viewer.

Architecture sketch: `architecture.excalidraw`.

```text
Hugging Face zips -> mp4 extraction -> Cosmos video embeddings -> Parquet -> Milvus -> Flask API -> web viewer
```

## Repository Status

This is publishable as a research/internal pipeline after generated data is removed from the repository. It is not a packaged product yet: most scripts intentionally use hardcoded constants at the top of each file, absolute local paths, and fail-fast assertions.

Before a public release, add a license and re-check the licenses/terms for the NVIDIA dataset and Cosmos model.

## Requirements

| Requirement | Purpose |
| --- | --- |
| Linux workstation | The scripts use local absolute paths and Docker. |
| NVIDIA GPU with CUDA | Cosmos embedding inference. |
| 1 or more GPUs | The orchestrator shards work across `GPU_IDS`. |
| Docker | Runs standalone Milvus. |
| `uv` | Creates the Python environment and installs packages. |
| Python 3.10 | Current tested environment for the main `.venv`. |
| Hugging Face access | Downloads dataset zips and loads `nvidia/Cosmos-Embed1-448p`. |
| `/mnt/ramcds` with about 64 GB | Temporary zip/download/extract workspace. |
| Disk space | Stores Parquet embeddings and Milvus volume data. |

Runtime services:

| Service | Port | Purpose |
| --- | ---: | --- |
| Milvus standalone | `19530` | Vector database API. |
| Milvus health | `9091` | Container health check. |
| Embedded etcd | `2379` | Milvus internal metadata. |
| Flask backend | `5000` | Text embedding and Milvus search API. |
| Static web viewer | `8000` | Browser UI. |

## Install

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds
uv venv --python 3.10
uv pip install flask pymilvus pyarrow rich tqdm numpy transformers accelerate huggingface-hub decord torch
```

If the dataset/model are not already cached:

```bash
.venv/bin/huggingface-cli login
```

The backend currently uses `LOCAL_FILES_ONLY = True`, so `nvidia/Cosmos-Embed1-448p` must already be present in the Hugging Face cache before backend startup.

## Components

| Path | Purpose |
| --- | --- |
| `buffered_pipeline/downloader.py` | Downloads zips from `nvidia/PhysicalAI-Autonomous-Vehicles`, extracts only `.mp4`, deletes zips, and writes `/mnt/ramcds/processed.json`. |
| `buffered_pipeline/embedder.py` | Embeds extracted videos with `nvidia/Cosmos-Embed1-448p`. |
| `buffered_pipeline/orchestrator.py` | Streams download/extract/embed with multiple download workers and one embedder process per GPU. |
| `buffered_pipeline/bench_orchestrator.py` | Benchmarks a small run. |
| `buffered_pipeline/merge_448_embeddings.py` | Merges existing 448p Parquet outputs. |
| `database/standalone_embed.sh` | Starts/stops/deletes standalone Milvus in Docker. |
| `database/create_collection.py` | Creates an empty Milvus collection. |
| `database/ingest.py` | Drops existing collections after confirmation, creates `cosmos_cds_test_00`, and ingests the merged Parquet file. |
| `backend/app.py` | Flask API that embeds text queries and searches Milvus. |
| `web_viewer/` | Static UI calling `http://127.0.0.1:5000/search`; can download the returned video-path list as JSON. |
| `backend/download_video_list.py` | Downloads result paths and videos into the shared dataset folder. |
| `download_video/download_video_list.py` | Thin wrapper for local `*_paths.json` files. |

## Generated Data

These paths are generated and should not be committed:

| Path | Meaning |
| --- | --- |
| `/mnt/ramcds` | Temporary zip and extraction workspace. |
| `buffered_pipeline/data/` | Parquet embeddings, reports, shards, logs. |
| `database/volumes/` | Docker Milvus persistent data. |
| `buffered_pipeline/test_embeddings/videos*/` | Local test clips. |
| `buffered_pipeline/test_embeddings/embeddings/` | Local test embedding outputs. |
| `/data0/sebastian.cavada/datasets/cosmos-cds/data/` | Downloaded query JSON files and videos grouped by query. |

## Build Embeddings

Prepare the temp workspace:

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

The current full merged 448p embedding artifact expected by ingest is:

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

`database/ingest.py` is destructive by design: it asks for `DELETE`, drops all existing Milvus collections, creates `cosmos_cds_test_00`, and inserts the Parquet embeddings.

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

## Start Backend

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds
bash backend/run.sh
```

API:

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | Backend status. |
| `GET /search?word=stroller&quantity=5` | Text-to-video search. |
| `POST /download` | Saves `{query}_paths.json` and native MP4s under `/data0/sebastian.cavada/datasets/cosmos-cds/data/{query}/`. |

## Start Web Viewer

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds/web_viewer
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000
```

Run a query, then click `Download data` to save the returned video paths and native MP4s on the backend. Downloaded results render as playable videos in the result grid.

## Normal Startup

If embeddings are already loaded into Milvus:

```bash
cd /home/cavadalab/Documents/scsv/covision/cosmos_cds/database
bash standalone_embed.sh start

cd /home/cavadalab/Documents/scsv/covision/cosmos_cds
bash backend/run.sh

cd /home/cavadalab/Documents/scsv/covision/cosmos_cds/web_viewer
python3 -m http.server 8000
```

## Known Limitations

- Scripts use hardcoded local paths and constants rather than CLI arguments.
- `backend/app.py` loads one text embedding model on startup and expects Milvus to already contain the collection.
- `database/standalone_embed.sh` uses `sudo docker`.
- `database/test_curl.sh` is an old direct-Milvus REST example and is not the main search test path.
- The repository does not yet include a `pyproject.toml` or pinned lockfile.
