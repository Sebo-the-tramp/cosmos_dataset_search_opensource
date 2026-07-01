from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

import downloader
import embedder

START_ZIP = 1_000
ZIP_LIMIT = 2_146
ZIP_END = START_ZIP + ZIP_LIMIT
GPU_IDS = [0, 1]
DOWNLOAD_WORKERS = 4
MAX_READY_ZIPS = 4
MAX_RAM_GB = 64.0
MAX_RAM_BYTES = int(MAX_RAM_GB * 1024**3)
STAGE_SPACE_MULTIPLIER = 2.0
KEEP_EXTRACTED = False
EXTRAPOLATED_VIDEOS = 300_000
STOP = "__STOP__"

BASE_DIR = Path("/home/cavadalab/Documents/scsv/covision/cosmos_cds/buffered_pipeline")
RUN_NAME = f"zips_{START_ZIP:04d}_{ZIP_END - 1:04d}"
OUTPUT_PATH = BASE_DIR / "data" / f"embeddings_{RUN_NAME}.parquet"
REPORT_PATH = BASE_DIR / "data" / f"orchestrator_report_{RUN_NAME}.json"
SHARD_DIR = BASE_DIR / "data" / f"embedding_shards_{RUN_NAME}"

console = Console()


def prepare_outputs() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = [x for x in [OUTPUT_PATH, REPORT_PATH, SHARD_DIR] if x.exists()]
    if existing:
        answer = input(f"{existing} exist. Type OVERWRITE to reset them: ")
        assert answer == "OVERWRITE"

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    if REPORT_PATH.exists():
        REPORT_PATH.unlink()
    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)

    SHARD_DIR.mkdir(parents=True)
    downloader.DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloader.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    downloader.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


def shard_videos(videos: list[Path]) -> list[list[Path]]:
    return [videos[idx::len(GPU_IDS)] for idx in range(len(GPU_IDS))]


def gpu_worker(device_id: int, task_queue: mp.Queue, result_queue: mp.Queue, output_path: Path) -> None:
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    embedder.seed_everything()
    processor = AutoProcessor.from_pretrained(embedder.MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(embedder.MODEL_NAME, trust_remote_code=True).to(device, dtype=embedder.DTYPE).eval()

    with pq.ParquetWriter(output_path, embedder.SCHEMA) as writer:
        while True:
            task = task_queue.get()
            if task == STOP:
                return

            video_paths = [Path(x) for x in task["videos"]]
            start = time.perf_counter()
            written = embedder.write_video_paths(model, processor, writer, video_paths, device)
            seconds = time.perf_counter() - start
            result_queue.put(
                {
                    "zip": task["zip"],
                    "gpu_id": device_id,
                    "videos": written,
                    "seconds": round(seconds, 4),
                    "videos_per_second": round(written / seconds, 4),
                    "output_path": str(output_path),
                }
            )


def merge_shards(shard_paths: list[Path]) -> None:
    tables = [pq.read_table(path) for path in shard_paths]
    pq.write_table(pa.concat_tables(tables), OUTPUT_PATH)


def save_report(report: dict[str, Any]) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def print_report(report: dict[str, Any]) -> None:
    table = Table(title="Streaming Multi-GPU Orchestrator")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for key, value in report.items():
        if key != "zips":
            table.add_row(key, str(value))

    console.print(table)


def live_report(zip_records: list[dict[str, Any]], start: float, status: str) -> dict[str, Any]:
    seconds = time.perf_counter() - start
    videos = sum(x["videos"] for x in zip_records)
    videos_per_second = videos / seconds if videos else 0.0
    extrapolated_seconds = EXTRAPOLATED_VIDEOS / videos_per_second if videos_per_second else 0.0
    return {
        "status": status,
        "run_name": RUN_NAME,
        "start_zip": START_ZIP,
        "zip_end": ZIP_END,
        "zip_limit": ZIP_LIMIT,
        "zips_done": len(zip_records),
        "videos": videos,
        "gpus": GPU_IDS,
        "download_workers": DOWNLOAD_WORKERS,
        "max_ready_zips": MAX_READY_ZIPS,
        "max_ram_gb": MAX_RAM_GB,
        "stage_space_multiplier": STAGE_SPACE_MULTIPLIER,
        "batch_size": embedder.BATCH_SIZE,
        "decode_workers_per_gpu": embedder.DECODE_WORKERS,
        "decode_resolution": embedder.DECODE_RESOLUTION,
        "seconds": round(seconds, 4),
        "videos_per_second": round(videos_per_second, 4),
        "extrapolated_videos": EXTRAPOLATED_VIDEOS,
        "extrapolated_seconds": round(extrapolated_seconds, 4),
        "extrapolated_hours": round(extrapolated_seconds / 3600, 4),
        "extrapolated_days": round(extrapolated_seconds / 86400, 4),
        "output_path": str(OUTPUT_PATH),
        "report_path": str(REPORT_PATH),
        "zips": zip_records,
    }


def estimated_stage_bytes(item: dict[str, Any]) -> int:
    return int(item["size_bytes"] * STAGE_SPACE_MULTIPLIER)


def reserved_pending_bytes(pending: dict[Future[dict[str, Any]], dict[str, Any]]) -> int:
    return sum(estimated_stage_bytes(item) for item in pending.values())


def planned_free_ram_bytes(pending: dict[Future[dict[str, Any]], dict[str, Any]]) -> int:
    usage = shutil.disk_usage(downloader.DATA_DIR)
    used = usage.total - usage.free
    return MAX_RAM_BYTES - used - reserved_pending_bytes(pending)


def collect_finished_zips(
    pending: dict[Future[dict[str, Any]], dict[str, Any]],
    ready: deque[dict[str, Any]],
) -> None:
    done = [future for future in pending if future.done()]
    for future in done:
        ready.append(future.result())
        del pending[future]


def submit_staging_jobs(
    zips: list[dict[str, Any]],
    next_zip: int,
    pending: dict[Future[dict[str, Any]], dict[str, Any]],
    ready: deque[dict[str, Any]],
    extract_pool: ThreadPoolExecutor,
) -> int:
    while next_zip < len(zips) and len(pending) < DOWNLOAD_WORKERS and len(pending) + len(ready) < MAX_READY_ZIPS:
        item = zips[next_zip]
        if planned_free_ram_bytes(pending) <= estimated_stage_bytes(item):
            break
        pending[extract_pool.submit(extract_zip, item)] = item
        next_zip += 1
    return next_zip


def wait_for_ready_zip(
    pending: dict[Future[dict[str, Any]], dict[str, Any]],
    ready: deque[dict[str, Any]],
) -> None:
    if ready:
        return

    assert pending
    wait(pending, return_when=FIRST_COMPLETED)
    collect_finished_zips(pending, ready)


def extract_zip(item: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    assert estimated_stage_bytes(item) <= MAX_RAM_BYTES
    extracted = downloader.download_extract_zip(item)
    extracted_dir = Path(extracted["extracted_dir"])
    videos = sorted(extracted_dir.glob("**/*.mp4"))
    assert len(videos) == extracted["videos"]
    return {
        "item": item,
        "extracted": extracted,
        "extracted_dir": extracted_dir,
        "videos": videos,
        "download_seconds": extracted["download_seconds"],
        "extract_seconds": extracted["extract_seconds"],
        "download_extract_seconds": round(time.perf_counter() - start, 4),
    }


def embed_extracted_zip(
    extracted_zip: dict[str, Any],
    task_queues: list[mp.Queue],
    result_queue: mp.Queue,
) -> dict[str, Any]:
    item = extracted_zip["item"]
    videos = extracted_zip["videos"]
    embed_start = time.perf_counter()
    shards = shard_videos(videos)
    for queue, gpu_id, shard in zip(task_queues, GPU_IDS, shards, strict=True):
        queue.put({"zip": item["zip"], "gpu_id": gpu_id, "videos": [str(x) for x in shard]})

    gpu_results = [result_queue.get() for _ in GPU_IDS]
    assert all(x["zip"] == item["zip"] for x in gpu_results)
    assert sum(x["videos"] for x in gpu_results) == len(videos)

    if not KEEP_EXTRACTED:
        shutil.rmtree(extracted_zip["extracted_dir"])

    return {
        "zip": item["zip"],
        "size_bytes": item["size_bytes"],
        "videos": len(videos),
        "reused_extracted": extracted_zip["extracted"]["reused"],
        "download_seconds": extracted_zip["download_seconds"],
        "extract_seconds": extracted_zip["extract_seconds"],
        "download_extract_seconds": extracted_zip["download_extract_seconds"],
        "embed_seconds": round(time.perf_counter() - embed_start, 4),
        "seconds": round(extracted_zip["download_extract_seconds"] + time.perf_counter() - embed_start, 4),
        "free_gb": downloader.free_gb(),
        "gpu_results": sorted(gpu_results, key=lambda x: x["gpu_id"]),
    }


def main() -> None:
    mp.set_start_method("spawn", force=True)
    prepare_outputs()

    all_zips = downloader.list_zip_items()
    zips = all_zips[START_ZIP:ZIP_END]
    assert len(zips) == ZIP_LIMIT

    task_queues = [mp.Queue() for _ in GPU_IDS]
    result_queue: mp.Queue = mp.Queue()
    shard_paths = [SHARD_DIR / f"gpu_{gpu_id}.parquet" for gpu_id in GPU_IDS]
    processes = [
        mp.Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue, shard_path))
        for gpu_id, task_queue, shard_path in zip(GPU_IDS, task_queues, shard_paths, strict=True)
    ]

    for process in processes:
        process.start()

    start = time.perf_counter()
    zip_records = []
    pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    ready: deque[dict[str, Any]] = deque()
    next_zip = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as extract_pool:
        with tqdm(total=len(zips), desc="zip", unit="zip") as progress:
            while len(zip_records) < len(zips):
                collect_finished_zips(pending, ready)
                next_zip = submit_staging_jobs(zips, next_zip, pending, ready, extract_pool)
                wait_for_ready_zip(pending, ready)

                extracted_zip = ready.popleft()
                next_zip = submit_staging_jobs(zips, next_zip, pending, ready, extract_pool)
                zip_records.append(embed_extracted_zip(extracted_zip, task_queues, result_queue))
                progress.update(1)
                save_report(live_report(zip_records, start, "running"))

    for queue in task_queues:
        queue.put(STOP)

    for process in processes:
        process.join()
        assert process.exitcode == 0

    merge_shards(shard_paths)
    report = live_report(zip_records, start, "done")
    save_report(report)
    print_report(report)


if __name__ == "__main__":
    main()
