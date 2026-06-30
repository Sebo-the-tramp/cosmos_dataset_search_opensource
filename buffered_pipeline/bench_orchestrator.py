from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

import embedder

VIDEO_COUNT = 100
EXTRAPOLATED_VIDEOS = 300_000
EXTRAPOLATED_GPUS = 2
DATA_DIR = Path("/home/cavadalab/Documents/scsv/covision/cosmos_cds/buffered_pipeline/data")
EMBEDDINGS_OUTPUT = DATA_DIR / "bench_100_embeddings.parquet"
REPORT_OUTPUT = DATA_DIR / "bench_100_report.json"

console = Console()


def prepare_outputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = [x for x in [EMBEDDINGS_OUTPUT, REPORT_OUTPUT] if x.exists()]
    if not existing:
        return

    answer = input(f"{existing} exist. Type OVERWRITE to reset them: ")
    assert answer == "OVERWRITE"
    for path in existing:
        path.unlink()


def seconds(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def save_report(report: dict[str, Any]) -> None:
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


def print_report(report: dict[str, Any]) -> None:
    table = Table(title="100 Video Embed Benchmark")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for key, value in report.items():
        table.add_row(key, str(value))

    console.print(table)


def add_stage_percentages(report: dict[str, Any]) -> dict[str, Any]:
    process_seconds = report["process_seconds"]
    stage_keys = ["decode_wait_seconds", "processor_seconds", "transfer_seconds", "forward_seconds", "output_seconds"]
    return {
        **report,
        **{f"{key}_pct": round(report[key] / process_seconds * 100, 2) for key in stage_keys},
    }


def add_extrapolation(report: dict[str, Any]) -> dict[str, Any]:
    seconds_per_video = report["process_seconds"] / report["videos"]
    single_gpu_seconds = seconds_per_video * EXTRAPOLATED_VIDEOS
    multi_gpu_seconds = single_gpu_seconds / EXTRAPOLATED_GPUS

    return {
        **report,
        "extrapolated_videos": EXTRAPOLATED_VIDEOS,
        "extrapolated_gpus": EXTRAPOLATED_GPUS,
        "extrapolated_assumption": "decode+embed only, zero download overhead, ideal 2 GPU scaling",
        "extrapolated_single_gpu_seconds": round(single_gpu_seconds, 4),
        "extrapolated_single_gpu_hours": round(single_gpu_seconds / 3600, 4),
        "extrapolated_single_gpu_days": round(single_gpu_seconds / 86400, 4),
        "extrapolated_2gpu_seconds": round(multi_gpu_seconds, 4),
        "extrapolated_2gpu_hours": round(multi_gpu_seconds / 3600, 4),
        "extrapolated_2gpu_days": round(multi_gpu_seconds / 86400, 4),
        "extrapolated_2gpu_videos_per_second": round(report["videos_per_second"] * EXTRAPOLATED_GPUS, 4),
    }


def main() -> None:
    prepare_outputs()
    embedder.seed_everything()
    embedder.OUTPUT_PATH = EMBEDDINGS_OUTPUT

    total_start = time.perf_counter()
    videos = embedder.list_videos()[:VIDEO_COUNT]
    assert len(videos) == VIDEO_COUNT

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(embedder.MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(embedder.MODEL_NAME, trust_remote_code=True).to(
        embedder.DEVICE, dtype=embedder.DTYPE
    ).eval()
    model_load_seconds = seconds(load_start)

    rows = []
    decode_wait_seconds = 0.0
    processor_seconds = 0.0
    transfer_seconds = 0.0
    forward_seconds = 0.0
    output_seconds = 0.0
    batches = [videos[start : start + embedder.BATCH_SIZE] for start in range(0, len(videos), embedder.BATCH_SIZE)]

    process_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=embedder.DECODE_WORKERS) as decode_pool:
        futures = embedder.submit_videos(batches[0], decode_pool)
        for batch_idx, batch_paths in enumerate(tqdm(batches, desc="bench", unit="batch")):
            next_futures = (
                embedder.submit_videos(batches[batch_idx + 1], decode_pool) if batch_idx + 1 < len(batches) else None
            )

            stage_start = time.perf_counter()
            batch_videos = embedder.collect_videos(futures)
            decode_wait_seconds += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            inputs = processor(videos=batch_videos)
            processor_seconds += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            inputs = inputs.to(embedder.DEVICE, dtype=embedder.DTYPE)
            torch.cuda.synchronize()
            transfer_seconds += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            with torch.inference_mode():
                out = model.get_video_embeddings(**inputs)
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            embeddings = out.visual_proj.float().cpu().numpy().tolist()
            output_seconds += time.perf_counter() - stage_start

            rows.extend(
                {
                    "video_path": str(video_path),
                    "chunk": video_path.parent.name,
                    "embedding": embedding,
                }
                for video_path, embedding in zip(batch_paths, embeddings, strict=True)
            )
            futures = next_futures

    process_seconds = seconds(process_start)

    write_start = time.perf_counter()
    embedder.save_embeddings(rows)
    write_seconds = seconds(write_start)
    total_seconds = seconds(total_start)

    report = add_extrapolation(add_stage_percentages({
        "videos": len(rows),
        "num_frames": embedder.NUM_FRAMES,
        "decode_resolution": embedder.DECODE_RESOLUTION,
        "batch_size": embedder.BATCH_SIZE,
        "decode_workers": embedder.DECODE_WORKERS,
        "device": embedder.DEVICE,
        "dtype": str(embedder.DTYPE),
        "model_load_seconds": model_load_seconds,
        "process_seconds": process_seconds,
        "decode_wait_seconds": round(decode_wait_seconds, 4),
        "processor_seconds": round(processor_seconds, 4),
        "transfer_seconds": round(transfer_seconds, 4),
        "forward_seconds": round(forward_seconds, 4),
        "output_seconds": round(output_seconds, 4),
        "write_seconds": write_seconds,
        "total_seconds": total_seconds,
        "seconds_per_video": round(process_seconds / len(rows), 4),
        "videos_per_second": round(len(rows) / process_seconds, 4),
        "max_gpu_memory_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 4),
        "embeddings_output": str(EMBEDDINGS_OUTPUT),
        "report_output": str(REPORT_OUTPUT),
    }))

    save_report(report)
    print_report(report)


if __name__ == "__main__":
    main()
