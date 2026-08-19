from __future__ import annotations

import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import decord
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from cosmos_cds.paths import DATA_DIR, WORK_DIR

SEED = 0
MODEL_NAME = "nvidia/Cosmos-Embed1-448p"
VIDEO_DIR = WORK_DIR / "camera_front_wide_120fov"
OUTPUT_PATH = DATA_DIR / "buffered_pipeline" / "embeddings.parquet"
SCHEMA = pa.schema(
    [
        ("video_path", pa.string()),
        ("chunk", pa.string()),
        ("embedding", pa.list_(pa.float32())),
    ]
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
NUM_FRAMES = 8
DECODE_RESOLUTION = 448
BATCH_SIZE = 16
DECODE_WORKERS = 16
SAVE_EVERY = 10
MAX_VIDEOS: int | None = None


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_output(output_path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        return []

    answer = input(f"{output_path} exists. Type OVERWRITE to reset it, Enter to continue: ")
    if answer == "OVERWRITE":
        return []

    table = pq.read_table(output_path)
    return table.to_pylist()


def list_videos() -> list[Path]:
    videos = sorted(VIDEO_DIR.glob("**/*.mp4"))
    assert videos
    return videos[:MAX_VIDEOS] if MAX_VIDEOS else videos


def load_video(video_path: Path) -> np.ndarray:
    reader = decord.VideoReader(str(video_path), width=DECODE_RESOLUTION, height=DECODE_RESOLUTION)
    frame_ids = np.linspace(0, len(reader) - 1, NUM_FRAMES, dtype=int).tolist()
    frames = reader.get_batch(frame_ids).asnumpy()
    return np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))


def load_videos(video_paths: list[Path], decode_pool: ThreadPoolExecutor | None = None) -> np.ndarray:
    if decode_pool:
        videos = list(decode_pool.map(load_video, video_paths))
    else:
        videos = [load_video(x) for x in video_paths]

    return np.concatenate(videos, axis=0)


def submit_videos(video_paths: list[Path], decode_pool: ThreadPoolExecutor) -> list[Future[np.ndarray]]:
    return [decode_pool.submit(load_video, video_path) for video_path in video_paths]


def collect_videos(futures: list[Future[np.ndarray]]) -> np.ndarray:
    return np.concatenate([future.result() for future in futures], axis=0)


def save_embeddings(rows: list[dict[str, Any]], output_path: Path = OUTPUT_PATH) -> None:
    pq.write_table(rows_to_table(rows), output_path)


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.table(
        {
            "video_path": pa.array([x["video_path"] for x in rows], type=pa.string()),
            "chunk": pa.array([x["chunk"] for x in rows], type=pa.string()),
            "embedding": pa.array([x["embedding"] for x in rows], type=pa.list_(pa.float32())),
        },
        schema=SCHEMA,
    )


def write_rows(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    writer.write_table(rows_to_table(rows))


def embedding_rows(video_paths: list[Path], embeddings: list[list[float]]) -> list[dict[str, Any]]:
    return [
        {"video_path": str(video_path), "chunk": video_path.parent.name, "embedding": embedding}
        for video_path, embedding in zip(video_paths, embeddings, strict=True)
    ]


def write_video_paths(
    model: AutoModel,
    processor: AutoProcessor,
    writer: pq.ParquetWriter,
    video_paths: list[Path],
    device: str,
    desc: str | None = None,
) -> int:
    batches = [video_paths[start : start + BATCH_SIZE] for start in range(0, len(video_paths), BATCH_SIZE)]
    if not batches:
        return 0

    written = 0
    iterator = enumerate(tqdm(batches, desc=desc, unit="batch")) if desc else enumerate(batches)
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as decode_pool:
        futures = submit_videos(batches[0], decode_pool)
        for batch_idx, batch_paths in iterator:
            next_futures = submit_videos(batches[batch_idx + 1], decode_pool) if batch_idx + 1 < len(batches) else None
            write_rows(writer, embedding_rows(batch_paths, embed_videos(model, processor, collect_videos(futures), device)))
            written += len(batch_paths)
            futures = next_futures

    return written


def embed_batch(
    model: AutoModel,
    processor: AutoProcessor,
    video_paths: list[Path],
    decode_pool: ThreadPoolExecutor | None = None,
    device: str = DEVICE,
) -> list[list[float]]:
    videos = load_videos(video_paths, decode_pool)
    return embed_videos(model, processor, videos, device)


def embed_videos(
    model: AutoModel,
    processor: AutoProcessor,
    videos: np.ndarray,
    device: str = DEVICE,
) -> list[list[float]]:
    inputs = processor(videos=videos).to(device, dtype=DTYPE)

    with torch.inference_mode():
        out = model.get_video_embeddings(**inputs)

    embeddings = out.visual_proj.float().cpu().numpy()
    return embeddings.tolist()


def embed_video_paths(
    video_paths: list[Path],
    output_path: Path,
    device: str,
    desc: str,
    initial_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seed_everything()
    torch.cuda.set_device(int(device.split(":")[-1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(initial_rows) if initial_rows else []

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device, dtype=DTYPE).eval()

    if not video_paths:
        save_embeddings(rows, output_path)
        return {"videos": len(rows), "processed_videos": 0, "output_path": str(output_path), "device": device}

    with pq.ParquetWriter(output_path, SCHEMA) as writer:
        if rows:
            write_rows(writer, rows)
        written = write_video_paths(model, processor, writer, video_paths, device, desc)

    return {"videos": len(rows) + written, "processed_videos": len(video_paths), "output_path": str(output_path), "device": device}


def main() -> None:
    seed_everything()
    rows = prepare_output(OUTPUT_PATH)
    embedded = {x["video_path"] for x in rows}
    video_paths = [x for x in list_videos() if str(x) not in embedded]

    result = embed_video_paths(video_paths, OUTPUT_PATH, DEVICE, "embed", rows)
    assert result["processed_videos"] == len(video_paths)


if __name__ == "__main__":
    main()
