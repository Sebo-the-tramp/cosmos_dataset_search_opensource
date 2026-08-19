from __future__ import annotations

import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_tree
from tqdm import tqdm

from cosmos_cds.paths import WORK_DIR

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"
ZIP_FOLDER = "camera/camera_front_wide_120fov"
HF_LINK_ZIPS = f"https://huggingface.co/datasets/{REPO_ID}/tree/main/{ZIP_FOLDER}"

DATA_DIR = WORK_DIR
DOWNLOAD_DIR = DATA_DIR / "zips"
EXTRACT_DIR = DATA_DIR / "camera_front_wide_120fov"
PROCESSED_METADATA = DATA_DIR / "processed.json"

MAX_RAM_GB = 64.0
MAX_RAM_BYTES = int(MAX_RAM_GB * 1024**3)
ZIP_SPACE_MULTIPLIER = 2.0
VIDEO_SUFFIX = ".mp4"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_bytes() -> int:
    return shutil.disk_usage(DATA_DIR).free


def used_bytes() -> int:
    usage = shutil.disk_usage(DATA_DIR)
    return usage.total - usage.free


def free_gb() -> float:
    return round(free_bytes() / 1024**3, 2)


def prepare_folders() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    if PROCESSED_METADATA.exists():
        answer = input(f"{PROCESSED_METADATA} exists. Type OVERWRITE to reset it, Enter to continue: ")
        if answer == "OVERWRITE":
            PROCESSED_METADATA.write_text("[]\n")
    else:
        PROCESSED_METADATA.write_text("[]\n")


def read_processed_metadata() -> dict[str, dict[str, Any]]:
    with PROCESSED_METADATA.open() as f:
        rows = json.load(f)

    assert isinstance(rows, list)
    return {x["zip"]: x for x in rows}


def save_processed_metadata(records: dict[str, dict[str, Any]]) -> None:
    rows = sorted(records.values(), key=lambda x: x["zip"])
    PROCESSED_METADATA.write_text(json.dumps(rows, indent=2) + "\n")


def list_zip_items() -> list[dict[str, Any]]:
    files = list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=ZIP_FOLDER, recursive=False)
    zips = [{"zip": x.path, "size_bytes": x.size} for x in files if x.path.endswith(".zip")]
    return sorted(zips, key=lambda x: x["zip"])


def has_space(size_bytes: int) -> bool:
    needed = int(size_bytes * ZIP_SPACE_MULTIPLIER)
    return used_bytes() + needed < MAX_RAM_BYTES


def record(item: dict[str, Any], status: str, error: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "zip": item["zip"],
        "status": status,
        "error": error,
        "size_bytes": item["size_bytes"],
        "free_gb": free_gb(),
        "updated_at": now(),
        **extra,
    }


def zip_target_dir(item: dict[str, Any]) -> Path:
    return EXTRACT_DIR / Path(item["zip"]).stem


def download_extract_zip(item: dict[str, Any]) -> dict[str, Any]:
    target_dir = zip_target_dir(item)
    if target_dir.exists():
        videos = sorted(target_dir.glob("**/*.mp4"))
        assert videos
        return record(
            item,
            "done",
            extracted_dir=str(target_dir),
            videos=len(videos),
            reused=True,
            download_seconds=0.0,
            extract_seconds=0.0,
        )

    download_start = time.perf_counter()
    local_zip = Path(
        hf_hub_download(REPO_ID, filename=item["zip"], repo_type=REPO_TYPE, local_dir=DOWNLOAD_DIR)
    )
    download_seconds = time.perf_counter() - download_start
    tmp_dir = EXTRACT_DIR / f"{local_zip.stem}.tmp"

    assert not tmp_dir.exists()
    tmp_dir.mkdir(parents=True)

    extract_start = time.perf_counter()
    with zipfile.ZipFile(local_zip) as zf:
        videos = [x for x in zf.infolist() if not x.is_dir() and x.filename.lower().endswith(VIDEO_SUFFIX)]
        assert videos
        for video in videos:
            zf.extract(video, tmp_dir)
    extract_seconds = time.perf_counter() - extract_start

    tmp_dir.rename(target_dir)
    local_zip.unlink()
    return record(
        item,
        "done",
        extracted_dir=str(target_dir),
        videos=len(videos),
        reused=False,
        download_seconds=round(download_seconds, 4),
        extract_seconds=round(extract_seconds, 4),
    )


def process_zip(item: dict[str, Any], records: dict[str, dict[str, Any]]) -> bool:
    records[item["zip"]] = download_extract_zip(item)
    save_processed_metadata(records)
    return has_space(0)


def main() -> None:
    prepare_folders()
    records = read_processed_metadata()
    done_zips = {k for k, v in records.items() if v["status"] == "done"}
    zips = [x for x in list_zip_items() if x["zip"] not in done_zips]

    for item in tqdm(zips, desc="download", unit="zip"):
        if not has_space(item["size_bytes"]):
            records[item["zip"]] = record(item, "no_space")
            save_processed_metadata(records)
            break

        records[item["zip"]] = record(item, "processing")
        save_processed_metadata(records)
        enough_space = process_zip(item, records)

        if not enough_space:
            break


if __name__ == "__main__":
    main()
