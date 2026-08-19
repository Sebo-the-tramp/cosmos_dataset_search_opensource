import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from cosmos_cds.paths import DATA_DIR, TEMP_DIR

os.environ.setdefault("HF_MODULES_CACHE", str(TEMP_DIR / "cosmos_cds_hf_modules"))

import physical_ai_av
from tqdm import tqdm


CLIP_DIR = DATA_DIR / "clips"
PATHS_SUFFIX = "_paths.json"
OUTPUT_DIR_SUFFIX = "_output_dir.txt"
REQUEST_ENV = "COSMOS_CDS_DOWNLOAD_REQUEST"
PROGRESS_ENV = "COSMOS_CDS_DOWNLOAD_PROGRESS"
VIDEO_CODEC = "h264_nvenc"
VIDEO_CQ = "20"
VIDEO_PRESET = "p4"

AVDI = None
CAMERA = None


def query_name_from_word(word: str) -> str:
    return "_".join(word.strip().lower().split())


def strip_video_id(path: str) -> str:
    return Path(path).name.split(".")[0].lower()


def get_avdi_camera() -> tuple[Any, str]:
    global AVDI, CAMERA
    if AVDI is None:
        AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()
        CAMERA = AVDI.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
    return AVDI, CAMERA


def save_paths_json(query_name: str, paths: list[str], output_dir: Path, overwrite: bool) -> Path:
    json_path = output_dir / f"{query_name}{PATHS_SUFFIX}"
    if overwrite or not json_path.exists():
        with open(json_path, "w") as f:
            json.dump(paths, f, indent=2)
    return json_path


def clip_path(clip_id: str, output_dir: Path) -> Path:
    return output_dir / f"{clip_id}.mp4"


def video_codec(path: Path) -> str:
    return subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip()


def transcode_video(source_path: Path, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            VIDEO_PRESET,
            "-cq",
            VIDEO_CQ,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ],
        check=True,
    )
    tmp_path.replace(output_path)


def write_clip_video(clip_id: str, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite and video_codec(output_path) == "h264":
        return False

    source_path = output_path.with_suffix(".source.mp4")
    avdi, camera = get_avdi_camera()
    chunk_filename = avdi.features.get_chunk_feature_filename(avdi.get_clip_chunk(clip_id), camera)
    clip_files = avdi.features.get_clip_files_in_zip(clip_id, camera)
    with avdi.open_file(chunk_filename, maybe_stream=True) as f:
        with zipfile.ZipFile(f, "r") as zf:
            with zf.open(clip_files["video"]) as src:
                with open(source_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    transcode_video(source_path, output_path)
    source_path.unlink()
    return True


def write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    tmp_path.replace(path)


def download_query(word: str, paths: list[str], overwrite: bool = False, output_dir: Path = CLIP_DIR) -> dict[str, Any]:
    assert paths, "No video paths provided"
    query_name = query_name_from_word(word)
    query_dir = DATA_DIR / query_name
    query_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_paths_json(query_name, paths, query_dir, overwrite)
    output_dir_path = query_dir / f"{query_name}{OUTPUT_DIR_SUFFIX}"
    output_dir_path.write_text(str(output_dir))
    progress_path = Path(os.environ[PROGRESS_ENV]) if PROGRESS_ENV in os.environ else None
    seen = set()
    written = 0
    skipped = 0
    done = 0

    write_progress(
        progress_path,
        {"state": "running", "query_name": query_name, "done": done, "total": len(paths), "written": written, "skipped": skipped},
    )

    for path in tqdm(paths, desc=f"Downloading {query_name}"):
        assert isinstance(path, str) and path.strip(), f"Invalid path: {path}"
        clip_id = strip_video_id(path)
        if clip_id in seen:
            skipped += 1
            done += 1
            write_progress(
                progress_path,
                {"state": "running", "query_name": query_name, "done": done, "total": len(paths), "written": written, "skipped": skipped},
            )
            continue
        seen.add(clip_id)
        if write_clip_video(clip_id, clip_path(clip_id, output_dir), overwrite):
            written += 1
        else:
            skipped += 1
        done += 1
        write_progress(
            progress_path,
            {"state": "running", "query_name": query_name, "done": done, "total": len(paths), "written": written, "skipped": skipped},
        )

    result = {
        "query_name": query_name,
        "query_dir": str(query_dir),
        "clip_dir": str(output_dir),
        "output_dir": str(output_dir),
        "paths_json": str(json_path),
        "output_dir_path": str(output_dir_path),
        "video_count": len(seen),
        "written": written,
        "skipped": skipped,
    }
    write_progress(progress_path, {"state": "done", "done": done, "total": len(paths), **result})
    return result


if __name__ == "__main__":
    request_path = Path(os.environ[REQUEST_ENV])
    with open(request_path) as f:
        request = json.load(f)
    print(
        json.dumps(
            download_query(
                request["word"],
                request["paths"],
                request["overwrite"],
                Path(request.get("output_dir", CLIP_DIR)),
            )
        )
    )
