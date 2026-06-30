import os
import random
import subprocess
import threading
import uuid
import json
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/data0/.cache")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/cosmos_cds_hf_modules")

import decord
import numpy as np
import torch
from flask import Flask, Response, jsonify, redirect, request, send_file
from pymilvus import MilvusClient
from transformers import AutoModel, AutoProcessor


SEED = 0
HOST = "127.0.0.1"
PORT = 5000
MODEL_NAME = "nvidia/Cosmos-Embed1-448p"
MILVUS_URI = "http://127.0.0.1:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "cosmos_cds_test_00"
VECTOR_FIELD = "embedding"
OUTPUT_FIELDS = ["video_path", "chunk"]
DEFAULT_QUANTITY = 10
MAX_QUANTITY = 1000
REMOVED_IDS = {467240255860630002}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
LOCAL_FILES_ONLY = True
BACKEND_DIR = Path(__file__).resolve().parent
DOWNLOAD_PYTHON = str(BACKEND_DIR / ".venv" / "bin" / "python")
DOWNLOAD_SCRIPT = BACKEND_DIR / "download_video_list.py"
DOWNLOAD_REQUEST_DIR = Path("/tmp/cosmos_cds_download_requests")
WEB_VIEWER_DIR = Path(__file__).resolve().parent.parent / "web_viewer"
DATA_DIR = Path("/data0/sebastian.cavada/datasets/cosmos-cds/data")
VIDEO_QUERY_DIR = Path("/tmp/cosmos_cds_video_queries")
PATHS_SUFFIX = "_paths.json"
VIDEO_CODEC = "h264"
VIDEO_ENCODER = "h264_nvenc"
NUM_FRAMES = 8
DECODE_RESOLUTION = 448
DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
VIDEO_LOCKS: dict[Path, threading.Lock] = {}
BROWSER_VIDEOS: set[Path] = set()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_processor_model() -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE, dtype=DTYPE).eval()
    return processor, model


def embed_text(text: str) -> list[float]:
    inputs = PROCESSOR(text=text).to(DEVICE)
    with torch.inference_mode():
        output = MODEL.get_text_embeddings(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    return output.text_proj.float().cpu().numpy()[0].tolist()


def parse_request() -> tuple[str, int]:
    payload = request.get_json(silent=True) or {}
    word = payload.get("word") or payload.get("query") or request.args.get("word") or request.args.get("query")
    quantity = parse_quantity(payload)
    assert isinstance(word, str) and word.strip()
    return word.strip(), quantity


def parse_quantity(payload: dict[str, Any] | None = None) -> int:
    payload = payload or request.get_json(silent=True) or {}
    quantity = int(
        payload.get("quantity")
        or payload.get("top_k")
        or request.form.get("quantity")
        or request.form.get("top_k")
        or request.args.get("quantity")
        or request.args.get("top_k")
        or DEFAULT_QUANTITY
    )
    assert 0 < quantity <= MAX_QUANTITY
    return quantity


def clean_hit(hit: dict[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity", {})
    video_path = entity["video_path"]
    return {
        "id": hit["id"],
        "score": hit["distance"],
        "clip_id": strip_video_id(video_path),
        "video_path": video_path,
        "chunk": entity["chunk"],
    }


def query_name_from_word(word: str) -> str:
    return "_".join(word.strip().lower().split())


def strip_video_id(path: str) -> str:
    return Path(path).name.split(".")[0].lower()


def served_video_url(query_name: str, clip_id: str) -> str:
    return f"{request.host_url.rstrip('/')}/video/{query_name}/{clip_id}.mp4"


def video_query_word(filename: str) -> str:
    name = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return f"video query {name}" if name else "video query"


def save_uploaded_video() -> tuple[Path, str]:
    VIDEO_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    payload = request.get_json(silent=True) or {}
    video_path = payload.get("video_path") or request.form.get("video_path")
    if video_path:
        path = Path(video_path)
        assert path.exists()
        return path, video_query_word(path.name)

    video = request.files["video"]
    suffix = Path(video.filename).suffix or ".mp4"
    path = VIDEO_QUERY_DIR / f"{uuid.uuid4().hex}{suffix}"
    video.save(path)
    return path, video_query_word(video.filename)


def load_video(path: Path) -> np.ndarray:
    reader = decord.VideoReader(str(path), width=DECODE_RESOLUTION, height=DECODE_RESOLUTION)
    assert len(reader) > 0
    frame_ids = np.linspace(0, len(reader) - 1, NUM_FRAMES, dtype=int).tolist()
    frames = reader.get_batch(frame_ids).asnumpy()
    return np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))


def video_codec(path: Path) -> str:
    return subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip()


def transcode_video(path: Path) -> None:
    tmp_path = path.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            VIDEO_ENCODER,
            "-preset",
            "p4",
            "-cq",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ],
        check=True,
    )
    tmp_path.replace(path)


def ensure_browser_video(path: Path) -> Path:
    if path in BROWSER_VIDEOS:
        return path

    if video_codec(path) == VIDEO_CODEC:
        BROWSER_VIDEOS.add(path)
        return path

    lock = VIDEO_LOCKS.setdefault(path, threading.Lock())
    with lock:
        if video_codec(path) != VIDEO_CODEC:
            transcode_video(path)
        BROWSER_VIDEOS.add(path)
    return path


def search_results(word: str, quantity: int) -> list[dict[str, Any]]:
    return search_vector(embed_text(word), quantity)


def search_vector(embedding: list[float], quantity: int) -> list[dict[str, Any]]:
    limit = min(quantity + len(REMOVED_IDS), MAX_QUANTITY)
    hits = CLIENT.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        anns_field=VECTOR_FIELD,
        limit=limit,
        output_fields=OUTPUT_FIELDS,
    )[0]
    rows = [clean_hit(hit) for hit in hits if int(hit["id"]) not in REMOVED_IDS]
    return rows[:quantity]


def embed_video(path: Path) -> list[float]:
    inputs = PROCESSOR(videos=load_video(path)).to(DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        output = MODEL.get_video_embeddings(**inputs)
    return output.visual_proj.float().cpu().numpy()[0].tolist()


def query_history() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    folders = [path for path in DATA_DIR.iterdir() if path.is_dir()]
    rows = []
    for folder in sorted(folders, key=lambda path: path.stat().st_mtime, reverse=True):
        rows.append(
            {
                "query_name": folder.name,
                "word": folder.name.replace("_", " "),
                "output_dir": str(folder),
                "paths_json": str(folder / f"{folder.name}{PATHS_SUFFIX}"),
                "video_count": len(list(folder.glob("*.mp4"))),
                "updated_at": folder.stat().st_mtime,
            }
        )
    return rows


def attach_video_urls(word: str, results: list[dict[str, Any]], existing_only: bool) -> list[dict[str, Any]]:
    query_name = query_name_from_word(word)
    rows = []
    for result in results:
        clip_id = strip_video_id(result["video_path"])
        local_path = DATA_DIR / query_name / f"{clip_id}.mp4"
        row = {
            **result,
            "clip_id": clip_id,
            "local_video_path": str(local_path),
            "video_downloaded": local_path.exists(),
        }
        if local_path.exists() or not existing_only:
            row["video_url"] = served_video_url(query_name, clip_id)
        rows.append(row)
    return rows


def local_video_path(clip_id: str) -> Path | None:
    matches = sorted(DATA_DIR.glob(f"*/{clip_id}.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def attach_local_video_urls(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        path = local_video_path(result["clip_id"])
        row = {**result, "video_downloaded": path is not None}
        if path is not None:
            row["local_video_path"] = str(path)
            row["video_url"] = served_video_url(path.parent.name, result["clip_id"])
        rows.append(row)
    return rows


def local_video_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for result in results:
        path = local_video_path(result["clip_id"])
        if path is None:
            continue
        rows.append(
            {
                "clip_id": result["clip_id"],
                "filename": path.name,
                "local_video_path": str(path),
                "video_path": result["video_path"],
                "url": served_video_url(path.parent.name, result["clip_id"]),
            }
        )
    return rows


def parse_overwrite() -> bool:
    payload = request.get_json(silent=True) or {}
    value = payload.get("overwrite") or request.args.get("overwrite") or False
    return str(value).strip().lower() in ("1", "true", "y", "yes")


def parse_result_paths() -> list[str]:
    payload = request.get_json(silent=True) or {}
    results = payload.get("results") or []
    return [result["video_path"] for result in results]


def run_download_job(
    job_id: str,
    word: str,
    overwrite: bool,
    request_path: Path,
    progress_path: Path,
    videos: list[dict[str, str]],
    total: int,
) -> None:
    DOWNLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "state": "running",
        "word": word,
        "done": 0,
        "total": total,
        "written": 0,
        "skipped": 0,
    }

    env = os.environ.copy()
    env["COSMOS_CDS_DOWNLOAD_REQUEST"] = str(request_path)
    env["COSMOS_CDS_DOWNLOAD_PROGRESS"] = str(progress_path)
    result = subprocess.run(
        [DOWNLOAD_PYTHON, str(DOWNLOAD_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    DOWNLOAD_JOBS[job_id] = {"job_id": job_id, "state": "done", "word": word, "done": total, "total": total, "videos": videos, **info}


def start_download_job(word: str, paths: list[str], overwrite: bool) -> dict[str, Any]:
    DOWNLOAD_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    request_path = DOWNLOAD_REQUEST_DIR / f"{job_id}.json"
    progress_path = DOWNLOAD_REQUEST_DIR / f"{job_id}.progress.json"
    with open(request_path, "w") as f:
        json.dump({"word": word, "paths": paths, "overwrite": overwrite}, f)

    videos = get_video_rows(word, paths)
    DOWNLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "state": "queued",
        "word": word,
        "done": 0,
        "total": len(paths),
        "written": 0,
        "skipped": 0,
    }
    thread = threading.Thread(
        target=run_download_job,
        args=(job_id, word, overwrite, request_path, progress_path, videos, len(paths)),
        daemon=True,
    )
    thread.start()
    return DOWNLOAD_JOBS[job_id]


def get_download_job(job_id: str) -> dict[str, Any]:
    job = DOWNLOAD_JOBS[job_id]
    progress_path = DOWNLOAD_REQUEST_DIR / f"{job_id}.progress.json"
    if progress_path.exists() and job["state"] != "done":
        with open(progress_path) as f:
            progress = json.load(f)
        job = {**job, **progress}
        DOWNLOAD_JOBS[job_id] = job
    return job


def get_video_rows(word: str, paths: list[str], existing_only: bool = False) -> list[dict[str, str]]:
    query_name = query_name_from_word(word)
    seen = set()
    rows = []
    for path in paths:
        clip_id = strip_video_id(path)
        if clip_id in seen:
            continue
        seen.add(clip_id)
        filename = f"{clip_id}.mp4"
        video_path = DATA_DIR / query_name / filename
        if existing_only and not video_path.exists():
            continue
        rows.append(
            {
                "clip_id": clip_id,
                "filename": filename,
                "local_video_path": str(video_path),
                "video_path": path,
                "url": served_video_url(query_name, clip_id),
            }
        )
    return rows


APP = Flask(__name__)
seed_everything()
PROCESSOR, MODEL = load_processor_model()
CLIENT = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
CLIENT.load_collection(COLLECTION_NAME)


@APP.after_request
def add_cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Range"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Expose-Headers"] = "Accept-Ranges, Content-Length, Content-Range"
    return response


@APP.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"ok": True, "collection": COLLECTION_NAME, "device": DEVICE})


@APP.route("/", methods=["GET"])
def root() -> Response:
    return redirect("/viewer/")


@APP.route("/viewer/", methods=["GET"])
def viewer() -> Response:
    return send_file(WEB_VIEWER_DIR / "index.html")


@APP.route("/viewer/<path:filename>", methods=["GET"])
def viewer_asset(filename: str) -> Response:
    path = WEB_VIEWER_DIR / filename
    assert path.exists()
    return send_file(path)


@APP.route("/history", methods=["GET"])
def history() -> Response:
    return jsonify({"data_dir": str(DATA_DIR), "queries": query_history()})


@APP.route("/search", methods=["GET", "POST", "OPTIONS"])
def search() -> Response:
    if request.method == "OPTIONS":
        return jsonify({})

    word, quantity = parse_request()
    results = search_results(word, quantity)
    results = attach_video_urls(word, results, existing_only=True)
    paths = [result["video_path"] for result in results]
    return jsonify(
        {
            "collection": COLLECTION_NAME,
            "word": word,
            "quantity": quantity,
            "ids": [result["id"] for result in results],
            "results": results,
            "videos": get_video_rows(word, paths, existing_only=True),
        }
    )


@APP.route("/search_video", methods=["POST", "OPTIONS"])
def search_video() -> Response:
    if request.method == "OPTIONS":
        return jsonify({})

    quantity = parse_quantity()
    video_path, word = save_uploaded_video()
    results = attach_local_video_urls(search_vector(embed_video(video_path), quantity))
    return jsonify(
        {
            "collection": COLLECTION_NAME,
            "mode": "video",
            "word": word,
            "video_query_path": str(video_path),
            "quantity": quantity,
            "ids": [result["id"] for result in results],
            "clip_ids": [result["clip_id"] for result in results],
            "results": results,
            "videos": local_video_rows(results),
        }
    )


@APP.route("/download", methods=["GET", "POST", "OPTIONS"])
def download() -> Response:
    if request.method == "OPTIONS":
        return jsonify({})

    word, quantity = parse_request()
    paths = parse_result_paths()
    if not paths:
        paths = [result["video_path"] for result in search_results(word, quantity)]
    job = start_download_job(word, paths, overwrite=parse_overwrite())
    return jsonify({"quantity": quantity, **job}), 202


@APP.route("/download/<job_id>", methods=["GET"])
def download_status(job_id: str) -> Response:
    assert job_id in DOWNLOAD_JOBS
    return jsonify(get_download_job(job_id))


@APP.route("/video/<query_name>/<filename>", methods=["GET"])
def video(query_name: str, filename: str) -> Response:
    assert query_name and "/" not in query_name
    assert filename.endswith(".mp4") and "/" not in filename
    path = DATA_DIR / query_name / filename
    assert path.exists(), f"Missing {path}"
    return send_file(ensure_browser_video(path), mimetype="video/mp4", conditional=True)


if __name__ == "__main__":
    APP.run(host=HOST, port=PORT)
