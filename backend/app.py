import os
import random
import shutil
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
BASE_OUTPUT_FIELDS = ["video_path", "chunk"]
METADATA_FIELDS = ["country", "month", "hour_of_day", "platform_class", "radar_config"]
FILTER_FIELDS = ["clip_id", *METADATA_FIELDS]
OUTPUT_FIELDS = BASE_OUTPUT_FIELDS
DEFAULT_QUANTITY = 10
MAX_QUANTITY = 1000
REMOVED_IDS = {467240255860630002}
APPLY_METADATA_FILTER = True
METADATA_OPERATORS = {"in", "==", "!=", ">", ">=", "<", "<="}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
LOCAL_FILES_ONLY = True
BACKEND_DIR = Path(__file__).resolve().parent
DOWNLOAD_PYTHON = str(BACKEND_DIR / ".venv" / "bin" / "python")
DOWNLOAD_SCRIPT = BACKEND_DIR / "download_video_list.py"
DOWNLOAD_REQUEST_DIR = Path("/tmp/cosmos_cds_download_requests")
WEB_VIEWER_DIR = Path(__file__).resolve().parent.parent / "web_viewer"
DATA_DIR = Path("/data0/sebastian.cavada/datasets/cosmos-cds/data")
CLIP_DIR = DATA_DIR / "clips"
VIDEO_QUERY_DIR = Path("/tmp/cosmos_cds_video_queries")
PATHS_SUFFIX = "_paths.json"
NUM_FRAMES = 8
DECODE_RESOLUTION = 448
DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
ACTIVE_METADATA_FIELDS: list[str] = []
ACTIVE_FILTER_FIELDS: set[str] = set()


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


def parse_quantity(value: Any = DEFAULT_QUANTITY) -> int:
    quantity = int(value)
    assert 0 < quantity <= MAX_QUANTITY
    return quantity


def clean_metadata_filter(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None

    metadata_filter = json.loads(value) if isinstance(value, str) else value
    assert isinstance(metadata_filter, dict)
    assert isinstance(metadata_filter["field"], str) and metadata_filter["field"].replace("_", "").replace(".", "").isalnum()
    assert metadata_filter["field"] in FILTER_FIELDS
    assert metadata_filter["operator"] in METADATA_OPERATORS
    assert "value" in metadata_filter
    return metadata_filter


def milvus_filter_expr(metadata_filter: dict[str, Any]) -> str:
    field = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]
    if operator == "in":
        assert isinstance(value, list)
        return f"{field} in {json.dumps(value)}"
    return f"{field} {operator} {json.dumps(value)}"


def search_request() -> tuple[str, int, dict[str, Any] | None]:
    word = request.args["word"].strip()
    assert word
    return word, parse_quantity(request.args.get("quantity", DEFAULT_QUANTITY)), clean_metadata_filter(request.args.get("metadata_filter"))


def clean_hit(hit: dict[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity", {})
    video_path = entity["video_path"]
    metadata = {field: entity[field] for field in ACTIVE_METADATA_FIELDS if field in entity}
    row = {
        "id": str(hit["id"]),
        "score": hit["distance"],
        "clip_id": strip_video_id(video_path),
        "video_path": video_path,
        "chunk": entity["chunk"],
    }
    if metadata:
        row["metadata"] = metadata
    return row


def query_name_from_word(word: str) -> str:
    return "_".join(word.strip().lower().split())


def strip_video_id(path: str) -> str:
    return Path(path).name.split(".")[0].lower()


def served_video_url(query_name: str, clip_id: str) -> str:
    return f"{request.host_url.rstrip('/')}/video/{query_name}/{clip_id}.mp4"


def clip_path(clip_id: str) -> Path:
    return CLIP_DIR / f"{clip_id}.mp4"


def video_query_word(filename: str) -> str:
    name = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return f"video query {name}" if name else "video query"


def save_uploaded_video() -> tuple[Path, str]:
    VIDEO_QUERY_DIR.mkdir(parents=True, exist_ok=True)
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


def search_results(word: str, quantity: int, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return search_vector(embed_text(word), quantity, metadata_filter)


def search_vector(embedding: list[float], quantity: int, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    limit = min(quantity + len(REMOVED_IDS), MAX_QUANTITY)
    search_kwargs = {}
    if is_metadata_filter_applied(metadata_filter):
        search_kwargs["filter"] = milvus_filter_expr(metadata_filter)
    hits = CLIENT.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        anns_field=VECTOR_FIELD,
        limit=limit,
        output_fields=OUTPUT_FIELDS,
        **search_kwargs,
    )[0]
    rows = [clean_hit(hit) for hit in hits if int(hit["id"]) not in REMOVED_IDS]
    return rows[:quantity]


def is_metadata_filter_applied(metadata_filter: dict[str, Any] | None) -> bool:
    return APPLY_METADATA_FILTER and metadata_filter is not None and metadata_filter["field"] in ACTIVE_FILTER_FIELDS


def metadata_status() -> dict[str, Any]:
    return {
        "configured_metadata_fields": METADATA_FIELDS,
        "configured_filter_fields": FILTER_FIELDS,
        "active_metadata_fields": ACTIVE_METADATA_FIELDS,
        "active_filter_fields": sorted(ACTIVE_FILTER_FIELDS),
        "metadata_available": bool(ACTIVE_METADATA_FIELDS),
        "metadata_filter_enabled": APPLY_METADATA_FILTER,
    }


def configure_collection_fields(client: MilvusClient) -> None:
    global ACTIVE_FILTER_FIELDS, ACTIVE_METADATA_FIELDS, OUTPUT_FIELDS
    fields = {field["name"] for field in client.describe_collection(COLLECTION_NAME)["fields"]}
    ACTIVE_METADATA_FIELDS = [field for field in METADATA_FIELDS if field in fields]
    ACTIVE_FILTER_FIELDS = {field for field in FILTER_FIELDS if field in fields}
    OUTPUT_FIELDS = [*BASE_OUTPUT_FIELDS, *ACTIVE_METADATA_FIELDS]


def embed_video(path: Path) -> list[float]:
    inputs = PROCESSOR(videos=load_video(path)).to(DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        output = MODEL.get_video_embeddings(**inputs)
    return output.visual_proj.float().cpu().numpy()[0].tolist()


def query_history() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    folders = [path for path in DATA_DIR.iterdir() if path.is_dir() and path != CLIP_DIR]
    rows = []
    for folder in sorted(folders, key=lambda path: path.stat().st_mtime, reverse=True):
        paths_json = folder / f"{folder.name}{PATHS_SUFFIX}"
        rows.append(
            {
                "query_name": folder.name,
                "word": folder.name.replace("_", " "),
                "output_dir": str(folder),
                "paths_json": str(paths_json),
                "video_count": query_video_count(folder, paths_json),
                "updated_at": paths_json.stat().st_mtime if paths_json.exists() else folder.stat().st_mtime,
            }
        )
    return rows


def query_video_count(folder: Path, paths_json: Path) -> int:
    if not paths_json.exists():
        return len(list(folder.glob("*.mp4")))
    with open(paths_json) as f:
        clip_ids = {strip_video_id(path) for path in json.load(f)}
    return sum((clip_path(clip_id).exists() or (folder / f"{clip_id}.mp4").exists()) for clip_id in clip_ids)


def referenced_clip_ids() -> set[str]:
    ids = set()
    for folder in DATA_DIR.iterdir():
        paths_json = folder / f"{folder.name}{PATHS_SUFFIX}"
        if not folder.is_dir() or folder == CLIP_DIR or not paths_json.exists():
            continue
        with open(paths_json) as f:
            ids.update(strip_video_id(path) for path in json.load(f))
    return ids


def prune_unused_clips() -> None:
    if not CLIP_DIR.exists():
        return
    used = referenced_clip_ids()
    for path in CLIP_DIR.glob("*.mp4"):
        if path.stem not in used:
            path.unlink()


def delete_query(query_name: str) -> None:
    assert query_name and "/" not in query_name
    path = DATA_DIR / query_name
    assert path.is_dir()
    shutil.rmtree(path)
    prune_unused_clips()


def attach_video_urls(word: str, results: list[dict[str, Any]], existing_only: bool) -> list[dict[str, Any]]:
    query_name = query_name_from_word(word)
    rows = []
    for result in results:
        clip_id = strip_video_id(result["video_path"])
        local_path = clip_path(clip_id)
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
    path = clip_path(clip_id)
    if path.exists():
        return path
    matches = sorted(DATA_DIR.glob(f"*/{clip_id}.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
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


def parse_overwrite(payload: dict[str, Any]) -> bool:
    value = payload.get("overwrite", False)
    return str(value).strip().lower() in ("1", "true", "y", "yes")


def result_paths(payload: dict[str, Any]) -> list[str]:
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
        video_path = clip_path(clip_id)
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


APP = Flask(__name__, static_folder=str(WEB_VIEWER_DIR), static_url_path="/viewer")
seed_everything()
PROCESSOR, MODEL = load_processor_model()
CLIENT = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
CLIENT.load_collection(COLLECTION_NAME)
configure_collection_fields(CLIENT)


@APP.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"ok": True, "collection": COLLECTION_NAME, "device": DEVICE, **metadata_status()})


@APP.route("/metadata", methods=["GET"])
def metadata() -> Response:
    return jsonify(metadata_status())


@APP.route("/", methods=["GET"])
def root() -> Response:
    return redirect("/viewer/")


@APP.route("/viewer/", methods=["GET"])
def viewer() -> Response:
    return send_file(WEB_VIEWER_DIR / "index.html")


@APP.route("/history", methods=["GET"])
def history() -> Response:
    return jsonify({"data_dir": str(DATA_DIR), "queries": query_history()})


@APP.route("/history/<query_name>", methods=["DELETE"])
def history_delete(query_name: str) -> Response:
    delete_query(query_name)
    return jsonify({"deleted": query_name, "queries": query_history()})


@APP.route("/search", methods=["GET"])
def search() -> Response:
    word, quantity, metadata_filter = search_request()
    results = search_results(word, quantity, metadata_filter)
    results = attach_video_urls(word, results, existing_only=True)
    paths = [result["video_path"] for result in results]
    return jsonify(
        {
            "collection": COLLECTION_NAME,
            "word": word,
            "quantity": quantity,
            "metadata_filter": metadata_filter,
            "metadata_filter_applied": is_metadata_filter_applied(metadata_filter),
            **metadata_status(),
            "ids": [result["id"] for result in results],
            "results": results,
            "videos": get_video_rows(word, paths, existing_only=True),
        }
    )


@APP.route("/search_video", methods=["POST"])
def search_video() -> Response:
    quantity = parse_quantity(request.form.get("quantity", DEFAULT_QUANTITY))
    metadata_filter = clean_metadata_filter(request.form.get("metadata_filter"))
    video_path, word = save_uploaded_video()
    results = attach_local_video_urls(search_vector(embed_video(video_path), quantity, metadata_filter))
    return jsonify(
        {
            "collection": COLLECTION_NAME,
            "mode": "video",
            "word": word,
            "video_query_path": str(video_path),
            "quantity": quantity,
            "metadata_filter": metadata_filter,
            "metadata_filter_applied": is_metadata_filter_applied(metadata_filter),
            **metadata_status(),
            "ids": [result["id"] for result in results],
            "clip_ids": [result["clip_id"] for result in results],
            "results": results,
            "videos": local_video_rows(results),
        }
    )


@APP.route("/download", methods=["POST"])
def download() -> Response:
    payload = request.get_json()
    word = payload["word"].strip()
    quantity = parse_quantity(payload.get("quantity", DEFAULT_QUANTITY))
    metadata_filter = clean_metadata_filter(payload.get("metadata_filter"))
    paths = result_paths(payload)
    if not paths:
        paths = [result["video_path"] for result in search_results(word, quantity, metadata_filter)]
    job = start_download_job(word, paths, overwrite=parse_overwrite(payload))
    return jsonify({"quantity": quantity, "metadata_filter": metadata_filter, **job}), 202


@APP.route("/download/<job_id>", methods=["GET"])
def download_status(job_id: str) -> Response:
    assert job_id in DOWNLOAD_JOBS
    return jsonify(get_download_job(job_id))


@APP.route("/video/<query_name>/<filename>", methods=["GET"])
def video(query_name: str, filename: str) -> Response:
    assert query_name and "/" not in query_name
    assert filename.endswith(".mp4") and "/" not in filename
    path = CLIP_DIR / filename
    if not path.exists():
        path = DATA_DIR / query_name / filename
    assert path.exists(), f"Missing {path}"
    return send_file(path, mimetype="video/mp4", conditional=True)


if __name__ == "__main__":
    APP.run(host=HOST, port=PORT)
