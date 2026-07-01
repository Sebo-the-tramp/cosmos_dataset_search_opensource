from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pymilvus import DataType, MilvusClient
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

URI = "http://127.0.0.1:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "cosmos_cds_test_00"
PARQUET_PATH = Path(
    "/home/cavadalab/Documents/scsv/covision/cosmos_cds/buffered_pipeline/data/embeddings_0000_3145.448.parquet"
)
METADATA_PARQUET_PATH = Path("/home/cavadalab/Documents/scsv/covision/cosmos_cds/database/data_collection.parquet")
BATCH_SIZE = 4096
VECTOR_FIELD = "embedding"
VECTOR_DIM = 768
METRIC_TYPE = "COSINE"
CLIP_ID_MAX_LENGTH = 64
VIDEO_PATH_MAX_LENGTH = 512
CHUNK_MAX_LENGTH = 128
VARCHAR_METADATA_FIELDS = ["country", "platform_class", "radar_config"]
INT_METADATA_FIELDS = ["month", "hour_of_day"]
METADATA_FIELDS = [*VARCHAR_METADATA_FIELDS, *INT_METADATA_FIELDS]
EMPTY_STRING = ""

console = Console()


def reset_collections(client: MilvusClient) -> list[str]:
    collections = client.list_collections()
    if not collections:
        return []

    answer = input(f"Drop all Milvus collections {collections}? Type DELETE: ")
    assert answer == "DELETE"
    for collection in collections:
        client.drop_collection(collection)
    return collections


def create_collection(client: MilvusClient) -> None:
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="clip_id", datatype=DataType.VARCHAR, max_length=CLIP_ID_MAX_LENGTH)
    schema.add_field(field_name="video_path", datatype=DataType.VARCHAR, max_length=VIDEO_PATH_MAX_LENGTH)
    schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=CHUNK_MAX_LENGTH)
    schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    for field in VARCHAR_METADATA_FIELDS:
        schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=64)
    for field in INT_METADATA_FIELDS:
        schema.add_field(field_name=field, datatype=DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type=METRIC_TYPE)
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def clip_id_from_video_path(video_path: str) -> str:
    return Path(video_path).name.split(".")[0].lower()


def clean_metadata(field: str, value: object) -> object:
    if field in INT_METADATA_FIELDS:
        assert value is not None
        return int(value)
    return EMPTY_STRING if value is None else str(value)


def load_metadata() -> dict[str, dict[str, object]]:
    assert METADATA_PARQUET_PATH.exists()
    table = pq.read_table(METADATA_PARQUET_PATH, columns=["clip_id", *METADATA_FIELDS])
    data = table.to_pydict()
    metadata = {}
    for values in zip(*(data[field] for field in ["clip_id", *METADATA_FIELDS]), strict=True):
        clip_id = str(values[0]).lower()
        assert clip_id not in metadata
        metadata[clip_id] = {field: clean_metadata(field, value) for field, value in zip(METADATA_FIELDS, values[1:], strict=True)}
    assert len(metadata) == table.num_rows
    return metadata


def rows_from_batch(batch: pa.RecordBatch, metadata: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    data = batch.to_pydict()
    rows = []
    for video_path, chunk, embedding in zip(data["video_path"], data["chunk"], data["embedding"], strict=True):
        clip_id = clip_id_from_video_path(video_path)
        rows.append({"clip_id": clip_id, "video_path": video_path, "chunk": chunk, "embedding": embedding, **metadata[clip_id]})
    return rows


def print_report(dropped: list[str], inserted: int, row_count: int, metadata_rows: int) -> None:
    table = Table(title="Milvus Full Ingest")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("dropped_collections", ", ".join(dropped) if dropped else "none")
    table.add_row("collection", COLLECTION_NAME)
    table.add_row("parquet", str(PARQUET_PATH))
    table.add_row("metadata_parquet", str(METADATA_PARQUET_PATH))
    table.add_row("metadata_rows", f"{metadata_rows:,}")
    table.add_row("inserted", f"{inserted:,}")
    table.add_row("row_count", f"{row_count:,}")
    console.print(table)


def main() -> None:
    assert PARQUET_PATH.exists()
    metadata = load_metadata()

    client = MilvusClient(uri=URI, token=TOKEN)
    dropped = reset_collections(client)
    create_collection(client)

    parquet_file = pq.ParquetFile(PARQUET_PATH)
    total = parquet_file.metadata.num_rows
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    inserted = 0

    for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE), total=batches, desc="ingest", unit="batch"):
        rows = rows_from_batch(batch, metadata)
        client.insert(collection_name=COLLECTION_NAME, data=rows)
        inserted += len(rows)

    assert inserted == total
    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    row_count = int(client.get_collection_stats(COLLECTION_NAME)["row_count"])
    assert row_count == inserted
    print_report(dropped, inserted, row_count, len(metadata))


if __name__ == "__main__":
    main()
