from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from cosmos_cds.paths import DATA_DIR as PROJECT_DATA_DIR

DATA_DIR = PROJECT_DATA_DIR / "buffered_pipeline"
INPUT_PATHS = [
    DATA_DIR / "embeddings.448.parquet",
    DATA_DIR / "embeddings_zips_1000_3145.parquet",
]
OUTPUT_PATH = DATA_DIR / "embeddings_0000_3145.448.parquet"
BATCH_SIZE = 8192

console = Console()


def prepare_output() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    assert all(path.exists() for path in INPUT_PATHS)

    if not OUTPUT_PATH.exists():
        return

    answer = input(f"{OUTPUT_PATH} exists. Type OVERWRITE to reset it: ")
    assert answer == "OVERWRITE"
    OUTPUT_PATH.unlink()


def print_report(input_rows: dict[Path, int], output_rows: int) -> None:
    table = Table(title="448p Embedding Merge")
    table.add_column("File")
    table.add_column("Rows", justify="right")

    for path, rows in input_rows.items():
        table.add_row(path.name, f"{rows:,}")
    table.add_row(OUTPUT_PATH.name, f"{output_rows:,}")

    console.print(table)


def main() -> None:
    prepare_output()

    input_files = [pq.ParquetFile(path) for path in INPUT_PATHS]
    schema = input_files[0].schema_arrow
    assert all(file.schema_arrow == schema for file in input_files)

    input_rows = {path: file.metadata.num_rows for path, file in zip(INPUT_PATHS, input_files, strict=True)}
    expected_rows = sum(input_rows.values())

    with pq.ParquetWriter(OUTPUT_PATH, schema) as writer:
        for path, file in zip(INPUT_PATHS, input_files, strict=True):
            progress = tqdm(total=file.metadata.num_rows, desc=path.name, unit="row")
            for batch in file.iter_batches(batch_size=BATCH_SIZE):
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
                progress.update(batch.num_rows)
            progress.close()

    output_rows = pq.ParquetFile(OUTPUT_PATH).metadata.num_rows
    assert output_rows == expected_rows
    print_report(input_rows, output_rows)


if __name__ == "__main__":
    main()
