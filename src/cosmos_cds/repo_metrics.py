import ast
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from cosmos_cds.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
CODE_SUFFIXES = {".py", ".js", ".css", ".html", ".sh", ".yaml", ".md"}
PY_COMPLEXITY_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.BoolOp,
    ast.IfExp,
    ast.Match,
)
TOP_FILES = 10

console = Console()


def repo_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if (ROOT / line).exists()]


def code_files(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.suffix in CODE_SUFFIXES]


def line_count(path: Path) -> int:
    return len(path.read_text().splitlines())


def py_complexity(path: Path) -> int:
    tree = ast.parse(path.read_text())
    return 1 + sum(isinstance(node, PY_COMPLEXITY_NODES) for node in ast.walk(tree))


def file_rows(paths: list[Path]) -> list[tuple[Path, int, int]]:
    rows = []
    for path in paths:
        complexity = py_complexity(path) if path.suffix == ".py" else 0
        rows.append((path, line_count(path), complexity))
    return sorted(rows, key=lambda row: row[1], reverse=True)


def print_summary(paths: list[Path], rows: list[tuple[Path, int, int]]) -> None:
    table = Table(title="Repo Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("repo_files", f"{len(paths):,}")
    table.add_row("code_files", f"{len(rows):,}")
    table.add_row("code_lines", f"{sum(row[1] for row in rows):,}")
    table.add_row("python_complexity", f"{sum(row[2] for row in rows):,}")
    console.print(table)


def print_largest(rows: list[tuple[Path, int, int]]) -> None:
    table = Table(title=f"Top {TOP_FILES} Files")
    table.add_column("File")
    table.add_column("Lines", justify="right")
    table.add_column("Py complexity", justify="right")
    for path, lines, complexity in rows[:TOP_FILES]:
        table.add_row(str(path.relative_to(ROOT)), f"{lines:,}", str(complexity or ""))
    console.print(table)


def main() -> None:
    paths = repo_files()
    rows = file_rows(code_files(paths))
    print_summary(paths, rows)
    print_largest(rows)


if __name__ == "__main__":
    main()
