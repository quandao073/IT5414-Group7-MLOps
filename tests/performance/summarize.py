"""Turn Locust CSV stats into a Markdown comparison table."""

import csv
from pathlib import Path

import typer


def read_stats(path: Path) -> dict[str, float | int]:
    """Read the Aggregated row of a Locust stats CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    aggregated = next(
        (row for row in rows if row.get("Type") == "Aggregated" or row.get("Name") == "Aggregated"),
        None,
    )
    if aggregated is None:
        raise ValueError(f"{path} has no Aggregated row")
    requests = int(aggregated["Request Count"])
    failures = int(aggregated["Failure Count"])
    return {
        "requests": requests,
        "failures": failures,
        "error_rate": failures / requests if requests else 0.0,
        "rps": float(aggregated["Requests/s"]),
        "p50_ms": float(aggregated["50%"]),
        "p95_ms": float(aggregated["95%"]),
    }


def to_markdown(rows: dict[str, dict[str, float | int]]) -> str:
    """Render one Markdown table row per backend."""
    lines = [
        "| Backend | Requests | RPS | p50 (ms) | p95 (ms) | Error rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for backend, stats in rows.items():
        lines.append(
            f"| {backend} | {stats['requests']} | {stats['rps']} | {stats['p50_ms']} | {stats['p95_ms']} "
            f"| {stats['error_rate']:.2%} |"
        )
    return "\n".join(lines)


def main(paths: list[Path]) -> None:
    """Print a Markdown table for the given Locust stats CSV files."""
    rows = {path.name.removesuffix("_stats.csv"): read_stats(path) for path in paths}
    print(to_markdown(rows))


if __name__ == "__main__":
    typer.run(main)
