from pathlib import Path

from tests.performance.summarize import read_stats, to_markdown


def test_read_stats_uses_the_aggregated_row(tmp_path: Path) -> None:
    path = tmp_path / "joblib_stats.csv"
    path.write_text(
        "Type,Name,Request Count,Failure Count,Requests/s,50%,95%\n"
        "POST,/predict,999,999,1.0,1.0,1.0\n"
        "Aggregated,,100,2,10.5,12.0,30.0\n",
        encoding="utf-8",
    )
    assert read_stats(path) == {
        "requests": 100,
        "failures": 2,
        "error_rate": 0.02,
        "rps": 10.5,
        "p50_ms": 12.0,
        "p95_ms": 30.0,
    }


def test_read_stats_rejects_a_file_without_the_aggregated_row(tmp_path: Path) -> None:
    path = tmp_path / "empty_stats.csv"
    path.write_text("Type,Name,Request Count,Failure Count,Requests/s,50%,95%\n", encoding="utf-8")
    try:
        read_stats(path)
    except ValueError as error:
        assert "Aggregated" in str(error)
    else:
        raise AssertionError("read_stats accepted a file without an Aggregated row")


def test_to_markdown_formats_each_backend() -> None:
    table = to_markdown(
        {
            "joblib": {
                "requests": 100,
                "failures": 2,
                "error_rate": 0.02,
                "rps": 10.5,
                "p50_ms": 12.0,
                "p95_ms": 30.0,
            }
        }
    )
    assert "| Backend | Requests | RPS | p50 (ms) | p95 (ms) | Error rate |" in table
    assert "| joblib | 100 | 10.50 | 12.0 | 30.0 | 2.00% |" in table


def test_to_markdown_rounds_rps_to_two_decimals() -> None:
    table = to_markdown(
        {
            "onnx": {
                "requests": 936,
                "failures": 0,
                "error_rate": 0.0,
                "rps": 15.78278678310969,
                "p50_ms": 1200.0,
                "p95_ms": 1400.0,
            }
        }
    )
    assert "| onnx | 936 | 15.78 | 1200.0 | 1400.0 | 0.00% |" in table
