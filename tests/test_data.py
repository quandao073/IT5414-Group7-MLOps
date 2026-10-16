from pathlib import Path

from hotel_booking.data import preprocess


def test_preprocess_runs(tmp_path: Path) -> None:
    """Test that preprocess accepts input and output folders without error."""
    preprocess(tmp_path, tmp_path / "processed")
