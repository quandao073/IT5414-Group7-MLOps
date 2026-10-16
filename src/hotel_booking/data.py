from pathlib import Path

import typer


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data and save it to the output folder."""
    print("Preprocessing data...")


if __name__ == "__main__":
    typer.run(preprocess)
