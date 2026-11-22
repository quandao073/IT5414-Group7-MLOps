"""Data drift with Evidently: the training data against each month of 2017, and City Hotel against Resort Hotel.

Each scenario gets an HTML report and one row in a summary table with the share of drifted columns. `column_drift`
is the reusable core: it takes any reference and current frames, so the drift service can call it on logged inputs.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import typer

from hotel_booking.config import MODEL_INPUT_COLUMNS, RAW_CATEGORICAL_INPUTS, TIME_COLUMN
from hotel_booking.data import read_split

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from evidently import DataDefinition, Dataset, Report
    from evidently.core.report import Snapshot
    from evidently.presets import DataDriftPreset

# Calendar columns drift by construction when the reference is a different period, so they are not compared.
CALENDAR_COLUMNS = ["arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"]
# Agent and company are identifiers: compared as categories, with missing values as their own category.
ID_LIKE_COLUMNS = ["agent", "company"]
DATASET_DRIFT_SHARE = 0.5
HOTEL_SCENARIO = "city_to_resort"


def drift_columns(exclude: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Columns compared for drift: the model inputs without the calendar columns and `exclude`.

    Args:
        exclude: Extra columns to leave out, e.g. the column that defines the scenario.

    Returns:
        The numerical and the categorical columns.
    """
    skip = set(CALENDAR_COLUMNS) | set(exclude)
    categorical = [c for c in RAW_CATEGORICAL_INPUTS + ID_LIKE_COLUMNS if c not in skip]
    numerical = [c for c in MODEL_INPUT_COLUMNS if c not in skip and c not in categorical]
    return numerical, categorical


def _prepare(frame: pd.DataFrame, numerical: list[str], categorical: list[str]) -> pd.DataFrame:
    """Keep the compared columns, with numbers as floats and categories as strings.

    Args:
        frame: Bookings with the model input columns.
        numerical: Numerical columns to keep.
        categorical: Categorical columns to keep.

    Returns:
        A frame Evidently can read.
    """
    out = frame[numerical].astype(float)
    for column in categorical:
        values = frame[column]
        if column in ID_LIKE_COLUMNS:
            values = values.map(lambda v: "none" if pd.isna(v) else str(int(v)))
        out[column] = values.astype(str)
    return out


def column_drift(
    reference: pd.DataFrame, current: pd.DataFrame, exclude: tuple[str, ...] = ()
) -> tuple[Snapshot, pd.DataFrame]:
    """Run Evidently's data drift preset and return the snapshot plus one row per compared column.

    Evidently picks the test per column from the column type and sample size (K-S or chi-squared on small samples,
    Wasserstein or Jensen-Shannon distance above 1000 rows). A p-value drifts below its threshold, a distance at or
    above it.

    Args:
        reference: Reference bookings, e.g. the training split.
        current: Bookings to compare with the reference.
        exclude: Extra columns to leave out.

    Returns:
        The Evidently snapshot and a table with `column`, `method`, `threshold`, `score` and `drifted`.
    """
    numerical, categorical = drift_columns(exclude)
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    datasets = [
        Dataset.from_pandas(_prepare(frame, numerical, categorical), data_definition=definition)
        for frame in (current, reference)
    ]
    snapshot = Report([DataDriftPreset(drift_share=DATASET_DRIFT_SHARE)]).run(*datasets)
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if not config["type"].endswith("ValueDrift"):
            continue
        method, threshold, score = config["method"], config["threshold"], float(metric["value"])
        drifted = score < threshold if "p_value" in method else score >= threshold
        rows.append(
            {"column": config["column"], "method": method, "threshold": threshold, "score": score, "drifted": drifted}
        )
    return snapshot, pd.DataFrame(rows, columns=["column", "method", "threshold", "score", "drifted"])


Scenario = tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]


def scenarios(processed_dir: Path) -> dict[str, Scenario]:
    """Build the drift scenarios from the processed splits.

    One scenario per arrival month of 2017 (`YYYY-MM`, from test and production holdout) against the whole training
    split, plus `city_to_resort`: City Hotel against Resort Hotel bookings of the training split, without the `hotel`
    column itself.

    Args:
        processed_dir: Folder written by `data.preprocess`.

    Returns:
        Reference, current and excluded columns for each scenario name.
    """
    train = read_split(processed_dir, "train")
    stream = pd.concat([read_split(processed_dir, name) for name in ("test", "production_holdout")])
    months = stream[TIME_COLUMN].dt.strftime("%Y-%m")
    result: dict[str, Scenario] = {month: (train, stream[months == month], ()) for month in sorted(months.unique())}
    city, resort = train[train["hotel"] == "City Hotel"], train[train["hotel"] == "Resort Hotel"]
    result[HOTEL_SCENARIO] = (city, resort, ("hotel",))
    return result


def main(processed_dir: Path, output_dir: Path) -> None:
    """Write an HTML report per scenario, a summary table and the per-column drift table.

    Outputs: `html/<scenario>.html`, `summary.csv` and `summary.json` (one row per scenario with the share of
    drifted columns) and `columns.csv` (one row per scenario and column).

    Args:
        processed_dir: Folder written by `data.preprocess`.
        output_dir: Folder for the reports.
    """
    (output_dir / "html").mkdir(parents=True, exist_ok=True)
    summary, tables = [], []
    for name, (reference, current, exclude) in scenarios(processed_dir).items():
        snapshot, table = column_drift(reference, current, exclude)
        snapshot.save_html(str(output_dir / "html" / f"{name}.html"))
        share = float(table["drifted"].mean())
        drifted = sorted(table.loc[table["drifted"], "column"])
        summary.append(
            {
                "scenario": name,
                "reference_rows": len(reference),
                "current_rows": len(current),
                "columns": len(table),
                "drifted_columns": len(drifted),
                "drift_share": round(share, 4),
                "dataset_drift": share >= DATASET_DRIFT_SHARE,
                "drifted": ", ".join(drifted),
            }
        )
        tables.append(table.assign(scenario=name))
        print(f"{name}: {len(drifted)}/{len(table)} columns drifted")
    frame = pd.DataFrame(summary)
    frame.to_csv(output_dir / "summary.csv", index=False)
    by_scenario = {row.pop("scenario"): row for row in frame.to_dict(orient="records")}
    (output_dir / "summary.json").write_text(f"{json.dumps(by_scenario, indent=2)}\n", encoding="utf-8")
    pd.concat(tables)[["scenario", "column", "method", "threshold", "score", "drifted"]].to_csv(
        output_dir / "columns.csv", index=False
    )


if __name__ == "__main__":
    typer.run(main)
