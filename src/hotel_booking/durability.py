"""Model durability: ROC-AUC over time and across hotels.

The monthly table scores one model fitted on the training period against every 2017 month. The transfer table fits
the model on one hotel and scores it on the other, next to the same model's in-domain score.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hotel_booking.config import MODEL_INPUT_COLUMNS, RANDOM_STATE, TARGET, TIME_COLUMN
from hotel_booking.data import read_split
from hotel_booking.evaluate import binary_metrics
from hotel_booking.model import ModelSpec, build_from_spec

MONTH_FORMAT = "%Y-%m"
BOOTSTRAP_ROUNDS = 500
CONFIDENCE = 95.0
HOTELS = ("City Hotel", "Resort Hotel")
HOTEL_COLUMN = "hotel"
METRIC_COLUMNS = ["month", "rows", "cancel_rate", "roc_auc", "ci_low", "ci_high", "average_precision"]
TRANSFER_COLUMNS = [
    "source_hotel",
    "target_hotel",
    "rows",
    "cancel_rate",
    "roc_auc",
    "average_precision",
    "in_domain",
    "transfer_gap",
]
CSV_COLUMNS = [
    "section",
    "scenario",
    "rows",
    "cancel_rate",
    "roc_auc",
    "ci_low",
    "ci_high",
    "average_precision",
    "in_domain",
    "transfer_gap",
]


def bootstrap_roc_auc(
    y_true: Any, proba: Any, rounds: int = BOOTSTRAP_ROUNDS, seed: int = RANDOM_STATE
) -> tuple[float, float, float]:
    """ROC-AUC with a percentile bootstrap interval.

    Args:
        y_true: True labels.
        proba: Predicted cancellation probabilities.
        rounds: Number of bootstrap resamples. Zero returns the point estimate as the whole interval.
        seed: Seed of the resampling generator.

    Returns:
        The point estimate, the lower and the upper bound of the interval.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    score = float(roc_auc_score(y_true, proba))
    if rounds == 0:
        return score, score, score
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(rounds):
        rows = rng.integers(0, len(y_true), len(y_true))
        sample = y_true[rows]
        if len(np.unique(sample)) < 2:
            continue
        samples.append(roc_auc_score(sample, proba[rows]))
    if not samples:
        return score, score, score
    tail = (100.0 - CONFIDENCE) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return score, float(min(low, score)), float(max(high, score))


def monthly_metrics(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Score a fitted model on every calendar month of a frame.

    Months with a single class have no ROC-AUC and are left out.

    Args:
        model: Fitted model with `predict_proba`.
        frame: Labelled rows with the arrival date and the target.

    Returns:
        One row per month, in calendar order.
    """
    months = pd.to_datetime(frame[TIME_COLUMN]).dt.strftime(MONTH_FORMAT)
    rows = []
    for month, group in frame.groupby(months, sort=True):
        y_true = group[TARGET].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        proba = model.predict_proba(group)[:, 1]
        roc_auc, ci_low, ci_high = bootstrap_roc_auc(y_true, proba)
        rows.append(
            {
                "month": month,
                "rows": len(group),
                "cancel_rate": float(y_true.mean()),
                "roc_auc": roc_auc,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "average_precision": float(average_precision_score(y_true, proba)),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def hotel_transfer_metrics(train: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Fit on one hotel and score every hotel, in-domain and transferred.

    A source or target hotel with no rows or a single class is skipped. The gap is the source hotel's in-domain
    ROC-AUC minus the row's ROC-AUC, zero on the diagonal and missing when the diagonal was skipped.

    Args:
        train: Labelled rows used to fit the model.
        test: Labelled rows used to score it.
        spec: Model recipe passed to `build_from_spec`.

    Returns:
        One row per source/target hotel pair.
    """
    rows = []
    diagonal = {}
    for source in HOTELS:
        source_train = train[train[HOTEL_COLUMN] == source]
        if source_train.empty or source_train[TARGET].nunique() < 2:
            continue
        model = build_from_spec(spec)
        model.fit(source_train[MODEL_INPUT_COLUMNS], source_train[TARGET])
        for target in HOTELS:
            target_test = test[test[HOTEL_COLUMN] == target]
            if target_test.empty or target_test[TARGET].nunique() < 2:
                continue
            proba = model.predict_proba(target_test[MODEL_INPUT_COLUMNS])[:, 1]
            metrics = binary_metrics(target_test[TARGET], proba)
            if source == target:
                diagonal[source] = metrics["roc_auc"]
            rows.append(
                {
                    "source_hotel": source,
                    "target_hotel": target,
                    "rows": len(target_test),
                    "cancel_rate": float(target_test[TARGET].mean()),
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "in_domain": source == target,
                }
            )
    table = pd.DataFrame(rows, columns=TRANSFER_COLUMNS[:-1])
    gaps: list[float] = []
    for source, in_domain, roc_auc in zip(table["source_hotel"], table["in_domain"], table["roc_auc"], strict=True):
        if bool(in_domain):
            gaps.append(0.0)
        else:
            gaps.append(float(diagonal.get(str(source), float("nan")) - float(roc_auc)))
    table["transfer_gap"] = gaps
    return table


def render_monthly_figure(table: pd.DataFrame, path: Path) -> Path:
    """Draw ROC-AUC with its interval and the cancel rate for every month.

    Args:
        table: Rows produced by `monthly_metrics`.
        path: PNG file to write; its parent folders are created.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(table))
    months = table["month"].astype(str).tolist()
    auc = table["roc_auc"].to_numpy(dtype=float)
    ci_low = table["ci_low"].to_numpy(dtype=float)
    ci_high = table["ci_high"].to_numpy(dtype=float)
    left.errorbar(positions, auc, yerr=[auc - ci_low, ci_high - auc], marker="o", capsize=4, color="tab:blue")
    left.axhline(float(np.mean(auc)), linestyle="--", color="grey", linewidth=1)
    left.set_xticks(positions, months, rotation=45, ha="right")
    left.set_xlabel("Arrival month")
    left.set_ylabel("ROC-AUC")
    left.set_title("ROC-AUC with 95% interval")
    right.bar(positions, table["cancel_rate"].to_numpy(dtype=float), color="tab:orange")
    right.set_xticks(positions, months, rotation=45, ha="right")
    right.set_xlabel("Arrival month")
    right.set_ylabel("Cancel rate")
    right.set_title("Cancel rate")
    fig.suptitle("Model durability by arrival month (2017)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_transfer_figure(table: pd.DataFrame, path: Path) -> Path:
    """Draw the in-domain and transferred ROC-AUC next to each other, one group per hotel.

    Pairs missing from the table are left blank.

    Args:
        table: Rows produced by `hotel_transfer_metrics`.
        path: PNG file to write; its parent folders are created.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("In-domain", "Transferred")
    positions = np.arange(len(labels))
    width = 0.35
    colors = ("tab:blue", "tab:orange")
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, hotel in enumerate(HOTELS):
        in_domain = table[(table["source_hotel"] == hotel) & (table["target_hotel"] == hotel)]
        transferred = table[(table["source_hotel"] == hotel) & (table["target_hotel"] != hotel)]
        values = [
            float(in_domain["roc_auc"].iloc[0]) if not in_domain.empty else float("nan"),
            float(transferred["roc_auc"].iloc[0]) if not transferred.empty else float("nan"),
        ]
        offset = (index - (len(HOTELS) - 1) / 2) * width
        labelled = False
        for position, value in zip(positions + offset, values, strict=True):
            if not np.isfinite(value):
                continue
            ax.bar(position, value, width, color=colors[index], label=None if labelled else hotel)
            labelled = True
            ax.annotate(f"{value:.3f}", (position, value), textcoords="offset points", xytext=(0, 3), ha="center")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC when the hotel changes")
    ax.legend(title="Source hotel", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(
    processed_dir: Path = Path("data/processed"),
    report_dir: Path = Path("reports/durability"),
    figure_dir: Path = Path("reports/figures"),
    spec_path: Path = Path("configs/model/champion.yaml"),
) -> None:
    """Fit the spec on the training split, then write the monthly and hotel-transfer reports.

    Args:
        processed_dir: Folder that holds `train.csv`, `test.csv` and `production_holdout.csv`.
        report_dir: Folder that receives `metrics.csv` and `summary.json`.
        figure_dir: Folder that receives the two PNG figures.
        spec_path: YAML model recipe fitted on the training split.
    """
    import yaml

    spec_path = Path(spec_path)
    spec: ModelSpec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    train = read_split(Path(processed_dir), "train")
    test = read_split(Path(processed_dir), "test")
    holdout = read_split(Path(processed_dir), "production_holdout")
    model = build_from_spec(spec)
    model.fit(train[MODEL_INPUT_COLUMNS], train[TARGET])
    monthly = monthly_metrics(model, pd.concat([test, holdout], ignore_index=True))
    transfer = hotel_transfer_metrics(train, test, spec)
    report_dir = Path(report_dir)
    figure_dir = Path(figure_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    monthly_long = monthly.assign(section="monthly", scenario=monthly["month"])
    transfer_long = transfer.assign(
        section="hotel_transfer",
        scenario=transfer["source_hotel"] + " -> " + transfer["target_hotel"],
    )
    combined = pd.concat([monthly_long, transfer_long], ignore_index=True).reindex(columns=CSV_COLUMNS)
    combined.to_csv(report_dir / "metrics.csv", index=False)
    summary = {
        "model": spec.get("name") or "Soft-voting ensemble",
        "spec": str(spec_path),
        "monthly": monthly.to_dict(orient="records"),
        "hotel_transfer": transfer.to_dict(orient="records"),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_monthly_figure(monthly, figure_dir / "durability_roc_by_month.png")
    render_transfer_figure(transfer, figure_dir / "durability_hotel_transfer.png")
    print("Monthly ROC-AUC: " + ", ".join(f"{row.month}={row.roc_auc:.3f}" for row in monthly.itertuples()))
    print(
        "Hotel transfer: "
        + ", ".join(f"{row.source_hotel}->{row.target_hotel}={row.roc_auc:.3f}" for row in transfer.itertuples())
    )


def _parse_args() -> argparse.Namespace:
    """Parse the command-line options of the durability report."""
    parser = argparse.ArgumentParser(description="Measure model durability by arrival month and across hotels.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/durability"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--spec-path", type=Path, default=Path("configs/model/champion.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.processed_dir, args.report_dir, args.figure_dir, args.spec_path)
