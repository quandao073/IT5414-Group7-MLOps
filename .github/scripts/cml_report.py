"""Write the CML pull request comment: test metrics of the PR model next to the base branch model.

Usage: `python cml_report.py <pr model_card.json> <base model_card.json or missing> <report.md>`.
"""

import json
import sys
from pathlib import Path

METRICS = [
    ("roc_auc", "ROC-AUC", True),
    ("pr_auc", "PR-AUC", True),
    ("f1", "F1", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("brier", "Brier", False),
    ("log_loss", "Log loss", False),
]


def read_card(path: Path) -> dict | None:
    """The model card at `path`, or `None` when that training run did not produce one."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def row(name: str, label: str, higher_is_better: bool, pr: dict, base: dict | None) -> str:
    """One table row; the delta is marked good or bad according to the metric's direction."""
    value = pr["test_metrics"][name]
    if base is None or name not in base["test_metrics"]:
        return f"| {label} | {value:.4f} | – | – |"
    delta = value - base["test_metrics"][name]
    better = delta > 0 if higher_is_better else delta < 0
    mark = "" if abs(delta) < 5e-4 else (" ✅" if better else " ⚠️")
    return f"| {label} | {value:.4f} | {base['test_metrics'][name]:.4f} | {delta:+.4f}{mark} |"


def main(pr_path: Path, base_path: Path, out_path: Path) -> None:
    """Write the Markdown report."""
    pr, base = read_card(pr_path), read_card(base_path)
    if pr is None:
        raise SystemExit(f"{pr_path} not found: training on the pull request failed.")
    lines = [
        "## Kết quả train trên CI",
        "",
        f"Mô hình `{pr['model_name']}`, train trên `train.csv` (07/2015–12/2016), đánh giá trên `test.csv` "
        f"(01–04/2017). Ngưỡng quyết định: {pr['decision_threshold']:.3f}"
        + (f" (base: {base['decision_threshold']:.3f})." if base else "."),
        "",
        "| Metric | PR | Base | Chênh lệch |",
        "|---|---:|---:|---:|",
        *(row(name, label, higher, pr, base) for name, label, higher in METRICS),
        "",
    ]
    if base is None:
        lines.append("Không train được mô hình trên nhánh base, nên không có cột so sánh.")
    else:
        lines.append("✅ tốt hơn base, ⚠️ kém hơn base (Brier và log loss: thấp hơn là tốt hơn).")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
