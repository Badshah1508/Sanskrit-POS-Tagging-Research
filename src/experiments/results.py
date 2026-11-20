from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULT_COLUMNS = [
    "model",
    "setting",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "tokens",
]


def write_results_summary(results_root: Path) -> pd.DataFrame:
    """Collect executed metric files and write CSV/Markdown summaries."""
    rows = []
    for metrics_path in sorted(results_root.glob("**/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        setting_dir = metrics_path.parent
        model = {"bilstm_crf": "BiLSTM-CRF"}.get(
            setting_dir.name, setting_dir.name
        )
        setting = {
            "in_domain": "literature_in_domain",
            "cross_domain": "literature_to_news_zero_shot",
        }.get(setting_dir.parent.name, setting_dir.parent.name)
        rows.append({"model": model, "setting": setting, **metrics})

    summary = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(["model", "setting"]).reset_index(drop=True)
    results_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_root / "model_performance_summary.csv", index=False)
    _write_markdown(summary, results_root / "model_performance_summary.md")
    return summary


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    headers = list(summary.columns)
    lines = [
        "# Model Performance Summary",
        "",
        "Only successfully executed evaluations are included. Missing models or "
        "settings are pending and are not represented by placeholder metrics.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary.itertuples(index=False, name=None):
        formatted = [
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
