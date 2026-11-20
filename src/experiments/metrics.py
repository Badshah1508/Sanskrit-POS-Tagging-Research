from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data import TARGET_TAGS


def evaluate_sequences(
    frame: pd.DataFrame,
    predictions: Sequence[Sequence[str]],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if len(frame) != len(predictions):
        raise ValueError("Prediction sentence count does not match the evaluation frame")
    gold = [tag for tags in frame["tags"] for tag in tags]
    predicted = [tag for sequence in predictions for tag in sequence]
    if len(gold) != len(predicted):
        raise ValueError("Prediction token count does not match gold token count")
    metrics = {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(precision_score(gold, predicted, labels=TARGET_TAGS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(gold, predicted, labels=TARGET_TAGS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(gold, predicted, labels=TARGET_TAGS, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(gold, predicted, labels=TARGET_TAGS, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(gold, predicted, labels=TARGET_TAGS, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(gold, predicted, labels=TARGET_TAGS, average="weighted", zero_division=0)),
        "tokens": len(gold),
    }
    report = pd.DataFrame(classification_report(
        gold, predicted, labels=TARGET_TAGS, output_dict=True, zero_division=0
    )).T.reset_index(names="tag")
    matrix = pd.DataFrame(
        confusion_matrix(gold, predicted, labels=TARGET_TAGS),
        index=TARGET_TAGS,
        columns=TARGET_TAGS,
    )
    return metrics, report, matrix


def save_evaluation(
    output_dir: Path,
    metrics: dict[str, float],
    report: pd.DataFrame,
    matrix: pd.DataFrame,
    predictions: Sequence[Sequence[str]],
    frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    report.to_csv(output_dir / "classification_report.csv", index=False)
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    rows = []
    for row, sequence in zip(frame.itertuples(), predictions):
        rows.extend(
            {"doc_id": row.doc_id, "sentence_id": row.sentence_id, "gold": gold, "predicted": pred}
            for gold, pred in zip(row.tags, sequence)
        )
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
