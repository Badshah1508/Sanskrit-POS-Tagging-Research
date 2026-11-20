from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_preparation.dataset import TARGET_TAGS


def evaluate_predictions(
    frame: pd.DataFrame,
    predictions: Sequence[Sequence[str]],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute comprehensive token-level POS evaluation metrics.
    
    Returns:
      (summary_metrics, per_tag_metrics, classification_report, confusion_matrix)
    """
    if len(frame) != len(predictions):
        raise ValueError(
            f"Sentence count mismatch: frame has {len(frame)} sentences, predictions has {len(predictions)}"
        )

    gold_tokens: list[str] = [tag for tags in frame["tags"] for tag in tags]
    pred_tokens: list[str] = [tag for seq in predictions for tag in seq]

    if len(gold_tokens) != len(pred_tokens):
        raise ValueError(
            f"Token count mismatch: gold has {len(gold_tokens)} tokens, predictions has {len(pred_tokens)}"
        )

    summary_metrics = {
        "accuracy": float(accuracy_score(gold_tokens, pred_tokens)),
        "macro_precision": float(
            precision_score(gold_tokens, pred_tokens, labels=TARGET_TAGS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(gold_tokens, pred_tokens, labels=TARGET_TAGS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(gold_tokens, pred_tokens, labels=TARGET_TAGS, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(gold_tokens, pred_tokens, labels=TARGET_TAGS, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(gold_tokens, pred_tokens, labels=TARGET_TAGS, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(gold_tokens, pred_tokens, labels=TARGET_TAGS, average="weighted", zero_division=0)
        ),
        "total_tokens": len(gold_tokens),
        "total_sentences": len(frame),
    }

    # Full classification report
    report_dict = classification_report(
        gold_tokens, pred_tokens, labels=TARGET_TAGS, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T.reset_index(names="tag")

    # Per-tag metrics table
    per_tag_rows = []
    for tag in TARGET_TAGS:
        tag_data = report_dict.get(tag, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        per_tag_rows.append({
            "tag": tag,
            "precision": float(tag_data["precision"]),
            "recall": float(tag_data["recall"]),
            "f1": float(tag_data["f1-score"]),
            "support": int(tag_data["support"]),
        })
    per_tag_df = pd.DataFrame(per_tag_rows)

    # 24x24 Confusion matrix
    conf_mat = confusion_matrix(gold_tokens, pred_tokens, labels=TARGET_TAGS)
    confusion_df = pd.DataFrame(conf_mat, index=TARGET_TAGS, columns=TARGET_TAGS)

    return summary_metrics, per_tag_df, report_df, confusion_df


def save_evaluation_results(
    output_dir: Path,
    summary_metrics: dict[str, Any],
    per_tag_df: pd.DataFrame,
    report_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    predictions: Sequence[Sequence[str]],
    frame: pd.DataFrame,
) -> None:
    """Save all standard evaluation artifacts into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. metrics.json & metrics.csv
    (output_dir / "metrics.json").write_text(
        json.dumps(summary_metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary_metrics]).to_csv(output_dir / "metrics.csv", index=False)

    # 2. per_tag_metrics.csv
    per_tag_df.to_csv(output_dir / "per_tag_metrics.csv", index=False)

    # 3. classification_report.csv
    report_df.to_csv(output_dir / "classification_report.csv", index=False)

    # 4. confusion_matrix.csv
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")

    # 5. predictions.csv
    pred_rows = []
    for row, pred_seq in zip(frame.itertuples(), predictions):
        doc_id = getattr(row, "doc_id", row.sentence_id)
        sentence_id = row.sentence_id
        for tid, (token, gold_tag, pred_tag) in enumerate(zip(row.tokens, row.tags, pred_seq)):
            pred_rows.append({
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "token_id": tid,
                "token": token,
                "gold": gold_tag,
                "predicted": pred_tag,
            })
    pd.DataFrame(pred_rows).to_csv(output_dir / "predictions.csv", index=False)
