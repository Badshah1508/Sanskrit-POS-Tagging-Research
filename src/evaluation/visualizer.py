from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_preparation.dataset import TARGET_TAGS


MODEL_ORDER = ["BiLSTM_CRF", "XLM_RoBERTa", "IndicBERT_v2", "MuRIL", "Sanskrit_BERT"]
MODEL_DISPLAY_NAMES = {
    "BiLSTM_CRF": "M1: BiLSTM-CRF",
    "XLM_RoBERTa": "M2: XLM-RoBERTa",
    "IndicBERT_v2": "M3: IndicBERT-v2",
    "MuRIL": "M4: MuRIL",
    "Sanskrit_BERT": "M5: Sanskrit-BERT",
}


def generate_comparison_tables(registry_df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Generate the four mandatory dissertation comparison tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}

    successful_df = registry_df[registry_df["status"] == "SUCCESS"].copy()

    # Table 1: In-Domain
    t1_rows = []
    for model_name in MODEL_ORDER:
        for setting in ["literature_in_domain", "news_in_domain"]:
            sub = successful_df[
                (successful_df["model"] == model_name) &
                (successful_df["setting"] == setting)
            ]
            if not sub.empty:
                row = sub.sort_values(by="validation_macro_f1", ascending=False).iloc[0]
                t1_rows.append({
                    "model_id": MODEL_DISPLAY_NAMES[model_name].split(":")[0],
                    "model": model_name,
                    "domain": "Literature" if "literature" in setting else "News",
                    "setting": setting,
                    "val_macro_f1": round(row.get("validation_macro_f1"), 4) if row.get("validation_macro_f1") is not None else None,
                    "test_macro_f1": round(row.get("test_macro_f1"), 4) if row.get("test_macro_f1") is not None else None,
                    "test_accuracy": round(row.get("test_accuracy"), 4) if row.get("test_accuracy") is not None else None,
                    "test_weighted_f1": round(row.get("test_weighted_f1"), 4) if row.get("test_weighted_f1") is not None else None,
                })
    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(output_dir / "table1_in_domain.csv", index=False)
    tables["table1"] = t1_df

    # Table 2: Zero-Shot Cross-Domain (Literature to News)
    t2_rows = []
    for model_name in MODEL_ORDER:
        sub_zs = successful_df[
            (successful_df["model"] == model_name) &
            (successful_df["setting"] == "zero_shot")
        ]
        sub_lit = successful_df[
            (successful_df["model"] == model_name) &
            (successful_df["setting"] == "literature_in_domain")
        ]
        if not sub_zs.empty and not sub_lit.empty:
            zs_row = sub_zs.sort_values(by="validation_macro_f1", ascending=False).iloc[0]
            lit_row = sub_lit.sort_values(by="validation_macro_f1", ascending=False).iloc[0]
            lit_f1 = lit_row["test_macro_f1"]
            zs_f1 = zs_row["test_macro_f1"]
            abs_deg = (lit_f1 - zs_f1) if (lit_f1 is not None and zs_f1 is not None) else None
            rel_deg = (abs_deg / lit_f1 * 100) if (lit_f1 and abs_deg is not None) else None

            t2_rows.append({
                "model_id": MODEL_DISPLAY_NAMES[model_name].split(":")[0],
                "model": model_name,
                "literature_test_macro_f1": round(lit_f1, 4) if lit_f1 is not None else None,
                "news_zero_shot_macro_f1": round(zs_f1, 4) if zs_f1 is not None else None,
                "absolute_f1_drop": round(abs_deg, 4) if abs_deg is not None else None,
                "relative_f1_drop_pct": round(rel_deg, 2) if rel_deg is not None else None,
                "zero_shot_accuracy": round(zs_row.get("test_accuracy"), 4) if zs_row.get("test_accuracy") is not None else None,
                "zero_shot_weighted_f1": round(zs_row.get("test_weighted_f1"), 4) if zs_row.get("test_weighted_f1") is not None else None,
            })
    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_csv(output_dir / "table2_zero_shot.csv", index=False)
    tables["table2"] = t2_df

    # Table 3: Few-Shot Adaptation (budget columns: 0, 50, 100, 200, 500, 1000)
    fs_runs = successful_df[
        successful_df["setting"].isin(["few_shot", "zero_shot"])
    ].copy()
    fs_runs.loc[fs_runs["setting"] == "zero_shot", "annotation_budget"] = 0

    t3_pivot = fs_runs.pivot_table(
        index="model",
        columns="annotation_budget",
        values="test_macro_f1",
        aggfunc="mean"
    ).reindex(MODEL_ORDER).reset_index()
    t3_pivot.columns = [str(c) for c in t3_pivot.columns]
    t3_pivot.to_csv(output_dir / "table3_few_shot.csv", index=False)
    tables["table3"] = t3_pivot

    # Table 4: Complete Individual Experiment History
    registry_df.to_csv(output_dir / "table4_experiment_history.csv", index=False)
    tables["table4"] = registry_df

    return tables


def plot_few_shot_curves(registry_df: pd.DataFrame, output_path: Path) -> None:
    """Plot few-shot adaptation learning curve (Budget vs Test Macro F1) across 5 models."""
    fs_runs = registry_df[
        registry_df["setting"].isin(["few_shot", "zero_shot"]) &
        (registry_df["status"] == "SUCCESS")
    ].copy()
    fs_runs.loc[fs_runs["setting"] == "zero_shot", "annotation_budget"] = 0

    if fs_runs.empty:
        return

    fs_runs["display_model"] = fs_runs["model"].map(MODEL_DISPLAY_NAMES)
    display_order = [MODEL_DISPLAY_NAMES[m] for m in MODEL_ORDER if m in fs_runs["model"].unique()]

    plt.figure(figsize=(10, 6.5))
    palette = sns.color_palette("tab10", n_colors=len(display_order))
    markers = ["o", "s", "^", "D", "v"]
    
    sns.lineplot(
        data=fs_runs,
        x="annotation_budget",
        y="test_macro_f1",
        hue="display_model",
        hue_order=display_order,
        style="display_model",
        style_order=display_order,
        markers=markers[:len(display_order)],
        dashes=False,
        linewidth=2.5,
        markersize=8,
        palette=palette,
        errorbar="sd",
    )
    plt.title("Few-Shot Adaptation: Target Domain News Macro F1 vs Sentence Budget (5 Models)", fontsize=13, pad=12)
    plt.xlabel("Annotation Budget (News Sentences)", fontsize=11)
    plt.ylabel("Locked Test Macro F1", fontsize=11)
    plt.xticks([0, 50, 100, 200, 500, 1000])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Model Architecture", fontsize=10, title_fontsize=11, loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(confusion_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot publication-quality confusion matrix heatmap."""
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        confusion_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_TAGS,
        yticklabels=TARGET_TAGS,
        cbar_kws={"label": "Token Count"},
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted POS Tag", fontsize=12)
    plt.ylabel("Gold POS Tag", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
