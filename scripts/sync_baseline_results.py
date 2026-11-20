"""Synchronize all experiment outputs and comparison tables across the five models:
M1: BiLSTM-CRF
M2: XLM-RoBERTa
M3: IndicBERT-v2
M4: MuRIL
M5: Sanskrit-BERT
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import pandas as pd
import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "outputs" / "experiments" / "experiment_registry.csv"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "baseline_experiments"
EVALS_DIR = PROJECT_ROOT / "outputs" / "evaluations"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

MODEL_MAP = {
    "BiLSTM_CRF": ("M1", "BiLSTM-CRF"),
    "XLM_RoBERTa": ("M2", "XLM-RoBERTa"),
    "IndicBERT_v2": ("M3", "IndicBERT-v2"),
    "MuRIL": ("M4", "MuRIL"),
    "Sanskrit_BERT": ("M5", "Sanskrit-BERT"),
}

SENTENCE_TOKEN_MAP = {
    50: 455,
    100: 911,
    200: 1823,
    500: 4560,
    1000: 9120,
}


def sync_all() -> None:
    if not REGISTRY_PATH.exists():
        print(f"Registry not found at {REGISTRY_PATH}")
        return

    reg_df = pd.read_csv(REGISTRY_PATH)
    successful_df = reg_df[reg_df["status"] == "SUCCESS"].copy()

    # 1. Literature In-Domain
    lit_rows = []
    for model_key, (mid, mname) in MODEL_MAP.items():
        sub = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "literature_in_domain")
        ]
        if not sub.empty:
            row = sub.sort_values(by="validation_macro_f1", ascending=False).iloc[0]
            lit_rows.append({
                "model_id": mid,
                "model_name": mname,
                "dataset": "Sanskrit Literature",
                "split": "held_out_test",
                "accuracy": row["test_accuracy"],
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": row["test_macro_f1"],
                "weighted_f1": row["test_weighted_f1"],
                "training_time": None,
                "inference_time": None,
                "checkpoint_path": str(Path(row["run_dir"]) / "checkpoint"),
                "experiment_status": "completed",
            })
        else:
            lit_rows.append({
                "model_id": mid,
                "model_name": mname,
                "dataset": "Sanskrit Literature",
                "split": "held_out_test",
                "accuracy": None,
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": None,
                "weighted_f1": None,
                "training_time": None,
                "inference_time": None,
                "checkpoint_path": f"outputs/baseline_experiments/checkpoints/{mname.lower()}_lit.pt",
                "experiment_status": "pending",
            })
    lit_df = pd.DataFrame(lit_rows)
    lit_df.to_csv(BASELINE_DIR / "literature_in_domain_results.csv", index=False)
    print("Updated literature_in_domain_results.csv")

    # 2. News In-Domain
    news_rows = []
    for model_key, (mid, mname) in MODEL_MAP.items():
        sub = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "news_in_domain")
        ]
        if not sub.empty:
            row = sub.sort_values(by="validation_macro_f1", ascending=False).iloc[0]
            news_rows.append({
                "model_id": mid,
                "model_name": mname,
                "dataset": "Sanskrit News",
                "split": "locked_test",
                "accuracy": row["test_accuracy"],
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": row["test_macro_f1"],
                "weighted_f1": row["test_weighted_f1"],
                "training_time": None,
                "inference_time": None,
                "checkpoint_path": str(Path(row["run_dir"]) / "checkpoint"),
                "experiment_status": "completed",
            })
        else:
            news_rows.append({
                "model_id": mid,
                "model_name": mname,
                "dataset": "Sanskrit News",
                "split": "locked_test",
                "accuracy": None,
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": None,
                "weighted_f1": None,
                "training_time": None,
                "inference_time": None,
                "checkpoint_path": f"outputs/baseline_experiments/checkpoints/{mname.lower()}_news.pt",
                "experiment_status": "pending",
            })
    news_df = pd.DataFrame(news_rows)
    news_df.to_csv(BASELINE_DIR / "news_in_domain_results.csv", index=False)
    print("Updated news_in_domain_results.csv")

    # 3. Cross-Domain Zero-Shot
    zs_rows = []
    for model_key, (mid, mname) in MODEL_MAP.items():
        sub_zs = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "zero_shot")
        ]
        sub_lit = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "literature_in_domain")
        ]
        lit_f1 = sub_lit.iloc[0]["test_macro_f1"] if not sub_lit.empty else None

        if not sub_zs.empty:
            row = sub_zs.iloc[0]
            zs_f1 = row["test_macro_f1"]
            abs_drop = (lit_f1 - zs_f1) if (lit_f1 is not None and zs_f1 is not None) else None
            rel_drop = (abs_drop / lit_f1 * 100) if (lit_f1 and abs_drop is not None) else None
            zs_rows.append({
                "model_id": mid,
                "model_name": mname,
                "source_domain": "Sanskrit Literature",
                "target_domain": "Sanskrit News",
                "accuracy": row["test_accuracy"],
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": zs_f1,
                "weighted_f1": row["test_weighted_f1"],
                "literature_in_domain_macro_f1": lit_f1,
                "absolute_f1_drop": round(abs_drop, 4) if abs_drop is not None else None,
                "relative_f1_drop_percent": round(rel_drop, 2) if rel_drop is not None else None,
                "checkpoint_path": str(Path(row["run_dir"]) / "checkpoint"),
                "experiment_status": "completed",
            })
        else:
            zs_rows.append({
                "model_id": mid,
                "model_name": mname,
                "source_domain": "Sanskrit Literature",
                "target_domain": "Sanskrit News",
                "accuracy": None,
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": None,
                "weighted_f1": None,
                "literature_in_domain_macro_f1": lit_f1,
                "absolute_f1_drop": None,
                "relative_f1_drop_percent": None,
                "checkpoint_path": f"outputs/baseline_experiments/checkpoints/{mname.lower()}_lit.pt",
                "experiment_status": "pending",
            })
    zs_df = pd.DataFrame(zs_rows)
    zs_df.to_csv(BASELINE_DIR / "cross_domain_zero_shot_results.csv", index=False)
    print("Updated cross_domain_zero_shot_results.csv")

    # 4. Few-Shot Adaptation
    fs_rows = []
    seeds = [42, 123, 2026]
    budgets = [50, 100, 200, 500, 1000]

    for model_key, (mid, mname) in MODEL_MAP.items():
        # Zero-shot row (budget 0)
        sub_zs = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "zero_shot")
        ]
        zs_f1 = sub_zs.iloc[0]["test_macro_f1"] if not sub_zs.empty else None
        zs_acc = sub_zs.iloc[0]["test_accuracy"] if not sub_zs.empty else None
        zs_wf1 = sub_zs.iloc[0]["test_weighted_f1"] if not sub_zs.empty else None
        zs_dir = str(Path(sub_zs.iloc[0]["run_dir"]) / "checkpoint") if not sub_zs.empty else ""

        fs_rows.append({
            "model_id": mid,
            "model_name": mname,
            "annotation_budget": 0,
            "random_seed": 42,
            "annotated_sentences": 0,
            "annotated_tokens": 0,
            "accuracy": zs_acc,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": zs_f1,
            "weighted_f1": zs_wf1,
            "zero_shot_macro_f1": zs_f1,
            "improvement_percentage_points": 0.0,
            "training_time": 0.0,
            "checkpoint_path": zs_dir,
            "experiment_status": "completed" if not sub_zs.empty else "pending",
        })

        for b in budgets:
            tokens_count = SENTENCE_TOKEN_MAP.get(b, b * 9)
            for s in seeds:
                sub_fs = successful_df[
                    (successful_df["model"] == model_key) &
                    (successful_df["setting"] == "few_shot") &
                    (successful_df["annotation_budget"] == b) &
                    (successful_df["seed"] == s)
                ]
                if not sub_fs.empty:
                    row = sub_fs.iloc[0]
                    f1 = row["test_macro_f1"]
                    imp = (f1 - zs_f1) * 100 if (f1 is not None and zs_f1 is not None) else None
                    fs_rows.append({
                        "model_id": mid,
                        "model_name": mname,
                        "annotation_budget": b,
                        "random_seed": s,
                        "annotated_sentences": b,
                        "annotated_tokens": tokens_count,
                        "accuracy": row["test_accuracy"],
                        "macro_precision": None,
                        "macro_recall": None,
                        "macro_f1": f1,
                        "weighted_f1": row["test_weighted_f1"],
                        "zero_shot_macro_f1": zs_f1,
                        "improvement_percentage_points": round(imp, 2) if imp is not None else None,
                        "training_time": None,
                        "checkpoint_path": str(Path(row["run_dir"]) / "checkpoint"),
                        "experiment_status": "completed",
                    })
                else:
                    fs_rows.append({
                        "model_id": mid,
                        "model_name": mname,
                        "annotation_budget": b,
                        "random_seed": s,
                        "annotated_sentences": b,
                        "annotated_tokens": tokens_count,
                        "accuracy": None,
                        "macro_precision": None,
                        "macro_recall": None,
                        "macro_f1": None,
                        "weighted_f1": None,
                        "zero_shot_macro_f1": zs_f1,
                        "improvement_percentage_points": None,
                        "training_time": None,
                        "checkpoint_path": f"outputs/baseline_experiments/checkpoints/{mname.lower()}_fs_{b}_s{s}.pt",
                        "experiment_status": "pending",
                    })
    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_csv(BASELINE_DIR / "few_shot_results.csv", index=False)
    print("Updated few_shot_results.csv")

    # 5. Master Results
    master_rows = []
    # Lit In-Domain
    for _, r in lit_df.iterrows():
        master_rows.append({
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "experiment_type": "literature_in_domain",
            "source_domain": "Sanskrit Literature",
            "target_domain": "Sanskrit Literature",
            "annotation_budget": 0,
            "random_seed": 42,
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "improvement_over_zero_shot": 0.0,
            "experiment_status": r["experiment_status"],
        })
    # News In-Domain
    for _, r in news_df.iterrows():
        master_rows.append({
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "experiment_type": "news_in_domain",
            "source_domain": "Sanskrit News",
            "target_domain": "Sanskrit News",
            "annotation_budget": 0,
            "random_seed": 42,
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "improvement_over_zero_shot": None,
            "experiment_status": r["experiment_status"],
        })
    # Zero-Shot
    for _, r in zs_df.iterrows():
        master_rows.append({
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "experiment_type": "cross_domain_zero_shot",
            "source_domain": "Sanskrit Literature",
            "target_domain": "Sanskrit News",
            "annotation_budget": 0,
            "random_seed": 42,
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "improvement_over_zero_shot": 0.0,
            "experiment_status": r["experiment_status"],
        })
    # Few-Shot (budget > 0)
    for _, r in fs_df[fs_df["annotation_budget"] > 0].iterrows():
        master_rows.append({
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "experiment_type": "few_shot_adaptation",
            "source_domain": "Sanskrit Literature",
            "target_domain": "Sanskrit News",
            "annotation_budget": r["annotation_budget"],
            "random_seed": r["random_seed"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "improvement_over_zero_shot": r["improvement_percentage_points"],
            "experiment_status": r["experiment_status"],
        })
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(BASELINE_DIR / "master_results.csv", index=False)
    print("Updated master_results.csv")

    # 6. Excel Workbook
    with pd.ExcelWriter(BASELINE_DIR / "baseline_experiment_results.xlsx", engine="openpyxl") as writer:
        lit_df.to_excel(writer, sheet_name="Literature In-Domain", index=False)
        news_df.to_excel(writer, sheet_name="News In-Domain", index=False)
        zs_df.to_excel(writer, sheet_name="Zero-Shot Cross-Domain", index=False)
        fs_df.to_excel(writer, sheet_name="Few-Shot Adaptation", index=False)
        master_df.to_excel(writer, sheet_name="Master Results", index=False)
    print("Updated baseline_experiment_results.xlsx")

    # 7. Model Performance Summary
    summary_rows = []
    for model_key, (mid, mname) in MODEL_MAP.items():
        # Lit
        s_lit = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "literature_in_domain")
        ]
        if not s_lit.empty:
            r = s_lit.iloc[0]
            summary_rows.append({
                "model": mname,
                "setting": "literature_in_domain",
                "accuracy": round(r["test_accuracy"], 4),
                "macro_f1": round(r["test_macro_f1"], 4),
                "weighted_f1": round(r["test_weighted_f1"], 4),
                "tokens": 1807,
            })
        # News
        s_news = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "news_in_domain")
        ]
        if not s_news.empty:
            r = s_news.iloc[0]
            summary_rows.append({
                "model": mname,
                "setting": "news_in_domain",
                "accuracy": round(r["test_accuracy"], 4),
                "macro_f1": round(r["test_macro_f1"], 4),
                "weighted_f1": round(r["test_weighted_f1"], 4),
                "tokens": 7324,
            })
        # Zero-shot
        s_zs = successful_df[
            (successful_df["model"] == model_key) &
            (successful_df["setting"] == "zero_shot")
        ]
        if not s_zs.empty:
            r = s_zs.iloc[0]
            summary_rows.append({
                "model": mname,
                "setting": "literature_to_news_zero_shot",
                "accuracy": round(r["test_accuracy"], 4),
                "macro_f1": round(r["test_macro_f1"], 4),
                "weighted_f1": round(r["test_weighted_f1"], 4),
                "tokens": 7324,
            })
    summary_df = pd.DataFrame(summary_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False)
    
    md_content = "# Model Performance Summary\n\n"
    md_content += "Only successfully executed evaluations are included. Missing models or settings are pending and are not represented by placeholder metrics.\n\n"
    md_content += summary_df.to_markdown(index=False) + "\n"
    (RESULTS_DIR / "model_performance_summary.md").write_text(md_content, encoding="utf-8")
    print("Updated model_performance_summary.csv and .md")


if __name__ == "__main__":
    sync_all()
