"""Master experiment runner for Sanskrit POS Cross-Domain Generalization and Few-Shot Adaptation.

Executes experiments strictly adhering to the research protocol:
- Exp 1: Literature In-Domain
- Exp 2: News In-Domain
- Exp 3: Zero-Shot Cross-Domain (Literature -> News)
- Exp 4: Few-Shot Adaptation (Budgets: 50, 100, 200, 500, 1000 across seeds 42, 123, 2026)
- Inference on unannotated Sanskrit news (3,000 sentences)
- Automatic generation of tables, plots, registries, and error reports
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_preparation.dataset import (
    TARGET_TAGS,
    get_nested_few_shot_subsets,
    load_dataset_splits,
    load_unannotated_news,
    validate_split_leakage,
)
from src.evaluation.evaluator import evaluate_predictions, save_evaluation_results
from src.evaluation.inference import (
    run_unannotated_inference_bilstm,
    run_unannotated_inference_transformer,
)
from src.evaluation.visualizer import (
    generate_comparison_tables,
    plot_confusion_matrix_heatmap,
    plot_few_shot_curves,
)
from src.experiments.tracking import (
    append_registry,
    create_hierarchical_run_directory,
)
from src.models.bilstm_crf import (
    BiLSTMCRF,
    SequenceTaggingDataset,
    build_vocabulary,
    collate_sequences,
)
from src.models.transformer_tagger import TransformerTagger
from src.training.trainer import decode_bilstm_predictions, train_bilstm_crf, train_transformer


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MODEL_SPECS = {
    "bilstm_crf": {
        "display_name": "BiLSTM_CRF",
        "type": "bilstm",
        "config_path": "configs/bilstm_crf.json",
    },
    "xlm_roberta": {
        "display_name": "XLM_RoBERTa",
        "type": "transformer",
        "hf_model_id": "FacebookAI/xlm-roberta-base",
        "config_path": "configs/xlm_roberta.json",
        "is_iast": False,
    },
    "indicbert_v2": {
        "display_name": "IndicBERT_v2",
        "type": "transformer",
        "hf_model_id": "ai4bharat/IndicBERTv2-MLM-only",
        "config_path": "configs/indicbert_v2.json",
        "is_iast": False,
    },
    "muril": {
        "display_name": "MuRIL",
        "type": "transformer",
        "hf_model_id": "google/muril-base-cased",
        "config_path": "configs/muril.json",
        "is_iast": False,
    },
    "sanskrit_bert": {
        "display_name": "Sanskrit_BERT",
        "type": "transformer",
        "hf_model_id": "Matej/bert-base-buddhist-sanskrit-v2",
        "config_path": "configs/sanskrit_bert.json",
        "is_iast": True,
    },
}


def run_bilstm_experiments(
    project_root: Path,
    splits: dict[str, pd.DataFrame],
    device: torch.device,
    config: dict[str, Any],
    seeds: list[int] = [42, 123, 2026],
    budgets: list[int] = [50, 100, 200, 500, 1000],
    skip_few_shot: bool = False,
) -> None:
    """Execute complete BiLSTM-CRF experimental suite."""
    print("=" * 70)
    print("STARTING BiLSTM-CRF EXPERIMENTS")
    print("=" * 70)

    exp_root = project_root / "outputs" / "experiments"
    registry_path = exp_root / "experiment_registry.csv"
    figures_dir = project_root / "outputs" / "figures"
    evals_dir = project_root / "outputs" / "evaluations"

    lit_train = splits["literature_train"]
    lit_val = splits["literature_validation"]
    lit_test = splits["literature_test"]

    news_pool = splits["news_adaptation_pool"]
    news_val = splits["news_validation"]
    news_test = splits["news_test"]

    # Build base literature vocabulary
    lit_vocab = build_vocabulary(lit_train["tokens"])
    print(f"Literature Vocabulary size: {len(lit_vocab)}")

    # -------------------------------------------------------------------------
    # Experiment 1: Literature In-Domain
    # -------------------------------------------------------------------------
    seed = config.get("seed", 42)
    set_seed(seed)
    print("\n>>> Running Experiment 1: Literature In-Domain (BiLSTM-CRF)...")
    exp_id, run_dir = create_hierarchical_run_directory(
        exp_root, model="BiLSTM_CRF", setting="literature_in_domain", seed=seed
    )
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        model = BiLSTMCRF(
            vocab_size=len(lit_vocab),
            num_labels=len(TARGET_TAGS),
            embedding_dim=config.get("embedding_dim", 128),
            hidden_size=config.get("hidden_size", 128),
            dropout=config.get("dropout", 0.2),
        ).to(device)

        best_model, val_metrics, history_df = train_bilstm_crf(
            model=model,
            train_df=lit_train,
            val_df=lit_val,
            vocabulary=lit_vocab,
            device=device,
            run_dir=run_dir,
            config=config,
        )

        # Evaluate on Literature Test
        test_ds = SequenceTaggingDataset(lit_test, lit_vocab)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.get("batch_size", 32), shuffle=False, collate_fn=collate_sequences
        )
        test_preds = decode_bilstm_predictions(best_model, test_loader, device)
        summary, per_tag, report, conf_mat = evaluate_predictions(lit_test, test_preds)

        save_evaluation_results(
            run_dir, summary, per_tag, report, conf_mat, test_preds, lit_test
        )
        plot_confusion_matrix_heatmap(
            conf_mat,
            "BiLSTM-CRF Literature In-Domain Confusion Matrix",
            figures_dir / "bilstm_lit_in_domain_confusion.png",
        )

        append_registry(
            registry_path,
            experiment_id=exp_id,
            model="BiLSTM_CRF",
            setting="literature_in_domain",
            annotation_budget=0,
            seed=seed,
            status="SUCCESS",
            run_dir=run_dir,
            validation_macro_f1=val_metrics.get("macro_f1"),
            test_macro_f1=summary["macro_f1"],
            test_accuracy=summary["accuracy"],
            test_weighted_f1=summary["weighted_f1"],
            started_at=started_at,
        )
        print(f"Exp 1 Completed: Val Macro F1={val_metrics.get('macro_f1'):.4f}, Test Macro F1={summary['macro_f1']:.4f}")
        
        # Save literature source pretrained model to standard model dir as well
        best_model.save(
            project_root / "outputs" / "models" / "bilstm_crf" / "bilstm_source_pretrained.pt",
            lit_vocab,
            config,
        )
        lit_checkpoint_path = run_dir / "checkpoint" / "model.pt"

    except Exception as e:
        print(f"Error in Exp 1: {e}")
        append_registry(
            registry_path,
            experiment_id=exp_id,
            model="BiLSTM_CRF",
            setting="literature_in_domain",
            annotation_budget=0,
            seed=seed,
            status="FAILED",
            run_dir=run_dir,
            error=str(e),
            started_at=started_at,
        )
        raise e

    # -------------------------------------------------------------------------
    # Experiment 2: News In-Domain Reference Run
    # -------------------------------------------------------------------------
    print("\n>>> Running Experiment 2: News In-Domain Reference Run (BiLSTM-CRF)...")
    set_seed(seed)
    exp_id, run_dir = create_hierarchical_run_directory(
        exp_root, model="BiLSTM_CRF", setting="news_in_domain", seed=seed
    )
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        news_vocab = build_vocabulary(news_pool["tokens"])
        news_model = BiLSTMCRF(
            vocab_size=len(news_vocab),
            num_labels=len(TARGET_TAGS),
            embedding_dim=config.get("embedding_dim", 128),
            hidden_size=config.get("hidden_size", 128),
            dropout=config.get("dropout", 0.2),
        ).to(device)

        best_news_model, news_val_metrics, _ = train_bilstm_crf(
            model=news_model,
            train_df=news_pool,
            val_df=news_val,
            vocabulary=news_vocab,
            device=device,
            run_dir=run_dir,
            config=config,
        )

        test_ds = SequenceTaggingDataset(news_test, news_vocab)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.get("batch_size", 32), shuffle=False, collate_fn=collate_sequences
        )
        test_preds = decode_bilstm_predictions(best_news_model, test_loader, device)
        summary, per_tag, report, conf_mat = evaluate_predictions(news_test, test_preds)

        save_evaluation_results(
            run_dir, summary, per_tag, report, conf_mat, test_preds, news_test
        )
        plot_confusion_matrix_heatmap(
            conf_mat,
            "BiLSTM-CRF News In-Domain Confusion Matrix",
            figures_dir / "bilstm_news_in_domain_confusion.png",
        )

        append_registry(
            registry_path,
            experiment_id=exp_id,
            model="BiLSTM_CRF",
            setting="news_in_domain",
            annotation_budget=0,
            seed=seed,
            status="SUCCESS",
            run_dir=run_dir,
            validation_macro_f1=news_val_metrics.get("macro_f1"),
            test_macro_f1=summary["macro_f1"],
            test_accuracy=summary["accuracy"],
            test_weighted_f1=summary["weighted_f1"],
            started_at=started_at,
        )
        print(f"Exp 2 Completed: News Val Macro F1={news_val_metrics.get('macro_f1'):.4f}, News Test Macro F1={summary['macro_f1']:.4f}")

    except Exception as e:
        print(f"Error in Exp 2: {e}")
        append_registry(
            registry_path,
            experiment_id=exp_id,
            model="BiLSTM_CRF",
            setting="news_in_domain",
            annotation_budget=0,
            seed=seed,
            status="FAILED",
            run_dir=run_dir,
            error=str(e),
            started_at=started_at,
        )

    # -------------------------------------------------------------------------
    # Experiment 3: Zero-Shot Cross-Domain Generalization (Literature -> News)
    # -------------------------------------------------------------------------
    print("\n>>> Running Experiment 3: Zero-Shot Cross-Domain Generalization (BiLSTM-CRF)...")
    exp_id, run_dir = create_hierarchical_run_directory(
        exp_root, model="BiLSTM_CRF", setting="zero_shot", seed=seed
    )
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        # Load literature-trained model
        lit_model, lit_vocab, _ = BiLSTMCRF.load(lit_checkpoint_path, device=device)
        
        # Evaluate directly on locked news test
        test_ds = SequenceTaggingDataset(news_test, lit_vocab)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.get("batch_size", 32), shuffle=False, collate_fn=collate_sequences
        )
        test_preds = decode_bilstm_predictions(lit_model, test_loader, device)
        summary, per_tag, report, conf_mat = evaluate_predictions(news_test, test_preds)

        save_evaluation_results(
            run_dir, summary, per_tag, report, conf_mat, test_preds, news_test
        )
        plot_confusion_matrix_heatmap(
            conf_mat,
            "BiLSTM-CRF Zero-Shot Cross-Domain Confusion Matrix",
            figures_dir / "bilstm_zero_shot_confusion.png",
        )

        append_registry(
            registry_path,
            experiment_id=exp_id,
            model="BiLSTM_CRF",
            setting="zero_shot",
            annotation_budget=0,
            seed=seed,
            status="SUCCESS",
            run_dir=run_dir,
            validation_macro_f1=val_metrics.get("macro_f1"), # Val from literature checkpoint
            test_macro_f1=summary["macro_f1"],
            test_accuracy=summary["accuracy"],
            test_weighted_f1=summary["weighted_f1"],
            started_at=started_at,
        )
        print(f"Exp 3 Completed: Zero-Shot News Test Macro F1={summary['macro_f1']:.4f}, Accuracy={summary['accuracy']:.4f}")

    except Exception as e:
        print(f"Error in Exp 3: {e}")
        append_registry(
            registry_path,
            experiment_id=exp_id,
            model="BiLSTM_CRF",
            setting="zero_shot",
            annotation_budget=0,
            seed=seed,
            status="FAILED",
            run_dir=run_dir,
            error=str(e),
            started_at=started_at,
        )

    # -------------------------------------------------------------------------
    # Experiment 4: Few-Shot Adaptation (Across Budgets & Seeds)
    # -------------------------------------------------------------------------
    if not skip_few_shot:
        print("\n>>> Running Experiment 4: Few-Shot Adaptation (BiLSTM-CRF)...")
        for s in seeds:
            print(f"\n--- Few-Shot Suite for Seed {s} ---")
            nested_subsets = get_nested_few_shot_subsets(news_pool, budgets=budgets, seed=s)

            for b in budgets:
                sub_df = nested_subsets[b]
                print(f"Fine-tuning budget = {b} sentences (seed {s})...")
                
                exp_id, run_dir = create_hierarchical_run_directory(
                    exp_root, model="BiLSTM_CRF", setting="few_shot", annotation_budget=b, seed=s
                )
                started_at = datetime.now(timezone.utc).isoformat()

                try:
                    set_seed(s)
                    # Load checkpoint from Exp 1 literature in-domain
                    ft_model, ft_vocab, _ = BiLSTMCRF.load(lit_checkpoint_path, device=device)
                    
                    # Adapt on target subset with early stopping on news validation
                    ft_config = copy.deepcopy(config)
                    ft_config["learning_rate"] = config.get("learning_rate", 0.001) * 0.5  # Slightly lower LR for fine-tuning
                    ft_config["epochs"] = 15

                    best_ft_model, ft_val_metrics, _ = train_bilstm_crf(
                        model=ft_model,
                        train_df=sub_df,
                        val_df=news_val,
                        vocabulary=ft_vocab,
                        device=device,
                        run_dir=run_dir,
                        config=ft_config,
                    )

                    # Evaluate on news locked test
                    test_ds = SequenceTaggingDataset(news_test, ft_vocab)
                    test_loader = torch.utils.data.DataLoader(
                        test_ds, batch_size=config.get("batch_size", 32), shuffle=False, collate_fn=collate_sequences
                    )
                    test_preds = decode_bilstm_predictions(best_ft_model, test_loader, device)
                    summary, per_tag, report, conf_mat = evaluate_predictions(news_test, test_preds)

                    save_evaluation_results(
                        run_dir, summary, per_tag, report, conf_mat, test_preds, news_test
                    )

                    append_registry(
                        registry_path,
                        experiment_id=exp_id,
                        model="BiLSTM_CRF",
                        setting="few_shot",
                        annotation_budget=b,
                        seed=s,
                        status="SUCCESS",
                        run_dir=run_dir,
                        validation_macro_f1=ft_val_metrics.get("macro_f1"),
                        test_macro_f1=summary["macro_f1"],
                        test_accuracy=summary["accuracy"],
                        test_weighted_f1=summary["weighted_f1"],
                        started_at=started_at,
                    )
                    print(f"Budget {b} (seed {s}) Completed: Val Macro F1={ft_val_metrics.get('macro_f1'):.4f}, Test Macro F1={summary['macro_f1']:.4f}")

                except Exception as e:
                    print(f"Error in Few-Shot budget {b} (seed {s}): {e}")
                    append_registry(
                        registry_path,
                        experiment_id=exp_id,
                        model="BiLSTM_CRF",
                        setting="few_shot",
                        annotation_budget=b,
                        seed=s,
                        status="FAILED",
                        run_dir=run_dir,
                        error=str(e),
                        started_at=started_at,
                    )

    # -------------------------------------------------------------------------
    # Inference on Unannotated Sanskrit News (3,000 sentences)
    # -------------------------------------------------------------------------
    print("\n>>> Running Unannotated News Inference on 3,000 sentences (BiLSTM-CRF)...")
    unannotated_df = load_unannotated_news(project_root)
    lit_model, lit_vocab, _ = BiLSTMCRF.load(lit_checkpoint_path, device=device)
    
    token_preds, sentence_preds = run_unannotated_inference_bilstm(
        model=lit_model,
        vocabulary=lit_vocab,
        unannotated_df=unannotated_df,
        device=device,
        model_name="BiLSTM_CRF",
        experiment_id="bilstm_literature_source_pretrained",
        annotation_budget=0,
    )
    
    unann_out_dir = project_root / "outputs" / "predictions" / "unannotated_news"
    unann_out_dir.mkdir(parents=True, exist_ok=True)
    token_preds.to_csv(unann_out_dir / "bilstm_unannotated_token_predictions.csv", index=False)
    sentence_preds.to_csv(unann_out_dir / "bilstm_unannotated_sentence_predictions.csv", index=False)
    print(f"Saved unannotated predictions ({len(token_preds)} tokens, {len(sentence_preds)} sentences) to {unann_out_dir}")

    # -------------------------------------------------------------------------
    # Update Tables & Plots
    # -------------------------------------------------------------------------
    reg_df = pd.read_csv(registry_path)
    tables = generate_comparison_tables(reg_df, evals_dir)
    plot_few_shot_curves(reg_df, figures_dir / "few_shot_macro_f1_curve.png")
    print(f"\nGenerated comparison tables and figures under {evals_dir} and {figures_dir}")


def run_transformer_experiments(
    project_root: Path,
    splits: dict[str, pd.DataFrame],
    device: torch.device,
    model_key: str,
    seeds: list[int] = [42, 123, 2026],
    budgets: list[int] = [50, 100, 200, 500, 1000],
    skip_few_shot: bool = False,
) -> None:
    """Execute complete experimental suite for a Transformer model."""
    spec = MODEL_SPECS[model_key]
    display_name = spec["display_name"]
    hf_model_id = spec["hf_model_id"]
    is_iast = spec.get("is_iast", False)

    cfg_path = project_root / spec["config_path"]
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    print("=" * 70)
    print(f"STARTING {display_name} ({hf_model_id}) EXPERIMENTS")
    print(f"IAST Transliteration: {is_iast}")
    print("=" * 70)

    exp_root = project_root / "outputs" / "experiments"
    registry_path = exp_root / "experiment_registry.csv"
    figures_dir = project_root / "outputs" / "figures"
    evals_dir = project_root / "outputs" / "evaluations"

    lit_train = splits["literature_train"]
    lit_val = splits["literature_validation"]
    lit_test = splits["literature_test"]

    news_pool = splits["news_adaptation_pool"]
    news_val = splits["news_validation"]
    news_test = splits["news_test"]

    # -------------------------------------------------------------------------
    # Experiment 1: Literature In-Domain
    # -------------------------------------------------------------------------
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"\n>>> Running Experiment 1: Literature In-Domain ({display_name})...")
    exp_id, run_dir = create_hierarchical_run_directory(
        exp_root, model=display_name, setting="literature_in_domain", seed=seed
    )
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        tagger = TransformerTagger(model_id=hf_model_id, is_iast=is_iast).to(device)
        best_tagger, val_metrics, history_df = train_transformer(
            tagger=tagger,
            train_df=lit_train,
            val_df=lit_val,
            device=device,
            run_dir=run_dir,
            config=config,
        )

        # Evaluate on Literature Test
        test_preds = best_tagger.predict_words(
            list(lit_test["tokens"]),
            device=device,
            batch_size=config.get("batch_size", 16),
            max_length=config.get("max_length", 256),
        )
        summary, per_tag, report, conf_mat = evaluate_predictions(lit_test, test_preds)

        save_evaluation_results(
            run_dir, summary, per_tag, report, conf_mat, test_preds, lit_test
        )
        plot_confusion_matrix_heatmap(
            conf_mat,
            f"{display_name} Literature In-Domain Confusion Matrix",
            figures_dir / f"{model_key}_lit_in_domain_confusion.png",
        )

        append_registry(
            registry_path,
            experiment_id=exp_id,
            model=display_name,
            setting="literature_in_domain",
            annotation_budget=0,
            seed=seed,
            status="SUCCESS",
            run_dir=run_dir,
            validation_macro_f1=val_metrics.get("macro_f1"),
            test_macro_f1=summary["macro_f1"],
            test_accuracy=summary["accuracy"],
            test_weighted_f1=summary["weighted_f1"],
            started_at=started_at,
        )
        print(f"Exp 1 Completed: Val Macro F1={val_metrics.get('macro_f1'):.4f}, Test Macro F1={summary['macro_f1']:.4f}")

        # Save literature source checkpoint
        lit_save_dir = project_root / "outputs" / "models" / model_key / f"{model_key}_source_pretrained"
        best_tagger.save(lit_save_dir, config)
        lit_checkpoint_path = run_dir / "checkpoint"

    except Exception as e:
        print(f"Error in Exp 1: {e}")
        append_registry(
            registry_path,
            experiment_id=exp_id,
            model=display_name,
            setting="literature_in_domain",
            annotation_budget=0,
            seed=seed,
            status="FAILED",
            run_dir=run_dir,
            error=str(e),
            started_at=started_at,
        )
        raise e

    # -------------------------------------------------------------------------
    # Experiment 2: News In-Domain Reference Run
    # -------------------------------------------------------------------------
    print(f"\n>>> Running Experiment 2: News In-Domain Reference Run ({display_name})...")
    set_seed(seed)
    exp_id, run_dir = create_hierarchical_run_directory(
        exp_root, model=display_name, setting="news_in_domain", seed=seed
    )
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        tagger = TransformerTagger(model_id=hf_model_id, is_iast=is_iast).to(device)
        best_news_tagger, news_val_metrics, _ = train_transformer(
            tagger=tagger,
            train_df=news_pool,
            val_df=news_val,
            device=device,
            run_dir=run_dir,
            config=config,
        )

        # Evaluate on News Test Locked
        test_preds = best_news_tagger.predict_words(
            list(news_test["tokens"]),
            device=device,
            batch_size=config.get("batch_size", 16),
            max_length=config.get("max_length", 256),
        )
        summary, per_tag, report, conf_mat = evaluate_predictions(news_test, test_preds)

        save_evaluation_results(
            run_dir, summary, per_tag, report, conf_mat, test_preds, news_test
        )
        plot_confusion_matrix_heatmap(
            conf_mat,
            f"{display_name} News In-Domain Confusion Matrix",
            figures_dir / f"{model_key}_news_in_domain_confusion.png",
        )

        append_registry(
            registry_path,
            experiment_id=exp_id,
            model=display_name,
            setting="news_in_domain",
            annotation_budget=0,
            seed=seed,
            status="SUCCESS",
            run_dir=run_dir,
            validation_macro_f1=news_val_metrics.get("macro_f1"),
            test_macro_f1=summary["macro_f1"],
            test_accuracy=summary["accuracy"],
            test_weighted_f1=summary["weighted_f1"],
            started_at=started_at,
        )
        print(f"Exp 2 Completed: News Val Macro F1={news_val_metrics.get('macro_f1'):.4f}, News Test Macro F1={summary['macro_f1']:.4f}")

    except Exception as e:
        print(f"Error in Exp 2: {e}")
        append_registry(
            registry_path,
            experiment_id=exp_id,
            model=display_name,
            setting="news_in_domain",
            annotation_budget=0,
            seed=seed,
            status="FAILED",
            run_dir=run_dir,
            error=str(e),
            started_at=started_at,
        )

    # -------------------------------------------------------------------------
    # Experiment 3: Zero-Shot Cross-Domain Generalization (Literature -> News)
    # -------------------------------------------------------------------------
    print(f"\n>>> Running Experiment 3: Zero-Shot Cross-Domain Generalization ({display_name})...")
    exp_id, run_dir = create_hierarchical_run_directory(
        exp_root, model=display_name, setting="zero_shot", seed=seed
    )
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        lit_tagger = TransformerTagger.load(lit_checkpoint_path, device=device)
        test_preds = lit_tagger.predict_words(
            list(news_test["tokens"]),
            device=device,
            batch_size=config.get("batch_size", 16),
            max_length=config.get("max_length", 256),
        )
        summary, per_tag, report, conf_mat = evaluate_predictions(news_test, test_preds)

        save_evaluation_results(
            run_dir, summary, per_tag, report, conf_mat, test_preds, news_test
        )
        plot_confusion_matrix_heatmap(
            conf_mat,
            f"{display_name} Zero-Shot Cross-Domain Confusion Matrix",
            figures_dir / f"{model_key}_zero_shot_confusion.png",
        )

        append_registry(
            registry_path,
            experiment_id=exp_id,
            model=display_name,
            setting="zero_shot",
            annotation_budget=0,
            seed=seed,
            status="SUCCESS",
            run_dir=run_dir,
            validation_macro_f1=val_metrics.get("macro_f1"),
            test_macro_f1=summary["macro_f1"],
            test_accuracy=summary["accuracy"],
            test_weighted_f1=summary["weighted_f1"],
            started_at=started_at,
        )
        print(f"Exp 3 Completed: Zero-Shot News Test Macro F1={summary['macro_f1']:.4f}, Accuracy={summary['accuracy']:.4f}")

    except Exception as e:
        print(f"Error in Exp 3: {e}")
        append_registry(
            registry_path,
            experiment_id=exp_id,
            model=display_name,
            setting="zero_shot",
            annotation_budget=0,
            seed=seed,
            status="FAILED",
            run_dir=run_dir,
            error=str(e),
            started_at=started_at,
        )

    # -------------------------------------------------------------------------
    # Experiment 4: Few-Shot Adaptation (Across Budgets & Seeds)
    # -------------------------------------------------------------------------
    if not skip_few_shot:
        print(f"\n>>> Running Experiment 4: Few-Shot Adaptation ({display_name})...")
        for s in seeds:
            print(f"\n--- Few-Shot Suite for Seed {s} ---")
            nested_subsets = get_nested_few_shot_subsets(news_pool, budgets=budgets, seed=s)

            for b in budgets:
                sub_df = nested_subsets[b]
                print(f"Fine-tuning budget = {b} sentences (seed {s})...")
                
                exp_id, run_dir = create_hierarchical_run_directory(
                    exp_root, model=display_name, setting="few_shot", annotation_budget=b, seed=s
                )
                started_at = datetime.now(timezone.utc).isoformat()

                try:
                    set_seed(s)
                    ft_tagger = TransformerTagger.load(lit_checkpoint_path, device=device)
                    
                    ft_config = copy.deepcopy(config)
                    ft_config["learning_rate"] = config.get("learning_rate", 3e-5) * 0.7
                    ft_config["epochs"] = 6
                    ft_config["early_stopping_patience"] = 2

                    best_ft_tagger, ft_val_metrics, _ = train_transformer(
                        tagger=ft_tagger,
                        train_df=sub_df,
                        val_df=news_val,
                        device=device,
                        run_dir=run_dir,
                        config=ft_config,
                    )

                    test_preds = best_ft_tagger.predict_words(
                        list(news_test["tokens"]),
                        device=device,
                        batch_size=config.get("batch_size", 16),
                        max_length=config.get("max_length", 256),
                    )
                    summary, per_tag, report, conf_mat = evaluate_predictions(news_test, test_preds)

                    save_evaluation_results(
                        run_dir, summary, per_tag, report, conf_mat, test_preds, news_test
                    )

                    append_registry(
                        registry_path,
                        experiment_id=exp_id,
                        model=display_name,
                        setting="few_shot",
                        annotation_budget=b,
                        seed=s,
                        status="SUCCESS",
                        run_dir=run_dir,
                        validation_macro_f1=ft_val_metrics.get("macro_f1"),
                        test_macro_f1=summary["macro_f1"],
                        test_accuracy=summary["accuracy"],
                        test_weighted_f1=summary["weighted_f1"],
                        started_at=started_at,
                    )
                    print(f"Budget {b} (seed {s}) Completed: Val Macro F1={ft_val_metrics.get('macro_f1'):.4f}, Test Macro F1={summary['macro_f1']:.4f}")

                except Exception as e:
                    print(f"Error in Few-Shot budget {b} (seed {s}): {e}")
                    append_registry(
                        registry_path,
                        experiment_id=exp_id,
                        model=display_name,
                        setting="few_shot",
                        annotation_budget=b,
                        seed=s,
                        status="FAILED",
                        run_dir=run_dir,
                        error=str(e),
                        started_at=started_at,
                    )

    # -------------------------------------------------------------------------
    # Inference on Unannotated Sanskrit News (3,000 sentences)
    # -------------------------------------------------------------------------
    print(f"\n>>> Running Unannotated News Inference on 3,000 sentences ({display_name})...")
    unannotated_df = load_unannotated_news(project_root)
    lit_tagger = TransformerTagger.load(lit_checkpoint_path, device=device)
    
    token_preds, sentence_preds = run_unannotated_inference_transformer(
        tagger=lit_tagger,
        unannotated_df=unannotated_df,
        device=device,
        model_name=display_name,
        experiment_id=f"{model_key}_literature_source_pretrained",
        annotation_budget=0,
    )
    
    unann_out_dir = project_root / "outputs" / "predictions" / "unannotated_news"
    unann_out_dir.mkdir(parents=True, exist_ok=True)
    token_preds.to_csv(unann_out_dir / f"{model_key}_unannotated_token_predictions.csv", index=False)
    sentence_preds.to_csv(unann_out_dir / f"{model_key}_unannotated_sentence_predictions.csv", index=False)
    print(f"Saved unannotated predictions ({len(token_preds)} tokens, {len(sentence_preds)} sentences) to {unann_out_dir}")

    # -------------------------------------------------------------------------
    # Update Tables & Plots
    # -------------------------------------------------------------------------
    reg_df = pd.read_csv(registry_path)
    tables = generate_comparison_tables(reg_df, evals_dir)
    plot_few_shot_curves(reg_df, figures_dir / "few_shot_macro_f1_curve.png")
    print(f"\nGenerated comparison tables and figures under {evals_dir} and {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Master Sanskrit POS experiment runner")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--model",
        type=str,
        choices=["bilstm_crf", "xlm_roberta", "indicbert_v2", "muril", "sanskrit_bert", "all"],
        default="bilstm_crf",
    )
    parser.add_argument("--skip_few_shot", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    splits = load_dataset_splits(root)
    validate_split_leakage(splits)
    print("Dataset splits loaded and verified successfully (Zero leakage).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}")

    # BiLSTM config
    bilstm_cfg_path = root / "configs" / "bilstm_crf.json"
    with open(bilstm_cfg_path, encoding="utf-8") as f:
        bilstm_cfg = json.load(f)

    if args.model in ("bilstm_crf", "all"):
        run_bilstm_experiments(
            project_root=root,
            splits=splits,
            device=device,
            config=bilstm_cfg,
            seeds=[42, 123, 2026],
            budgets=[50, 100, 200, 500, 1000],
            skip_few_shot=args.skip_few_shot,
        )

    transformer_models = ["xlm_roberta", "indicbert_v2", "muril", "sanskrit_bert"]
    for t_model in transformer_models:
        if args.model in (t_model, "all"):
            run_transformer_experiments(
                project_root=root,
                splits=splits,
                device=device,
                model_key=t_model,
                seeds=[42, 123, 2026],
                budgets=[50, 100, 200, 500, 1000],
                skip_few_shot=args.skip_few_shot,
            )

    print("\nExperiment execution completed successfully!")


if __name__ == "__main__":
    main()
