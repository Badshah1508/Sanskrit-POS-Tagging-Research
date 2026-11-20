from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data_preparation.dataset import TARGET_TAGS, TAG2ID, ID2TAG
from src.evaluation.evaluator import evaluate_predictions
from src.models.bilstm_crf import (
    BiLSTMCRF,
    SequenceTaggingDataset,
    collate_sequences,
)
from src.models.transformer_tagger import (
    TransformerTagger,
    WordTokenClassificationDataset,
)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def decode_bilstm_predictions(
    model: BiLSTMCRF,
    loader: DataLoader,
    device: torch.device,
) -> list[list[str]]:
    """Viterbi-decode word predictions excluding padding."""
    model.eval()
    predictions: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            best_paths = model(input_ids, mask=mask)
            for path in best_paths:
                predictions.append([ID2TAG[idx] for idx in path])
    return predictions


def train_bilstm_crf(
    model: BiLSTMCRF,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocabulary: dict[str, int],
    device: torch.device,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[BiLSTMCRF, dict[str, Any], pd.DataFrame]:
    """Train BiLSTM-CRF with validation Macro F1 early stopping and logging."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "training.log")

    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 1e-4)
    patience = config.get("early_stopping_patience", 3)

    logger.info(f"Starting BiLSTM-CRF training on {device} for {epochs} epochs")
    logger.info(f"Train sentences: {len(train_df)}, Validation sentences: {len(val_df)}")

    train_ds = SequenceTaggingDataset(train_df, vocabulary)
    val_ds = SequenceTaggingDataset(val_df, vocabulary)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    patience_counter = 0
    history_records = []

    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, mask=mask, tags=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Validation evaluation
        val_preds = decode_bilstm_predictions(model, val_loader, device)
        val_metrics, _, _, _ = evaluate_predictions(val_df, val_preds)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_wf1 = val_metrics["weighted_f1"]

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f} - "
            f"Val Macro F1: {val_f1:.4f} - Val Weighted F1: {val_wf1:.4f}"
        )

        history_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_weighted_f1": val_wf1,
            "duration_seconds": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            model.save(checkpoint_dir / "model.pt", vocabulary, config)
            logger.info(f"==> New best validation Macro F1: {best_val_f1:.4f} (Saved checkpoint)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    # Load best checkpoint
    best_model, _, _ = BiLSTMCRF.load(checkpoint_dir / "model.pt", device=device)
    logger.info(f"Training finished. Best epoch: {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

    return best_model, best_metrics, history_df


def train_transformer(
    tagger: TransformerTagger,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[TransformerTagger, dict[str, Any], pd.DataFrame]:
    """Train transformer tagger with word-level validation Macro F1 early stopping."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "training.log")

    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 8)
    lr = config.get("learning_rate", 3e-5)
    weight_decay = config.get("weight_decay", 0.01)
    max_length = config.get("max_length", 256)
    warmup_ratio = config.get("warmup_ratio", 0.1)
    patience = config.get("early_stopping_patience", 2)

    logger.info(f"Starting Transformer training ({tagger.model_id}) on {device}")
    logger.info(f"Train sentences: {len(train_df)}, Validation sentences: {len(val_df)}")

    train_ds = WordTokenClassificationDataset(
        train_df, tagger.tokenizer, max_length=max_length, is_iast=getattr(tagger, "is_iast", False)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(
        tagger.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    patience_counter = 0
    history_records = []

    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tagger.model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = tagger.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Word-level validation evaluation
        val_tokens_list = list(val_df["tokens"])
        val_preds = tagger.predict_words(
            val_tokens_list, device=device, batch_size=batch_size, max_length=max_length
        )
        val_metrics, _, _, _ = evaluate_predictions(val_df, val_preds)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_wf1 = val_metrics["weighted_f1"]

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f} - "
            f"Val Macro F1: {val_f1:.4f} - Val Weighted F1: {val_wf1:.4f}"
        )

        history_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_weighted_f1": val_wf1,
            "duration_seconds": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            tagger.save(checkpoint_dir, config)
            logger.info(f"==> New best validation Macro F1: {best_val_f1:.4f} (Saved checkpoint)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    best_tagger = TransformerTagger.load(checkpoint_dir, device=device)
    logger.info(f"Training finished. Best epoch: {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

    return best_tagger, best_metrics, history_df
