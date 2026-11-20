from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from src.data_preparation.dataset import ID2TAG, load_unannotated_news
from src.models.bilstm_crf import BiLSTMCRF, SequenceTaggingDataset, collate_sequences
from src.models.transformer_tagger import TransformerTagger


def simple_sanskrit_tokenize(text: str) -> list[str]:
    """Tokenize raw Sanskrit sentences by whitespace and punctuation boundaries."""
    # Retain punctuation marks as separate tokens if attached
    import re
    tokens = []
    # Split by whitespace first
    raw_tokens = str(text).strip().split()
    for tok in raw_tokens:
        # Separate trailing Sanskrit danda (। or ॥) or commas/quotes
        sub_tokens = re.findall(r"[\w-]+|[।॥\.,;:!\?\"'()\[\]]", tok)
        if sub_tokens:
            tokens.extend(sub_tokens)
        else:
            tokens.append(tok)
    return tokens


def run_unannotated_inference_bilstm(
    model: BiLSTMCRF,
    vocabulary: dict[str, int],
    unannotated_df: pd.DataFrame,
    device: torch.device,
    model_name: str,
    experiment_id: str,
    annotation_budget: int,
    batch_size: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run inference on unannotated news using BiLSTM-CRF."""
    model.eval()
    
    # Pre-tokenize sentences
    sentence_tokens = []
    for row in unannotated_df.itertuples():
        tokens = simple_sanskrit_tokenize(row.sentence)
        sentence_tokens.append(tokens)

    # Wrap in dummy df for sequence tagging
    eval_frame = pd.DataFrame({
        "doc_id": unannotated_df["doc_id"],
        "sentence_id": unannotated_df["sentence_id"],
        "tokens": sentence_tokens,
        "tags": [["N"] * len(t) for t in sentence_tokens],  # dummy tags
    })

    dataset = SequenceTaggingDataset(eval_frame, vocabulary)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences
    )

    all_preds: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            paths = model(input_ids, mask=mask)
            for path in paths:
                all_preds.append([ID2TAG[idx] for idx in path])

    token_rows = []
    sentence_rows = []

    for row, tokens, preds in zip(unannotated_df.itertuples(), sentence_tokens, all_preds):
        doc_id = row.doc_id
        sentence_id = row.sentence_id
        
        for tid, (tok, ptag) in enumerate(zip(tokens, preds)):
            token_rows.append({
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "token_id": tid,
                "token": tok,
                "predicted_pos": ptag,
                "model_name": model_name,
                "experiment_id": experiment_id,
                "annotation_budget": annotation_budget,
                "confidence": 1.0,
            })

        sentence_rows.append({
            "doc_id": doc_id,
            "sentence_id": sentence_id,
            "sentence": row.sentence,
            "tokens": " ".join(tokens),
            "predicted_tags": " ".join(preds),
            "model_name": model_name,
            "experiment_id": experiment_id,
            "annotation_budget": annotation_budget,
        })

    token_df = pd.DataFrame(token_rows)
    sentence_df = pd.DataFrame(sentence_rows)
    return token_df, sentence_df


def run_unannotated_inference_transformer(
    tagger: TransformerTagger,
    unannotated_df: pd.DataFrame,
    device: torch.device,
    model_name: str,
    experiment_id: str,
    annotation_budget: int,
    batch_size: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run inference on unannotated news using Transformer tagger."""
    sentence_tokens = []
    for row in unannotated_df.itertuples():
        tokens = simple_sanskrit_tokenize(row.sentence)
        sentence_tokens.append(tokens)

    all_preds = tagger.predict_words(
        sentence_tokens, device=device, batch_size=batch_size
    )

    token_rows = []
    sentence_rows = []

    for row, tokens, preds in zip(unannotated_df.itertuples(), sentence_tokens, all_preds):
        doc_id = row.doc_id
        sentence_id = row.sentence_id
        
        for tid, (tok, ptag) in enumerate(zip(tokens, preds)):
            token_rows.append({
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "token_id": tid,
                "token": tok,
                "predicted_pos": ptag,
                "model_name": model_name,
                "experiment_id": experiment_id,
                "annotation_budget": annotation_budget,
                "confidence": 1.0,
            })

        sentence_rows.append({
            "doc_id": doc_id,
            "sentence_id": sentence_id,
            "sentence": row.sentence,
            "tokens": " ".join(tokens),
            "predicted_tags": " ".join(preds),
            "model_name": model_name,
            "experiment_id": experiment_id,
            "annotation_budget": annotation_budget,
        })

    token_df = pd.DataFrame(token_rows)
    sentence_df = pd.DataFrame(sentence_rows)
    return token_df, sentence_df
