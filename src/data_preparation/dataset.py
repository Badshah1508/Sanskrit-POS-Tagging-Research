from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

TARGET_TAGS: list[str] = [
    "N", "NS", "NV", "KNV", "SN", "SNV", "SNS", "SAM", "SAMY", "SAMP",
    "KP", "UP", "AVN", "AVP", "AVUP", "SAMB", "AVU", "K", "AKV", "AB",
    "AVC", "DVI", "A", "PU",
]

TAG2ID: dict[str, int] = {tag: idx for idx, tag in enumerate(TARGET_TAGS)}
ID2TAG: dict[int, str] = {idx: tag for idx, tag in enumerate(TARGET_TAGS)}


def sentence_key(row: Any) -> str:
    """Generate composite sentence key: doc_id::sentence_id."""
    doc_id = getattr(row, "doc_id", None)
    sentence_id = getattr(row, "sentence_id", None)
    if doc_id is None and isinstance(row, dict):
        doc_id = row.get("doc_id", row.get("sentence_id"))
        sentence_id = row.get("sentence_id")
    elif doc_id is None and isinstance(row, pd.Series):
        doc_id = row["doc_id"] if "doc_id" in row else row["sentence_id"]
        sentence_id = row["sentence_id"]
    return f"{doc_id}::{sentence_id}"


def load_dataset_splits(project_root: Path) -> dict[str, pd.DataFrame]:
    """Load and validate all standard experiment splits.
    
    Returns a dictionary of DataFrames where each row is a sentence:
      - doc_id: Any
      - sentence_id: Any
      - tokens: list[str]
      - tags: list[str]
    """
    root = Path(project_root)
    lit_dir = root / "outputs" / "datasets" / "Final_Literature_Data"
    datasets_dir = root / "outputs" / "datasets"

    def _load_split(name: str, folder: Path, basename: str, csv_fallback: str | None = None) -> pd.DataFrame:
        json_path = folder / f"{basename}.json"
        pkl_path = folder / f"{basename}.pkl"

        # 1. Primary: JSON (version-independent)
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return pd.DataFrame(json.load(f))
            except Exception:
                pass

        # 2. Secondary: Pickle
        if pkl_path.is_file():
            try:
                return pd.read_pickle(pkl_path)
            except Exception:
                pass

        # 3. Tertiary: CSV Fallback
        search_dirs = [folder, folder.parent, root / "outputs" / "datasets"]
        for sdir in search_dirs:
            if csv_fallback and (sdir / csv_fallback).is_file():
                df_csv = pd.read_csv(sdir / csv_fallback)
                tag_col = "tag" if "tag" in df_csv.columns else "gold_sanskrit_tag"
                grouped = df_csv.groupby(["doc_id", "sentence_id"], sort=False).agg({
                    "token": list,
                    tag_col: list
                }).reset_index().rename(columns={"token": "tokens", tag_col: "tags"})
                return grouped

        raise FileNotFoundError(f"Could not load split '{name}' from {folder}")

    splits = {
        "literature_train": _load_split("literature_train", lit_dir, "literature_train", "literature_train.csv"),
        "literature_validation": _load_split("literature_validation", lit_dir, "literature_validation", "literature_validation.csv"),
        "literature_test": _load_split("literature_test", lit_dir, "literature_test_locked", "literature_test_gold.csv"),
        "news_adaptation_pool": _load_split("news_adaptation_pool", datasets_dir, "news_train_pool", "news_adaptation_pool.csv"),
        "news_validation": _load_split("news_validation", datasets_dir, "news_validation", "news_validation.csv"),
        "news_test": _load_split("news_test", datasets_dir, "news_test_locked", "news_test_gold.csv"),
    }

    for name, df in splits.items():
        validate_split_frame(df, name)

    return splits


def load_unannotated_news(project_root: Path) -> pd.DataFrame:
    """Load the 3,000 unannotated news sentences."""
    path = Path(project_root) / "data" / "raw" / "News_Unannotated.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Unannotated news dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def validate_split_frame(frame: pd.DataFrame, name: str) -> None:
    """Assert frame schema and tag validity."""
    required = {"tokens", "tags", "sentence_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    
    # Check invalid tags
    invalid = sorted({tag for tags in frame["tags"] for tag in tags if tag not in TARGET_TAGS})
    if invalid:
        raise ValueError(f"{name} contains invalid target tags: {invalid}")
    
    # Check token/tag length parity
    for row in frame.itertuples():
        if len(row.tokens) != len(row.tags):
            raise ValueError(f"{name} sentence {row.sentence_id} has length mismatch: {len(row.tokens)} tokens vs {len(row.tags)} tags")


def validate_split_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Verify zero sentence and zero document leakage across splits."""
    # Literature domain check
    lit_seen: dict[str, str] = {}
    for split_name in ["literature_train", "literature_validation", "literature_test"]:
        if split_name in splits:
            for row in splits[split_name].itertuples():
                key = sentence_key(row)
                if key in lit_seen:
                    raise AssertionError(f"Literature sentence leakage: {key} in {lit_seen[key]} and {split_name}")
                lit_seen[key] = split_name

    # News domain document-level check
    news_docs: dict[Any, str] = {}
    for split_name in ["news_adaptation_pool", "news_validation", "news_test"]:
        if split_name in splits:
            for doc in splits[split_name]["doc_id"].unique():
                if doc in news_docs:
                    raise AssertionError(f"News document leakage: {doc} in {news_docs[doc]} and {split_name}")
                news_docs[doc] = split_name


def get_nested_few_shot_subsets(
    pool_df: pd.DataFrame,
    budgets: Sequence[int] = (0, 50, 100, 200, 500, 1000),
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Sample strictly nested few-shot subsets from the adaptation pool.
    
    For a fixed seed, the 50-sentence subset is strictly contained within
    the 100, 200, 500, and 1,000 subsets.
    """
    sorted_budgets = sorted(set(budgets))
    max_budget = sorted_budgets[-1] if sorted_budgets else 0
    if max_budget > len(pool_df):
        raise ValueError(f"Max budget {max_budget} exceeds pool size {len(pool_df)}")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(pool_df))

    subsets: dict[int, pd.DataFrame] = {}
    for b in sorted_budgets:
        if b == 0:
            subsets[0] = pool_df.iloc[:0].copy().reset_index(drop=True)
        else:
            subset_indices = indices[:b]
            subsets[b] = pool_df.iloc[subset_indices].copy().reset_index(drop=True)

    return subsets
