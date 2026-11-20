from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_TAGS = [
    "N", "NS", "NV", "KNV", "SN", "SNV", "SNS", "SAM", "SAMY", "SAMP",
    "KP", "UP", "AVN", "AVP", "AVUP", "SAMB", "AVU", "K", "AKV", "AB",
    "AVC", "DVI", "A", "PU",
]


def _sentence_key(row: pd.Series) -> str:
    return f"{row['doc_id']}::{row['sentence_id']}"


def validate_frame(frame: pd.DataFrame, name: str) -> None:
    required = {"tokens", "tags", "sentence_id", "doc_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    invalid = sorted({tag for tags in frame["tags"] for tag in tags if tag not in TARGET_TAGS})
    if invalid:
        raise ValueError(f"{name} contains invalid tags: {invalid}")
    if any(len(row.tokens) != len(row.tags) for row in frame.itertuples()):
        raise ValueError(f"{name} contains token/tag length mismatches")
    keys = frame.apply(_sentence_key, axis=1)
    if keys.duplicated().any():
        raise ValueError(f"{name} contains duplicate sentence keys")


def audit_raw_corpus(
    path: Path,
    *,
    name: str,
    sentence_columns: tuple[str, str],
    token_column: str,
    tag_column: str | None = None,
) -> dict[str, Any]:
    """Audit a token-row CSV without changing or filtering its contents."""
    frame = pd.read_csv(path)
    missing = (set(sentence_columns) | {token_column}) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    sentence_keys = frame[list(sentence_columns)].astype("string").agg("::".join, axis=1)
    token_counts = frame.groupby(list(sentence_columns), dropna=False).size()
    report: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "rows": int(len(frame)),
        "sentences": int(sentence_keys.nunique()),
        "sentence_key_repeated_token_rows": int(
            len(frame) - sentence_keys.nunique()
        ),
        "missing_tokens": int(frame[token_column].isna().sum()),
        "tokens": int(frame[token_column].notna().sum()),
        "duplicate_token_rows": int(frame.duplicated().sum()),
        "max_tokens_per_sentence": int(token_counts.max()) if len(token_counts) else 0,
    }
    if tag_column is not None:
        if tag_column not in frame:
            raise ValueError(f"{name} is missing tag column {tag_column!r}")
        report["missing_tags"] = int(frame[tag_column].isna().sum())
        report["unique_tags"] = int(frame[tag_column].dropna().nunique())
    return report


def create_group_splits(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    group_column: str = "doc_id",
    seed: int = 42,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    prefix: str = "dataset",
) -> dict[str, Path]:
    """Create reproducible train/validation/test splits by group.

    The split is performed on unique group IDs, so no sentence from one
    document can cross a split boundary. The manifest stores the exact IDs.
    """
    if not 0 < train_size < 1 or not 0 < validation_size < 1:
        raise ValueError("train_size and validation_size must be between zero and one")
    if train_size + validation_size >= 1:
        raise ValueError("train_size + validation_size must be less than one")
    if group_column not in frame:
        raise ValueError(f"Missing group column {group_column!r}")
    groups = frame[group_column].astype("string").drop_duplicates().tolist()
    train_groups, remainder_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=seed,
    )
    validation_fraction = validation_size / (1 - train_size)
    validation_groups, test_groups = train_test_split(
        remainder_groups,
        train_size=validation_fraction,
        random_state=seed,
    )
    assignments = {
        "train": set(train_groups),
        "validation": set(validation_groups),
        "test": set(test_groups),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, split_groups in assignments.items():
        split_frame = frame[
            frame[group_column].astype("string").isin(split_groups)
        ].reset_index(drop=True)
        path = output_dir / f"{prefix}_{split}.pkl"
        split_frame.to_pickle(path)
        paths[split] = path
    manifest = {
        "seed": seed,
        "group_column": group_column,
        "requested_proportions": {
            "train": train_size,
            "validation": validation_size,
            "test": 1 - train_size - validation_size,
        },
        "sentence_counts": {
            split: int(len(frame[
                frame[group_column].astype("string").isin(groups_for_split)
            ]))
            for split, groups_for_split in assignments.items()
        },
        "group_ids": {
            split: sorted(groups_for_split) for split, groups_for_split in assignments.items()
        },
    }
    manifest_path = output_dir / f"{prefix}_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    assert_no_overlap(
        *((
            split,
            frame[frame[group_column].astype("string").isin(groups_for_split)],
        ) for split, groups_for_split in assignments.items())
    )
    return paths


def create_literature_test_split(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> dict[str, Path]:
    """Create a held-out literature test split without using target data.

    The existing validation split is preserved. Because the source corpus has
    no document identifier beyond sentence_id, sentence_id is used as the
    grouping key and is recorded explicitly in the split manifest.
    """
    train = pd.read_pickle(train_path).reset_index(drop=True)
    validation = pd.read_pickle(validation_path).reset_index(drop=True)
    validate_frame(train, "literature_train")
    validate_frame(validation, "literature_validation")

    groups = train["doc_id"].astype(str).drop_duplicates().tolist()
    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, random_state=seed
    )
    train_group_set = set(train_groups)
    test_group_set = set(test_groups)
    train_out = train[train["doc_id"].astype(str).isin(train_group_set)].reset_index(drop=True)
    test_out = train[train["doc_id"].astype(str).isin(test_group_set)].reset_index(drop=True)
    if set(train_out["doc_id"].astype(str)) & set(test_out["doc_id"].astype(str)):
        raise AssertionError("Literature train/test document leakage detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "literature_train.pkl",
        "validation": output_dir / "literature_validation.pkl",
        "test": output_dir / "literature_test_locked.pkl",
        "manifest": output_dir / "literature_split_manifest.json",
    }
    train_out.to_pickle(paths["train"])
    validation.to_pickle(paths["validation"])
    test_out.to_pickle(paths["test"])
    manifest = {
        "seed": seed,
        "group_column": "doc_id",
        "source_train_preserved": False,
        "source_validation_preserved": True,
        "train_sentences": len(train_out),
        "validation_sentences": len(validation),
        "test_sentences": len(test_out),
        "train_groups": sorted(map(str, train_group_set)),
        "test_groups": sorted(map(str, test_group_set)),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def load_experiment_frames(root: Path) -> dict[str, pd.DataFrame]:
    final = root / "outputs" / "datasets" / "Final_Literature_Data"
    datasets = root / "outputs" / "datasets"
    frames = {
        "literature_train": pd.read_pickle(final / "literature_train.pkl")
        if (final / "literature_train.pkl").exists()
        else pd.read_pickle(final / "literature_train_final.pkl"),
        "literature_validation": pd.read_pickle(final / "literature_validation.pkl")
        if (final / "literature_validation.pkl").exists()
        else pd.read_pickle(final / "literature_validation_final.pkl"),
        "news_train": pd.read_pickle(datasets / "news_train_pool.pkl"),
        "news_validation": pd.read_pickle(datasets / "news_validation.pkl"),
        "news_test": pd.read_pickle(datasets / "news_test_locked.pkl"),
    }
    for name, frame in frames.items():
        validate_frame(frame, name)
    return frames


def assert_no_overlap(*frames: tuple[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, frame in frames:
        for key in frame.apply(_sentence_key, axis=1):
            previous = seen.setdefault(key, name)
            if previous != name:
                raise ValueError(f"Sentence leakage: {key} appears in {previous} and {name}")
