from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REGISTRY_COLUMNS = [
    "experiment_id",
    "model",
    "setting",
    "annotation_budget",
    "seed",
    "status",
    "run_dir",
    "validation_macro_f1",
    "test_macro_f1",
    "test_accuracy",
    "test_weighted_f1",
    "error",
    "started_at",
    "completed_at",
]


def create_hierarchical_run_directory(
    experiments_root: Path,
    *,
    model: str,
    setting: str,
    annotation_budget: int = 0,
    seed: int = 42,
) -> tuple[str, Path]:
    """Create a structured, collision-resistant run directory adhering to the dissertation spec.
    
    Structure:
      experiments/
        <Model>/
          <setting>/
            [<budget>/]
              [seed_<seed>/]
                run_<uuid>/
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_token = uuid.uuid4().hex[:6]
    experiment_id = f"{model}_{setting}_{annotation_budget}sent_s{seed}_{timestamp}_{run_token}"

    if setting == "few_shot":
        run_dir = experiments_root / model / setting / str(annotation_budget) / f"seed_{seed}" / f"run_{timestamp}_{run_token}"
    elif setting in ("literature_in_domain", "news_in_domain", "zero_shot"):
        run_dir = experiments_root / model / setting / f"run_{timestamp}_{run_token}"
    else:
        run_dir = experiments_root / model / setting / f"run_{timestamp}_{run_token}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return experiment_id, run_dir


def append_registry(
    registry_path: Path,
    *,
    experiment_id: str,
    model: str,
    setting: str,
    annotation_budget: int,
    seed: int,
    status: str,
    run_dir: Path,
    validation_macro_f1: float | None = None,
    test_macro_f1: float | None = None,
    test_accuracy: float | None = None,
    test_weighted_f1: float | None = None,
    error: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Append one run record and update the best model index."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment_id": experiment_id,
        "model": model,
        "setting": setting,
        "annotation_budget": annotation_budget,
        "seed": seed,
        "status": status,
        "run_dir": str(run_dir),
        "validation_macro_f1": validation_macro_f1,
        "test_macro_f1": test_macro_f1,
        "test_accuracy": test_accuracy,
        "test_weighted_f1": test_weighted_f1,
        "error": error,
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
    }
    
    exists = registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    # Update best_model_index.csv
    update_best_model_index(registry_path)


def update_best_model_index(registry_path: Path) -> None:
    """Compute and persist the best model index based on validation Macro F1."""
    if not registry_path.exists():
        return
    try:
        df = pd.read_csv(registry_path)
        successful = df[df["status"] == "SUCCESS"].dropna(subset=["validation_macro_f1"])
        if successful.empty:
            return
        
        # Best model per (model, setting, annotation_budget)
        best_indices = successful.groupby(["model", "setting", "annotation_budget"])["validation_macro_f1"].idxmax()
        best_df = successful.loc[best_indices].reset_index(drop=True)
        best_index_path = registry_path.parent / "best_model_index.csv"
        best_df.to_csv(best_index_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to update best model index: {e}")
