from __future__ import annotations

import argparse
from pathlib import Path

from .data import assert_no_overlap, create_literature_test_split, load_experiment_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    final = root / "outputs" / "datasets" / "Final_Literature_Data"
    paths = create_literature_test_split(
        final / "literature_train_final.pkl",
        final / "literature_validation_final.pkl",
        final,
    )
    frames = load_experiment_frames(root)
    import pandas as pd

    frames["literature_train"] = pd.read_pickle(paths["train"])
    frames["literature_validation"] = pd.read_pickle(paths["validation"])
    literature_test = pd.read_pickle(paths["test"])
    assert_no_overlap(
        ("literature_train", frames["literature_train"]),
        ("literature_validation", frames["literature_validation"]),
        ("literature_test", literature_test),
    )
    assert_no_overlap(
        ("news_train", frames["news_train"]),
        ("news_validation", frames["news_validation"]),
        ("news_test", frames["news_test"]),
    )
    print("Prepared and validated literature test split:", paths["test"])
    print("No sentence/document-key overlap across experiment splits.")


if __name__ == "__main__":
    main()
