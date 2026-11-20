"""Evaluation and metrics package for Sanskrit POS Cross-Domain experiments."""
from .evaluator import (
    evaluate_predictions,
    save_evaluation_results,
)

__all__ = [
    "evaluate_predictions",
    "save_evaluation_results",
]
