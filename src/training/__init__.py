"""Training package for Sanskrit POS models."""
from .trainer import (
    train_bilstm_crf,
    train_transformer,
)

__all__ = [
    "train_bilstm_crf",
    "train_transformer",
]
