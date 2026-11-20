from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchcrf import CRF
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .data import TARGET_TAGS


class BiLSTMTagger(nn.Module):
    """Word-level BiLSTM emission model retained for compatibility."""

    def __init__(self, vocabulary: dict[str, int], hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), 128, padding_idx=0)
        self.encoder = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, len(TARGET_TAGS))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.embedding(token_ids))
        return self.classifier(encoded)


class BiLSTMCRF(nn.Module):
    """Padding-aware word-level BiLSTM-CRF sequence tagger.

    ``mask`` must be true for real tokens and false for padding. The CRF
    therefore never scores padding transitions or includes them in decoding.
    """

    def __init__(
        self,
        vocabulary: dict[str, int],
        num_labels: int = len(TARGET_TAGS),
        embedding_dim: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            len(vocabulary), embedding_dim, padding_idx=vocabulary.get("<pad>", 0)
        )
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def emissions(self, token_ids: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.embedding(token_ids))
        return self.classifier(self.dropout(encoded))

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        emissions = self.emissions(token_ids)
        crf_mask = mask.to(dtype=torch.bool)
        if tags is not None:
            return -self.crf(emissions, tags, mask=crf_mask, reduction="mean")
        return self.crf.decode(emissions, mask=crf_mask)


def build_vocabulary(
    sequences: Sequence[Sequence[str]],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build a deterministic vocabulary with explicit padding and unknown IDs."""
    counts: dict[str, int] = {}
    for sequence in sequences:
        for token in sequence:
            counts[token] = counts.get(token, 0) + 1
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token in sorted(token for token, count in counts.items() if count >= min_frequency):
        vocabulary[token] = len(vocabulary)
    return vocabulary


class SequenceTaggingDataset(Dataset):
    """Dataset for padded word-level BiLSTM-CRF batches."""

    def __init__(
        self,
        frame,
        vocabulary: dict[str, int],
        label_to_id: dict[str, int] | None = None,
    ) -> None:
        self.examples = []
        label_to_id = label_to_id or {tag: index for index, tag in enumerate(TARGET_TAGS)}
        for row in frame.itertuples():
            token_ids = [
                vocabulary.get(token, vocabulary["<unk>"]) for token in row.tokens
            ]
            labels = [label_to_id[tag] for tag in row.tags]
            self.examples.append(
                {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "mask": torch.ones(len(token_ids), dtype=torch.bool),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def collate_sequences(
    batch: Sequence[dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Pad a sequence batch while preserving a boolean CRF mask."""
    return {
        "input_ids": pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        ),
        "labels": pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "mask": pad_sequence(
            [item["mask"] for item in batch],
            batch_first=True,
            padding_value=False,
        ),
    }


class WordTokenClassificationDataset(Dataset):
    def __init__(self, frame, tokenizer, label_to_id: dict[str, int], max_length: int = 256):
        self.examples = []
        for row in frame.itertuples():
            encoded = tokenizer(
                list(row.tokens),
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            word_ids = encoded.word_ids()
            labels = [
                -100
                if word_id is None or (
                    index > 0 and word_id == word_ids[index - 1]
                )
                else label_to_id[row.tags[word_id]]
                for index, word_id in enumerate(word_ids)
            ]
            self.examples.append({
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def build_transformer(model_id: str, cache_dir: str | Path | None = None):
    label_to_id = {tag: index for index, tag in enumerate(TARGET_TAGS)}
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(TARGET_TAGS),
        id2label={i: tag for tag, i in label_to_id.items()},
        label2id=label_to_id,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model, label_to_id
