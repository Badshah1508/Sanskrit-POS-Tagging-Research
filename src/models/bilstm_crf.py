from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchcrf import CRF

from src.data_preparation.dataset import TARGET_TAGS, TAG2ID, ID2TAG


def build_vocabulary(
    sequences: Sequence[Sequence[str]],
    min_frequency: int = 1,
) -> dict[str, dict[str, int]]:
    """Build multi-tier lexical, affix, and character vocabularies."""
    w_counts, s_counts, p_counts, c_counts = {}, {}, {}, {}
    for seq in sequences:
        for w in seq:
            w_counts[w] = w_counts.get(w, 0) + 1
            s = w[-3:] if len(w) >= 3 else w
            p = w[:3] if len(w) >= 3 else w
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1
            for c in w:
                c_counts[c] = c_counts.get(c, 0) + 1

    def make_vocab(counts, min_f=1):
        v = {"<pad>": 0, "<unk>": 1}
        for item, count in sorted(counts.items()):
            if count >= min_f and item not in ["<pad>", "<unk>"]:
                v[item] = len(v)
        return v

    return {
        "word": make_vocab(w_counts, min_frequency),
        "suffix": make_vocab(s_counts, 1),
        "prefix": make_vocab(p_counts, 1),
        "char": make_vocab(c_counts, 1),
    }


class SequenceTaggingDataset(Dataset):
    """Dataset for padded word-level and morphologically-aware BiLSTM-CRF batches."""

    def __init__(
        self,
        frame: Any,
        vocabulary: dict[str, Any],
        label_to_id: dict[str, int] | None = None,
        max_word_len: int = 15,
    ) -> None:
        self.examples = []
        label_to_id = label_to_id or TAG2ID

        if isinstance(vocabulary, dict) and "word" in vocabulary:
            self.w_voc = vocabulary["word"]
            self.s_voc = vocabulary["suffix"]
            self.p_voc = vocabulary["prefix"]
            self.c_voc = vocabulary["char"]
            self.has_morph = True
        else:
            self.w_voc = vocabulary
            self.has_morph = False

        self.max_word_len = max_word_len

        for row in frame.itertuples():
            w_ids = [self.w_voc.get(tok, 1) for tok in row.tokens]
            labels = [label_to_id.get(tag, 0) for tag in row.tags]

            item = {
                "input_ids": torch.tensor(w_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "mask": torch.ones(len(w_ids), dtype=torch.bool),
            }

            if self.has_morph:
                s_ids = [self.s_voc.get(tok[-3:] if len(tok) >= 3 else tok, 1) for tok in row.tokens]
                p_ids = [self.p_voc.get(tok[:3] if len(tok) >= 3 else tok, 1) for tok in row.tokens]

                c_matrix = []
                for tok in row.tokens:
                    c_row = [self.c_voc.get(c, 1) for c in tok[:max_word_len]]
                    if len(c_row) < max_word_len:
                        c_row += [0] * (max_word_len - len(c_row))
                    c_matrix.append(c_row)

                item["suffix_ids"] = torch.tensor(s_ids, dtype=torch.long)
                item["prefix_ids"] = torch.tensor(p_ids, dtype=torch.long)
                item["char_ids"] = torch.tensor(c_matrix, dtype=torch.long)

            self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def collate_sequences(batch: Sequence[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate padded batch of sequences."""
    collated = {
        "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id),
        "labels": pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=0),
        "mask": pad_sequence([item["mask"] for item in batch], batch_first=True, padding_value=False),
    }
    if "suffix_ids" in batch[0]:
        collated["suffix_ids"] = pad_sequence([item["suffix_ids"] for item in batch], batch_first=True, padding_value=pad_id)
        collated["prefix_ids"] = pad_sequence([item["prefix_ids"] for item in batch], batch_first=True, padding_value=pad_id)
        collated["char_ids"] = pad_sequence([item["char_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    return collated


class BiLSTMCRF(nn.Module):
    """Padding-aware word-level and morphologically-aware BiLSTM-CRF sequence tagger."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = len(TARGET_TAGS),
        embedding_dim: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.3,
        pad_idx: int = 0,
        use_morphology: bool = True,
        char_vocab_size: int = 120,
        suffix_vocab_size: int = 1500,
        prefix_vocab_size: int = 1500,
        char_dim: int = 32,
        char_filters: int = 64,
        affix_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout
        self.pad_idx = pad_idx
        self.use_morphology = use_morphology
        self.num_layers = num_layers
        self.char_vocab_size = char_vocab_size
        self.suffix_vocab_size = suffix_vocab_size
        self.prefix_vocab_size = prefix_vocab_size
        self.char_dim = char_dim
        self.char_filters = char_filters
        self.affix_dim = affix_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if use_morphology:
            self.suffix_embedding = nn.Embedding(suffix_vocab_size, affix_dim, padding_idx=0)
            self.prefix_embedding = nn.Embedding(prefix_vocab_size, affix_dim, padding_idx=0)
            self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
            self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
            lstm_in_dim = embedding_dim + (affix_dim * 2) + char_filters
        else:
            lstm_in_dim = embedding_dim

        self.encoder = nn.LSTM(
            lstm_in_dim,
            hidden_size,
            num_layers=num_layers if use_morphology else 1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if (use_morphology and num_layers > 1) else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2) if use_morphology else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def emissions(
        self,
        token_ids: torch.Tensor,
        suffix_ids: torch.Tensor | None = None,
        prefix_ids: torch.Tensor | None = None,
        char_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        if self.use_morphology and suffix_ids is not None and char_ids is not None:
            s_emb = self.suffix_embedding(suffix_ids)
            p_emb = self.prefix_embedding(prefix_ids)
            B, S, C = char_ids.shape
            chars_flat = char_ids.view(B * S, C)
            c_emb = self.char_embedding(chars_flat).transpose(1, 2)
            c_conv = torch.relu(self.char_cnn(c_emb))
            c_pooled, _ = torch.max(c_conv, dim=2)
            c_repr = c_pooled.view(B, S, -1)
            token_repr = torch.cat([embedded, s_emb, p_emb, c_repr], dim=-1)
        else:
            token_repr = embedded

        encoded, _ = self.encoder(token_repr)
        encoded = self.layer_norm(encoded)
        dropped = self.dropout(encoded)
        return self.classifier(dropped)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor | None = None,
        suffix_ids: torch.Tensor | None = None,
        prefix_ids: torch.Tensor | None = None,
        char_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        emissions = self.emissions(token_ids, suffix_ids, prefix_ids, char_ids)
        crf_mask = mask.to(dtype=torch.bool)
        if tags is not None:
            return -self.crf(emissions, tags, mask=crf_mask, reduction="mean")
        return self.crf.decode(emissions, mask=crf_mask)

    def save(self, checkpoint_path: Path, vocabulary: dict[str, Any], config: dict[str, Any]) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "vocabulary": vocabulary,
            "config": config,
            "vocab_size": self.vocab_size,
            "num_labels": self.num_labels,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout_prob,
            "pad_idx": self.pad_idx,
            "use_morphology": self.use_morphology,
            "num_layers": self.num_layers,
            "char_vocab_size": self.char_vocab_size,
            "suffix_vocab_size": self.suffix_vocab_size,
            "prefix_vocab_size": self.prefix_vocab_size,
            "char_dim": self.char_dim,
            "char_filters": self.char_filters,
            "affix_dim": self.affix_dim,
        }
        torch.save(payload, checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> tuple[BiLSTMCRF, dict[str, Any], dict[str, Any]]:
        checkpoint_path = Path(checkpoint_path)
        try:
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=device)
        model = cls(
            vocab_size=payload["vocab_size"],
            num_labels=payload["num_labels"],
            embedding_dim=payload["embedding_dim"],
            hidden_size=payload["hidden_size"],
            dropout=payload["dropout"],
            pad_idx=payload["pad_idx"],
            use_morphology=payload.get("use_morphology", True),
            num_layers=payload.get("num_layers", 2),
            char_vocab_size=payload.get("char_vocab_size", 120),
            suffix_vocab_size=payload.get("suffix_vocab_size", 1500),
            prefix_vocab_size=payload.get("prefix_vocab_size", 1500),
            char_dim=payload.get("char_dim", 32),
            char_filters=payload.get("char_filters", 64),
            affix_dim=payload.get("affix_dim", 32),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, payload["vocabulary"], payload.get("config", {})
