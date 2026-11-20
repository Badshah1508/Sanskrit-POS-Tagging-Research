from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.data_preparation.dataset import TARGET_TAGS, TAG2ID, ID2TAG


def devanagari_to_iast(text: str) -> str:
    """Accurately transliterates Devanagari Sanskrit text to IAST Latin script.
    
    Preserves 1:1 token boundary alignment so word-level supervision
    remains strictly identical to Devanagari models.
    Used for Latin/IAST-pretrained models like Matej/bert-base-buddhist-sanskrit-v2 (M5).
    """
    vowels = {
        'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
        'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ॡ': 'ḹ', 'ए': 'e', 'ऐ': 'ai',
        'ओ': 'o', 'औ': 'au'
    }
    consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ṅ',
        'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'ñ',
        'ट': 'ṭ', 'ठ': 'ṭh', 'ड': 'ḍ', 'ढ': 'ḍh', 'ण': 'ṇ',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
        'श': 'ś', 'ष': 'ṣ', 'स': 's', 'ह': 'h'
    }
    matras = {
        'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū',
        'ृ': 'ṛ', 'ॄ': 'ṝ', 'ॢ': 'ḷ', 'ॣ': 'ḹ', 'े': 'e', 'ै': 'ai',
        'ो': 'o', 'ौ': 'au'
    }
    specials = {
        'ं': 'ṁ', 'ः': 'ḥ', 'ँ': 'm̐', 'ऽ': "'", '।': '.', '॥': '..',
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
    }

    res = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in consonants:
            base = consonants[c]
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt == '्':
                    res.append(base)
                    i += 2
                    continue
                elif nxt in matras:
                    res.append(base + matras[nxt])
                    i += 2
                    continue
                else:
                    res.append(base + 'a')
                    i += 1
                    continue
            else:
                res.append(base + 'a')
                i += 1
                continue
        elif c in vowels:
            res.append(vowels[c])
            i += 1
        elif c in matras:
            res.append(matras[c])
            i += 1
        elif c in specials:
            res.append(specials[c])
            i += 1
        elif c == '्':
            i += 1
        else:
            res.append(c)
            i += 1
    return ''.join(res)


class WordTokenClassificationDataset(Dataset):
    """Dataset with strict word-to-subword alignment for Sanskrit tokens.
    
    The first subword token of each Sanskrit word receives its target tag ID.
    Subsequent subword tokens and special tokens receive label -100 (ignored in loss).
    When is_iast=True, words are transliterated to IAST before tokenization,
    preserving exact 1:1 word index alignment.
    """

    def __init__(
        self,
        frame: Any,
        tokenizer: PreTrainedTokenizerBase,
        label_to_id: dict[str, int] | None = None,
        max_length: int = 256,
        is_labeled: bool = True,
        is_iast: bool = False,
    ) -> None:
        self.examples = []
        self.raw_tokens = []
        label_to_id = label_to_id or TAG2ID
        self.is_iast = is_iast
        
        for row in frame.itertuples():
            tokens = list(row.tokens)
            self.raw_tokens.append(tokens)
            tokens_to_tokenize = [devanagari_to_iast(t) for t in tokens] if is_iast else tokens
            encoded = tokenizer(
                tokens_to_tokenize,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            word_ids = encoded.word_ids()
            
            if is_labeled and hasattr(row, "tags") and row.tags is not None:
                labels = []
                for index, word_id in enumerate(word_ids):
                    if word_id is None:
                        labels.append(-100)
                    elif index > 0 and word_id == word_ids[index - 1]:
                        # Subsequent subword of the same word
                        labels.append(-100)
                    else:
                        # First subword of word_id
                        labels.append(label_to_id[row.tags[word_id]])
            else:
                labels = [-100] * len(encoded["input_ids"])

            self.examples.append({
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "word_ids": word_ids,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


class TransformerTagger:
    """Wrapper around HuggingFace AutoModelForTokenClassification."""

    def __init__(
        self,
        model_id: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model: PreTrainedModel | None = None,
        cache_dir: Path | None = None,
        is_iast: bool | None = None,
    ) -> None:
        self.model_id = model_id
        if is_iast is not None:
            self.is_iast = is_iast
        else:
            self.is_iast = "buddhist-sanskrit" in model_id.lower() or "sanskrit_bert" in model_id.lower() or "sanskrit-bert" in model_id.lower()

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model = model or AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=len(TARGET_TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

    def to(self, device: torch.device) -> TransformerTagger:
        self.model.to(device)
        return self

    def predict_words(
        self,
        sentences: Sequence[Sequence[str]],
        device: torch.device,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> list[list[str]]:
        """Run batch inference and reconstruct word-level POS predictions."""
        self.model.eval()
        predictions: list[list[str]] = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_words = sentences[i : i + batch_size]
                if self.is_iast:
                    batch_to_encode = [[devanagari_to_iast(w) for w in words] for words in batch_words]
                else:
                    batch_to_encode = list(batch_words)

                encoded = self.tokenizer(
                    batch_to_encode,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                outputs = self.model(**encoded)
                logits = outputs.logits  # [batch_size, seq_len, num_labels]
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                for j, words in enumerate(batch_words):
                    word_ids = encoded.word_ids(batch_index=j)
                    sent_preds: list[str] = []
                    seen_word_ids = set()

                    for token_idx, word_id in enumerate(word_ids):
                        if word_id is not None and word_id not in seen_word_ids:
                            pred_tag_id = int(preds[j, token_idx])
                            sent_preds.append(ID2TAG[pred_tag_id])
                            seen_word_ids.add(word_id)

                    # In case of extreme truncation, pad remaining words with default N
                    while len(sent_preds) < len(words):
                        sent_preds.append("N")

                    predictions.append(sent_preds)

        return predictions

    def save(self, output_dir: Path, config: dict[str, Any] | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        meta = {"is_iast": self.is_iast, "model_id": self.model_id}
        (output_dir / "tagger_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if config:
            (output_dir / "experiment_config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
        device: torch.device = torch.device("cpu"),
        is_iast: bool | None = None,
    ) -> TransformerTagger:
        checkpoint_dir = Path(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        model.to(device)
        meta_file = checkpoint_dir / "tagger_meta.json"
        if is_iast is None and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                is_iast = meta.get("is_iast", False)
            except Exception:
                pass
        return cls(model_id=str(checkpoint_dir), tokenizer=tokenizer, model=model, is_iast=is_iast)


def build_transformer_pipeline(
    model_id: str,
    cache_dir: Path | None = None,
    is_iast: bool | None = None,
) -> tuple[PreTrainedTokenizerBase, AutoModelForTokenClassification]:
    tagger = TransformerTagger(model_id=model_id, cache_dir=cache_dir, is_iast=is_iast)
    return tagger.tokenizer, tagger.model
