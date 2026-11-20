"""Pre-implementation verification script for M4 (MuRIL) and M5 (Sanskrit-BERT).

Verifies:
1. Checkpoint loading (AutoTokenizer, AutoModelForTokenClassification)
2. Tokenizer compatibility with actual Sanskrit dataset script (Devanagari vs IAST)
3. Vocabulary coverage and tokenization quality (fragmentation rate, UNK tokens, sample words)
4. Token-level fine-tuning support (forward pass, loss with -100 ignore_index, backward pass, optimizer step)
5. GPU memory requirements (forward + backward pass with sequence length 256 at batch sizes 8 and 16)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_preparation.dataset import TARGET_TAGS, TAG2ID, ID2TAG, load_dataset_splits
from src.models.transformer_tagger import devanagari_to_iast, WordTokenClassificationDataset, TransformerTagger


def run_precheck():
    print("=" * 80)
    print("PRE-IMPLEMENTATION VERIFICATION FOR M4 (MuRIL) AND M5 (Sanskrit-BERT)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    splits = load_dataset_splits(PROJECT_DIR)
    lit_train = splits["literature_train"]
    lit_test = splits["literature_test"]
    news_test = splits["news_test"]
    print(f"Loaded datasets: Lit Train={len(lit_train)}, Lit Test={len(lit_test)}, News Test={len(news_test)}")

    models_info = [
        {
            "id": "M4",
            "name": "MuRIL",
            "checkpoint": "google/muril-base-cased",
            "is_iast": False,
            "description": "Multilingual Representations for Indian Languages (Native Devanagari)"
        },
        {
            "id": "M5",
            "name": "Sanskrit-BERT",
            "checkpoint": "Matej/bert-base-buddhist-sanskrit-v2",
            "is_iast": True,
            "description": "Buddhist Sanskrit BERT (IAST transliterated with 1:1 token alignment)"
        }
    ]

    report_data = {}

    for m_info in models_info:
        m_id = m_info["id"]
        m_name = m_info["name"]
        ckpt = m_info["checkpoint"]
        is_iast = m_info["is_iast"]

        print(f"\n{'='*75}")
        print(f"VERIFYING {m_id}: {m_name} ({ckpt})")
        print(f"Script Mode: {'IAST (Transliterated)' if is_iast else 'Devanagari (Native)'}")
        print(f"{'='*75}")

        m_report = {}

        # ---------------------------------------------------------------------
        # 1. Checkpoint Loading
        # ---------------------------------------------------------------------
        print("\n[CHECK 1] Checkpoint Loading...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForTokenClassification.from_pretrained(
            ckpt,
            num_labels=len(TARGET_TAGS),
            id2label=ID2TAG,
            label2id=TAG2ID,
            ignore_mismatched_sizes=True,
        )
        load_time = time.time() - t0
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Loaded successfully in {load_time:.2f}s")
        print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Hidden Size: {model.config.hidden_size}, Num Layers: {model.config.num_hidden_layers}")
        print(f"Vocab Size in Config: {model.config.vocab_size}, Tokenizer Vocab Size: {len(tokenizer)}")
        m_report["checkpoint_loaded"] = True
        m_report["total_params"] = total_params
        m_report["vocab_size"] = len(tokenizer)

        # ---------------------------------------------------------------------
        # 2. Tokenizer Compatibility with Dataset Script & Encoding
        # ---------------------------------------------------------------------
        print("\n[CHECK 2] Tokenizer Compatibility with Sanskrit Script...")
        sample_lit_sentence = lit_test.iloc[0]["tokens"]
        sample_news_sentence = news_test.iloc[0]["tokens"]
        print(f"Sample Lit Sentence: {' '.join(sample_lit_sentence[:8])}...")
        print(f"Sample News Sentence: {' '.join(sample_news_sentence[:8])}...")

        # Test raw Devanagari behavior
        raw_encoded = tokenizer(sample_lit_sentence, is_split_into_words=True)
        raw_tokens = tokenizer.convert_ids_to_tokens(raw_encoded["input_ids"])
        raw_unk_count = raw_tokens.count(tokenizer.unk_token)
        print(f"Raw Devanagari Input -> Total subwords: {len(raw_tokens)}, UNK count: {raw_unk_count}")

        if is_iast:
            # Test IAST transliteration behavior
            iast_words = [devanagari_to_iast(w) for w in sample_lit_sentence]
            print(f"Transliterated to IAST: {' '.join(iast_words[:8])}...")
            iast_encoded = tokenizer(iast_words, is_split_into_words=True)
            iast_tokens = tokenizer.convert_ids_to_tokens(iast_encoded["input_ids"])
            iast_unk_count = iast_tokens.count(tokenizer.unk_token)
            print(f"IAST Input -> Total subwords: {len(iast_tokens)}, UNK count: {iast_unk_count}")
            print(f"Subwords preview: {iast_tokens[:15]}")
            m_report["devanagari_unk_count"] = raw_unk_count
            m_report["iast_unk_count"] = iast_unk_count
            m_report["requires_transliteration"] = True
            m_report["transliteration_compatible"] = (iast_unk_count == 0)
        else:
            print(f"Subwords preview: {raw_tokens[:15]}")
            m_report["devanagari_unk_count"] = raw_unk_count
            m_report["requires_transliteration"] = False
            m_report["devanagari_compatible"] = (raw_unk_count == 0)

        # ---------------------------------------------------------------------
        # 3. Vocabulary Coverage & Tokenization Quality
        # ---------------------------------------------------------------------
        print("\n[CHECK 3] Vocabulary Coverage and Tokenization Quality across 100 sentences...")
        # Inspect on 100 literature sentences and 100 news sentences
        test_dfs = [("Literature Held-Out Test", lit_test.head(100)), ("News Locked Test", news_test.head(100))]
        coverage_stats = {}

        for corpus_name, df_subset in test_dfs:
            total_words = 0
            total_subwords = 0
            total_unks = 0

            for words in df_subset["tokens"]:
                total_words += len(words)
                words_to_tok = [devanagari_to_iast(w) for w in words] if is_iast else list(words)
                enc = tokenizer(words_to_tok, is_split_into_words=True, add_special_tokens=False)
                subwords = tokenizer.convert_ids_to_tokens(enc["input_ids"])
                total_subwords += len(subwords)
                total_unks += subwords.count(tokenizer.unk_token)

            subwords_per_word = total_subwords / max(total_words, 1)
            unk_rate = (total_unks / max(total_subwords, 1)) * 100.0
            coverage_stats[corpus_name] = {
                "total_words": total_words,
                "total_subwords": total_subwords,
                "subwords_per_word": subwords_per_word,
                "total_unks": total_unks,
                "unk_rate_pct": unk_rate,
            }
            print(f"  {corpus_name}: {total_words} words -> {total_subwords} subwords ({subwords_per_word:.2f} subwords/word), UNKs: {total_unks} ({unk_rate:.3f}%)")

        m_report["coverage_stats"] = coverage_stats

        # Sample word tokenization audit
        sample_words = ["धर्मक्षेत्रे", "कुरुक्षेत्रे", "प्रतिवर्षं", "संस्कृतभारती", "विश्वविद्यालयस्य"]
        print("  Sample morphological tokenizations:")
        sample_audit = []
        for w in sample_words:
            w_input = devanagari_to_iast(w) if is_iast else w
            enc = tokenizer(w_input, add_special_tokens=False)
            subs = tokenizer.convert_ids_to_tokens(enc["input_ids"])
            sample_audit.append({"word": w, "input": w_input, "subwords": subs})
            print(f"    '{w}' -> {subs}")
        m_report["sample_audit"] = sample_audit

        # ---------------------------------------------------------------------
        # 4. Token-Level Fine-Tuning Support & Loss Computation
        # ---------------------------------------------------------------------
        print("\n[CHECK 4] Token-Level Fine-Tuning Support (Forward + Backward Pass)...")
        model.to(device)
        model.train()

        # Build actual dataset slice using WordTokenClassificationDataset
        train_slice = lit_train.head(8)
        dataset_slice = WordTokenClassificationDataset(
            train_slice, tokenizer, label_to_id=TAG2ID, max_length=256, is_iast=is_iast
        )
        batch = [dataset_slice[i] for i in range(4)]
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
        labels = torch.stack([b["labels"] for b in batch]).to(device)

        # Check label mask distribution
        active_labels = (labels != -100).sum().item()
        ignored_labels = (labels == -100).sum().item()
        print(f"  Batch shape: {input_ids.shape}, Active labeled tokens: {active_labels}, Ignored (-100): {ignored_labels}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        print(f"  Forward Pass Successful: Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
        assert logits.shape == (4, 256, len(TARGET_TAGS)), f"Unexpected logits shape: {logits.shape}"
        assert not torch.isnan(loss), "Loss is NaN!"

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"  Backward Pass & Optimizer Step Successful: Grad Norm: {grad_norm.item():.4f}")
        m_report["finetuning_supported"] = True
        m_report["loss_computed"] = loss.item()
        m_report["grad_norm"] = grad_norm.item()

        # ---------------------------------------------------------------------
        # 5. GPU Memory Requirements Check
        # ---------------------------------------------------------------------
        print("\n[CHECK 5] GPU Memory Benchmarking (Seq Len 256)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            for bs in [8, 16]:
                dummy_input = torch.randint(0, len(tokenizer), (bs, 256), device=device)
                dummy_mask = torch.ones((bs, 256), dtype=torch.long, device=device)
                dummy_labels = torch.randint(0, len(TARGET_TAGS), (bs, 256), device=device)
                dummy_labels[:, 128:] = -100  # Partial mask

                torch.cuda.reset_peak_memory_stats()
                t_start = time.time()
                outputs = model(input_ids=dummy_input, attention_mask=dummy_mask, labels=dummy_labels)
                outputs.loss.backward()
                elapsed = time.time() - t_start

                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)

                print(f"  Batch Size {bs:2d}: Peak Allocated: {peak_allocated:.1f} MB, Reserved: {reserved:.1f} MB, Step time: {elapsed*1000:.1f} ms")
                m_report[f"gpu_mem_bs_{bs}_mb"] = peak_allocated
        else:
            print("  CPU Execution: Peak Memory Check N/A")

        report_data[m_id] = m_report

    out_file = PROJECT_DIR / "outputs" / "logs" / "m4_m5_precheck_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nVerification summary saved to: {out_file}")
    print("=" * 80)
    print("ALL PRE-IMPLEMENTATION VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_precheck()
