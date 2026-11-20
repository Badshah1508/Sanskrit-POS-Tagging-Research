# Dataset Validation Report
**Project**: Sanskrit POS Tagging: Cross-Domain Generalization and Few-Shot Adaptation  
**Date**: 2026-11-18  
**Scope**: Source Domain (Literature), Target Domain (News Annotated), and Target Domain (News Unannotated)

---

## 1. Summary Overview

This report documents the rigorous schema and data integrity verification performed on all three datasets in `data/raw/`.

| Metric | Literature_Corpus.csv | News_Annotated.csv | News_Unannotated.csv |
|---|---|---|---|
| **Domain** | Source (Literature) | Target Gold (News) | Target Unannotated (News) |
| **Row Count** | 11,899 | 24,299 | 3,000 |
| **Unique Sentences** | 1,461 | 2,645 | 3,000 |
| **Unique Documents** | 1,461 (derived) | 1,971 | 1,554 |
| **Tokens (non-null)** | 11,895 | 24,299 | ~52,600 (estimated) |
| **Missing Tokens** | 4 (whitespace/empty) | 0 | 0 |
| **Missing POS Tags** | 24 | 0 | 3,000 (unannotated) |
| **Unique Tags** | 148 raw morph-tags | 23 target tags | N/A |
| **Duplicate Rows** | 0 | 0 | 0 |
| **Duplicate Sentence Texts** | 49 | 20 | 23 |
| **Token Monotonicity** | 100% sequential | 100% sequential | Sequential |
| **Min Tokens / Sentence** | 1 | 1 | 1 |
| **Max Tokens / Sentence** | 56 | 68 | 84 |
| **Mean Tokens / Sentence** | 8.14 | 9.19 | 17.53 |

---

## 2. Dataset Schema & Column Details

### 2.1 Source Domain: `Literature_Corpus.csv`
- **Columns**: `sentence_id` (int64), `word_number` (int64), `word` (object), `lemma` (object), `tag` (object), `sentence` (object).
- **Structure**: One row per token.
- **Key Findings**:
  - `doc_id` is not present in the raw file; `sentence_id` is unique per sentence and serves as the group identity for splitting.
  - 4 rows have null `word` due to blank whitespace rows in original transcription.
  - 24 rows have null `tag`.
  - Token ordering (`word_number`) within every sentence is strictly monotonic.

### 2.2 Target Domain Gold: `News_Annotated.csv`
- **Columns**: `doc_id` (int64), `sentence_id` (int64), `annotation_id` (int64), `sentence_text` (object), `token_id` (int64), `token` (object), `lemma` (object), `upos` (object), `stanza_upos` (object), `xpos` (object), `feats` (object), `gold_sanskrit_tag` (object), `annotation_status` (object), `annotation_note` (object), `verified_by` (object), `verified_at` (object).
- **Structure**: One row per token.
- **Key Findings**:
  - Composite sentence identity is strictly `(doc_id, sentence_id)` because `sentence_id` indexes sentences within documents (0, 1, 2, ...).
  - 2,645 composite sentences across 1,971 unique document IDs.
  - Zero missing tokens, zero missing gold tags.
  - All gold tags adhere strictly to the target 24-tag schema.
  - Token ordering (`token_id`) within every composite sentence is strictly monotonic.

### 2.3 Target Domain Unannotated: `News_Unannotated.csv`
- **Columns**: `doc_id` (int64), `sentence_id` (int64), `sentence` (object), `source_doc_id` (int64), `sentence_in_doc` (int64), `annotation_id` (int64).
- **Structure**: One row per sentence (3,000 sentences).
- **Key Findings**:
  - Composite sentence identity is strictly `(doc_id, sentence_id)` (3,000 unique composite sentences across 1,554 documents).
  - Contains raw Sanskrit news sentences intended for model inference, disagreement analysis, and semi-supervised exploration.
  - Will NOT be evaluated with accuracy or F1 since no gold labels are present.

---

## 3. Verification Decisions

1. **Composite Sentence Identity**: For news data, all grouping and splitting operations MUST use composite key `f"{doc_id}::{sentence_id}"`.
2. **Document-Level Group Separation**: Document boundaries (`doc_id`) are preserved in news splits to prevent leakage across sentences belonging to the same article.
3. **Incomplete Sentence Handling in Literature**: The 473 unmapped / incomplete occurrences in Literature Corpus are segregated from final training data (`excluded_train_incomplete.csv`, `excluded_validation_incomplete.csv`) to ensure 100% gold label purity in model training.
