# Project Audit Report: Panchatantra-Based Cross-Domain Sanskrit POS Tagging

**Dissertation Title:** Cross-Domain Generalization and Few-Shot Adaptation for Sanskrit POS Tagging  
**Audit Date:** November 26, 2026  
**Output Directory:** `outputs/Panchatantra_output/`  
**Primary Notebook:** `notebooks/Sanskrit_Panchatantra_CrossDomain_Experiments.ipynb`

---

## 1. Existing Project Architecture

The repository (`/home/siba/P25DS023/Cross-Domain-23`) is organized into modular Python packages and experiment configs:

- `data/raw/`:
  - `Literature_Corpus.csv` (2.5 MB, 11,899 tokens, 1,461 sentences — **reference only; excluded from Panchatantra experiments**)
  - `Panchatantra_clean.csv` (9.5 MB, 43,653 tokens, 5,393 sentences — **new source-domain dataset**)
  - `News_Annotated.csv` (11.7 MB, 24,299 tokens, 2,645 sentences, 1,971 documents — **target-domain dataset**)
  - `News_Unannotated.csv` (857 KB, 3,000 sentences)
  - `sa_ufal_ud_test_panchtantra_clean.csv` (382 KB, UD-tagged Panchatantra subset — reference only)
- `configs/`:
  - `bilstm_crf.json` (`embedding_dim=128`, `hidden_size=128`, `num_layers=2`, `dropout=0.3`, `use_morphology=false`, `lr=0.001`, `batch_size=32`, `epochs=25`, `patience=5`, `seed=42`)
  - `xlm_roberta.json` (`FacebookAI/xlm-roberta-base`, `lr=3e-5`, `batch_size=8`, `epochs=10`, `max_length=256`, `patience=2`, `seed=42`)
  - `indicbert_v2.json` (`ai4bharat/IndicBERTv2-MLM-only`, same base hyperparameters)
  - `muril.json` (`google/muril-base-cased`, same base hyperparameters)
  - `sanskrit_bert.json` (`Matej/bert-base-buddhist-sanskrit-v2`, same base hyperparameters)
- `src/`:
  - `src/data_preparation/dataset.py`: Defines the 24 `TARGET_TAGS`, `TAG2ID`, `ID2TAG`, `validate_split_frame`, `validate_split_leakage`, and `get_nested_few_shot_subsets`.
  - `src/models/bilstm_crf.py`: Word-level `BiLSTMCRF` with `torchcrf.CRF`, `SequenceTaggingDataset`, `build_vocabulary`, `collate_sequences`.
  - `src/models/transformer_tagger.py`: `TransformerTagger` wrapping `AutoModelForTokenClassification`, `WordTokenClassificationDataset` with first-subword label alignment (`-100` for subsequent subwords and special tokens), and `devanagari_to_iast()` 1:1 transliteration for M5.
  - `src/training/trainer.py`: `train_bilstm_crf`, `decode_bilstm_predictions`, and `train_transformer` with validation Macro F1 checkpointing and early stopping.
  - `src/evaluation/evaluator.py`: `evaluate_predictions` and `save_evaluation_results` computing Accuracy, Macro/Weighted Precision/Recall/F1, per-tag metrics, and 24x24 confusion matrices.
  - `src/evaluation/visualizer.py` & `src/experiments/tracking.py`: Plotting helpers and run registry utilities.

---

## 2. Previous Literature-to-News Methodology & Audit

- The previous experiment trained on `Literature_Corpus.csv` (904 train / 279 val / 227 test sentences after mapping JNU tags to the 24 `TARGET_TAGS` and excluding incomplete sentences) and evaluated on `News_Annotated.csv` (`news_train_pool`: 1,484 sentences; `news_validation`: 370 sentences; `news_test_locked`: 791 sentences).
- Only **BiLSTM-CRF (M1)** had completed full execution in the previous `outputs/baseline_experiments/` and `outputs/experiments/BiLSTM_CRF/` directories; transformer runs (M2–M5) were previously pending GPU execution.
- All previous outputs in `outputs/baseline_experiments/`, `outputs/experiments/`, `outputs/datasets/`, and `notebooks/Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb` are **preserved untouched**.

---

## 3. Reusable vs. New Components for Panchatantra Experiment

### Reusable (Verified)
- Target 24-tag inventory (`TARGET_TAGS`), `validate_split_frame`, `get_nested_few_shot_subsets`.
- Existing validated News splits (`outputs/datasets/news_train_pool.json`, `news_validation.json`, `news_test_locked.json`).
- `BiLSTMCRF`, `TransformerTagger`, `WordTokenClassificationDataset` (with DataLoader collate fix already applied), `train_bilstm_crf`, `train_transformer`, `evaluate_predictions`, `save_evaluation_results`.

### New Implementations Required
1. **Panchatantra POS Mapping:** `Panchatantra_clean.csv` uses a 28-tag Sanskrit tagset (`NP, PUNC, SN, A, KP, KN, NV, KKS, AVK, AN, NS, APY, KV, SNV, AP, AUD, INTF, KNV, ASMB, SSN, ASD, AUP, SMKY, ABS, DV, SMKP, SMK, UA`) rather than the JNU `NC.*/JJ.*/V.*` tagset used in `Literature_Corpus.csv`. A verified linguistic mapping to the 24 `TARGET_TAGS` must be implemented and audited.
2. **Panchatantra Splitting (70/15/15):** `Panchatantra_clean.csv` has 5,393 `sentence_id`s and no `doc_id` column. Sentence-level 70/15/15 splitting (`RANDOM_SEED = 42`) with exact duplicate sentence text deduplication across splits is required.
3. **Notebook-First Pipeline:** `notebooks/Sanskrit_Panchatantra_CrossDomain_Experiments.ipynb` implementing all 19 sections, 3-stage methodology (Stage 1: Panchatantra In-Domain; Stage 2: Zero-Shot News; Stage 3: Few-Shot News Adaptation at `[50, 100, 200, 500, 1000]`), resume support, and automated report/figure generation under `outputs/Panchatantra_output/`.

---

## 4. Model & Hardware Verification Status

| ID | Model | Checkpoint / Implementation | Status |
|---|---|---|---|
| **M1** | BiLSTM-CRF | `src/models/bilstm_crf.py` (`use_morphology=false`) | Verified & Ready |
| **M2** | XLM-RoBERTa | `FacebookAI/xlm-roberta-base` | Cached & Verified |
| **M3** | IndicBERT v2 | `ai4bharat/IndicBERTv2-MLM-only` | Cached & Verified |
| **M4** | MuRIL | `google/muril-base-cased` | Cached & Verified |
| **M5** | Sanskrit-BERT | `Matej/bert-base-buddhist-sanskrit-v2` (IAST via `devanagari_to_iast`) | Cached & Verified |

- **Hardware:** SLURM cluster (`gpu` partition) with **NVIDIA H100 NVL (94 GB VRAM)** compute nodes (`node1`, `node2`).
- **Software Environment:** `/home/siba/anaconda3/envs/muril_env` (Python 3.12.12, PyTorch 2.10.0+cu128, Transformers 5.2.0, `pytorch-crf` 0.7.2, `seqeval` 1.2.2, `nbformat` 5.11.1, `nbclient` 0.11.0).

---

## 5. Identified Methodological & Data Issues

1. **Tagset Semantic Alignment (`NV` vs `A`):** In `Panchatantra_clean.csv`, `NV` denotes nominal adjectives (`महान्`, `महती`, `मूर्खः`), whereas in `News_Annotated.csv` (`TARGET_TAGS`), adjectives are tagged `A` and non-finite verbals/participles (`उक्तम्`, `कृतम्`, `श्रुत्वा`, `गत्वा`) are tagged `NV`. Mapping `Panchatantra NV -> A` and `Panchatantra KN/KKS -> NV` aligns both corpora consistently.
2. **Missing & Unresolved Source Tags:** 6 tokens in `Panchatantra_clean.csv` have `NaN` tags and 1 token has tag `UA` (`प्रतिदिनम्/A`). Sentences containing these 7 tokens are excluded rather than fabricated, and transparently reported.
3. **Tokenization Artifacts:** 113 tokens in `Panchatantra_clean.csv` contain embedded `/` from upstream comma-attached slash tags (e.g., `मित्राणि/NP,` tagged `PUNC`); these are documented in `DATASET_AUDIT_REPORT.md`.
4. **M5 Script Mismatch:** `Matej/bert-base-buddhist-sanskrit-v2` only supports Romanized IAST Latin script; `devanagari_to_iast` transliteration is applied at word level to preserve 1:1 token alignment.
