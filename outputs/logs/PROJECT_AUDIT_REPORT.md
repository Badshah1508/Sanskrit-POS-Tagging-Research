# Complete Project Audit Report
**Project Title**: Sanskrit POS Tagging: Cross-Domain Generalization and Few-Shot Adaptation  
**Dissertation**: M.Tech Data Science Dissertation  
**Audit Date**: 2026-11-18  
**Workspace Path**: `/home/badshah/Desktop/Cross_Domain`  
**Auditor**: Antigravity AI (Google DeepMind)

---

## 1. Executive Summary

This audit report evaluates the existing codebase, dataset integrity, mapping status, model implementations, and experimental results for the M.Tech Data Science dissertation project on **Sanskrit POS Tagging: Cross-Domain Generalization and Few-Shot Adaptation**.

The core objective is to investigate:
1. **In-Domain Generalization**: Performance within the Sanskrit Literature domain (Source).
2. **Cross-Domain Degradation**: Performance loss when literature-trained models are evaluated on Sanskrit News (Target) without adaptation (Zero-shot).
3. **Few-Shot Adaptation**: Performance recovery when fine-tuning literature-trained models on small annotation budgets (0, 50, 100, 200, 500, 1,000 sentences) from the target news domain.
4. **Target Domain Benchmark**: Target in-domain performance (News trained -> News evaluated).
5. **Unannotated Target Inference**: Applying trained taggers to 3,000 unannotated Sanskrit news sentences.

---

## 2. Environment and Hardware Audit

- **Operating System**: Linux (x86_64)
- **CPU**: AMD Ryzen 3 3250U with Radeon Graphics (2 cores, 4 threads / vCPUs)
- **RAM**: 5.2 GiB Total (Available: ~1.4 GiB, Swap: 4.0 GiB actively utilized)
- **GPU / Accelerator**: None available locally (`nvidia-smi` command not found). Local PyTorch is CPU-only (`torch==2.15.0+cpu`, `CUDA available: False`).
- **Python Virtual Environment**: `.venv` at `/home/badshah/Desktop/Cross_Domain/.venv` (Python 3.14.4).
- **Installed Packages**:
  - `torch==2.15.0+cpu`
  - `transformers==4.57.6`
  - `pytorch-crf==0.7.2`
  - `seqeval==1.2.2`
  - `datasets==5.0.1`
  - `pandas==3.1.1`
  - `scikit-learn==1.9.1`
  - `matplotlib==3.11.2`
  - `seaborn==0.13.2`
  - `sentencepiece==0.2.3`
  - `jupyter==1.1.1`
- **Execution Constraints**:
  - Word-level BiLSTM-CRF models can train and evaluate efficiently on the local CPU runtime.
  - Transformer models (`FacebookAI/xlm-roberta-base`, `ai4bharat/IndicBERTv2-MLM-only`) will require substantial execution time on CPU with 5.2 GiB RAM, or can execute via Kaggle GPU runtime with persistent synchronization.

---

## 3. Directory Structure Audit

| Path | Status | Purpose & Contents |
|---|---|---|
| `configs/` | Verified | Model hyperparameter JSONs: `bilstm_crf.json`, `xlm_roberta.json`, `indicbert_v2.json`. |
| `data/raw/` | Verified | Primary CSV datasets (`Literature_Corpus.csv`, `News_Annotated.csv`, `News_Unannotated.csv`) plus reference files (`MSRIndic_JNUTagsetTaggedCorpora_clean(1).csv`, `Panchatantra_clean.csv`, `sa_ufal_ud_test_panchtantra_clean.csv`). |
| `data/processed/` | Verified | Target directory for standardized, cleaned processed splits. |
| `notebooks/` | Verified | `optimized_sanskrit_pos_notebook.ipynb` (447 cells) and backup `optimized_sanskrit_pos_notebook.ipynb.bak`. |
| `outputs/datasets/` | Verified | Split pickles (`source_prepared.pkl`, `target_prepared.pkl`, `news_train_pool.pkl`, `news_validation.pkl`, `news_test_locked.pkl`), `Final_Literature_Data/` pickles. Missing split CSVs (`literature_train.csv`, `news_test_gold.csv`, etc.). |
| `outputs/mapping_reviews/` | Verified | Contains source prefix inventories, JQ and PPR occurrence reviews, manual reviews, gold review sheets, and `unresolved_mapping_queue.csv`. |
| `outputs/models/` | Verified | `bilstm_crf/bilstm_source_pretrained.pt` checkpoint present. XLM-RoBERTa and IndicBERT directories present. |
| `outputs/results/` | Verified | Initial BiLSTM-CRF results for literature in-domain and cross-domain zero-shot. |
| `outputs/evaluations/` | Verified | `FINAL_EXPERIMENT_REPORT.md` present. |
| `outputs/logs/` | Verified | Contains this audit report and future execution logs. |
| `src/experiments/` | Verified | Active modular codebase: `data.py`, `models.py`, `metrics.py`, `prepare.py`, `results.py`, `tracking.py`. |
| `src/data_preparation/` | Verified | Directory placeholder (functions currently centralized in `src/experiments/`). |
| `src/models/` | Verified | Directory placeholder (functions currently centralized in `src/experiments/`). |
| `src/training/` | Verified | Directory placeholder (functions currently centralized in `src/experiments/`). |
| `src/evaluation/` | Verified | Directory placeholder (functions currently centralized in `src/experiments/`). |

---

## 4. Dataset Integrity & Schema Audit

### 4.1 Source Domain: Literature (`data/raw/Literature_Corpus.csv`)
- **Total Token Rows**: 11,899
- **Columns**: `['sentence_id', 'word_number', 'word', 'lemma', 'tag', 'sentence']`
- **Total Sentences**: 1,461 unique sentence IDs.
- **Document ID**: No document identifier column is provided in the raw CSV; `sentence_id` serves as the primary grouping key.
- **Missing Values**: No missing words or tags.
- **Tagset**: Source tagset consists of JNU / morphosyntactic tags with prefixes (e.g. `NC`, `NP`, `JJ`, `PPR`, `V`, `PU`, `ALC`, `AMN`, `JQ`, `CX`, `DAB`, etc.).

### 4.2 Target Domain Gold: Annotated News (`data/raw/News_Annotated.csv`)
- **Total Token Rows**: 24,299
- **Columns**: `['doc_id', 'sentence_id', 'annotation_id', 'sentence_text', 'token_id', 'token', 'lemma', 'upos', 'stanza_upos', 'xpos', 'feats', 'gold_sanskrit_tag', 'annotation_status', 'annotation_note', 'verified_by', 'verified_at']`
- **Total Unique Sentences**: 2,645 composite sentences (`doc_id` + `sentence_id`).
- **Total Unique Documents**: 1,971 documents.
- **Gold Tags**: `gold_sanskrit_tag` conforms to the 24-tag target tagset (`N`, `NS`, `NV`, `KNV`, `SN`, `SNV`, `SNS`, `SAM`, `SAMY`, `SAMP`, `KP`, `UP`, `AVN`, `AVP`, `AVUP`, `SAMB`, `AVU`, `K`, `AKV`, `AB`, `AVC`, `DVI`, `A`, `PU`).
- **Annotation Status**: Verified by human Sanskrit scholars.

### 4.3 Target Domain Unannotated: News Sentences (`data/raw/News_Unannotated.csv`)
- **Total Sentence Rows**: 3,000 sentences.
- **Columns**: `['doc_id', 'sentence_id', 'sentence', 'source_doc_id', 'sentence_in_doc', 'annotation_id']`
- **Gold Tags**: None (purely raw sentences).
- **Usage**: Designated exclusively for downstream inference and model agreement analysis. No accuracy or F1 will be computed without gold labels.

### 4.4 Additional Raw Datasets Identified
- `data/raw/MSRIndic_JNUTagsetTaggedCorpora_clean(1).csv`: Identical bit-for-bit duplicate of `Literature_Corpus.csv` (11,899 rows). Kept preserved per instructions.
- `data/raw/Panchatantra_clean.csv`: 43,653 rows. Independent corpus not used in primary literature-news setup.
- `data/raw/sa_ufal_ud_test_panchtantra_clean.csv`: 1,843 rows. Universal Dependencies Panchatantra test set.

---

## 5. POS Mapping Audit

- **Target Tagset (24 Tags)**:
  `N`, `NS`, `NV`, `KNV`, `SN`, `SNV`, `SNS`, `SAM`, `SAMY`, `SAMP`, `KP`, `UP`, `AVN`, `AVP`, `AVUP`, `SAMB`, `AVU`, `K`, `AKV`, `AB`, `AVC`, `DVI`, `A`, `PU`.
- **Current Mapping Status**:
  - Source prefix mappings established: `NC->N`, `NP->NS`, `JJ->A`, `V->KP`, `PU->PU`, `CEM->AVN`, `CNG->AVN`, `CIN->AVU`, `CCD->SAMB`, `CSB->SAMB`.
  - Manual review collections completed: `PPR_final_mappings.csv` (contextual pronouns mapped to `SN`, `SNV`, `SNS`), `Gold_Review/Sanskrit_POS_Fast_Review_All.csv`, `Gold_Review/Sanskrit_POS_Medium_Review.csv`.
  - Historical unresolved queue: 473 occurrences (`outputs/mapping_reviews/unresolved_mapping_queue.csv`).
  - Strict research protocol: Incomplete sentences containing unresolved tokens were safely excluded from `literature_train.pkl` and `literature_validation.pkl` (`excluded_train_incomplete.csv` = 34 sentences, `excluded_validation_incomplete.csv` = 13 sentences) rather than guessing labels.

---

## 6. Dataset Splits & Leakage Audit

### 6.1 Literature Splits (Group-aware by `doc_id` / `sentence_id`)
- **Literature Train**: 904 sentences (7,292 tokens).
- **Literature Validation**: 279 sentences (2,246 tokens).
- **Literature Held-out Test**: 227 sentences (1,807 tokens).
- **Total Validated Literature Sentences**: 1,410 sentences (11,345 tokens).
- **Leakage Check**: Passed. Zero sentence overlap between train, validation, and locked test.

### 6.2 News Splits (Document-level grouped separation)
- **News Adaptation Pool (Train)**: 1,484 sentences (1,103 documents, 13,544 tokens).
- **News Validation**: 370 sentences (276 documents, 3,431 tokens).
- **News Locked Test**: 791 sentences (592 documents, 7,324 tokens).
- **Leakage Check**: Passed. Zero document overlap and zero sentence overlap between adaptation pool, validation, and locked test.
- **Pending Deliverable**: Exporting individual split CSV files (`literature_train.csv`, `news_test_unlabeled.csv`, `news_test_gold.csv`, etc.) and consolidated manifest JSON.

---

## 7. Model Implementations & Verification Status

### 7.1 M1: BiLSTM-CRF
- **Architecture**: Word embedding (dim=128), 2-layer Bidirectional LSTM (hidden_size=128), Dropout (0.2), Linear projection, Linear-Chain CRF (`pytorch-crf`).
- **Masking & Padding**: Padding index 0 handled properly; boolean mask passed to CRF for forward loss and Viterbi decoding.
- **Checkpoint**: `outputs/models/bilstm_crf/bilstm_source_pretrained.pt` trained on literature.
- **Executed In-Domain**: Accuracy = 0.6635, Macro F1 = 0.3543, Weighted F1 = 0.6527 (Tokens = 1,807).
- **Executed Zero-Shot**: Accuracy = 0.2211, Macro F1 = 0.0984, Weighted F1 = 0.2510 (Tokens = 7,324).
- **Pending Runs**: News in-domain reference run, Few-shot adaptation (budgets 0, 50, 100, 200, 500, 1,000 sentences across seeds).

### 7.2 M2: XLM-RoBERTa (`FacebookAI/xlm-roberta-base`)
- **Implementation**: WordTokenClassificationDataset with subword-to-word label alignment (-100 for subsequent subwords and special tokens).
- **Status**: Smoke-tested. Full training pending due to CPU resource constraints / runtime allocation.

### 7.3 M3: IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`)
- **Implementation**: WordTokenClassificationDataset with subword alignment.
- **Status**: Smoke-tested. Full training pending due to CPU resource constraints / runtime allocation.

---

## 8. Prioritized Remediation & Action Plan

1. **Stage 3**: Generate standardized `dataset_statistics.csv`, `dataset_validation_report.md`, and `dataset_schema_report.json`.
2. **Stage 4**: Verify and audit POS mappings, recalculate exact coverage, produce `final_mapping_table.csv`, `mapping_coverage_report.csv`, `unmapped_occurrences.csv`, `mapping_audit_log.csv`, and `mapping_summary.json`.
3. **Stage 5 & 6**: Export all standard split CSVs (`literature_train.csv`, `literature_validation.csv`, `literature_test_unlabeled.csv`, `literature_test_gold.csv`, `news_adaptation_pool.csv`, `news_validation.csv`, `news_test_unlabeled.csv`, `news_test_gold.csv`, `split_manifest.json`) and run automated leakage verification.
4. **Stage 7**: Inspect, update, and verify centralized path configuration in `notebooks/optimized_sanskrit_pos_notebook.ipynb` and all utility scripts.
5. **Stage 8 - 13**: Implement modular, executable pipelines for all three models (BiLSTM-CRF, XLM-R, IndicBERT v2), execute experiments in research order, record tracking registries, and evaluate few-shot adaptation budgets.
6. **Stage 14**: Run batch inference on `News_Unannotated.csv` (3,000 sentences) and generate agreement reports.
7. **Stage 15 & 16**: Generate comparison tables, figures, confusion matrices, and the comprehensive dissertation experiment report.
