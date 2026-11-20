# Sanskrit POS Tagging: Five-Model Baseline Experimental Evaluation
**Dissertation**: Cross-Domain Generalization and Few-Shot Adaptation for Sanskrit POS Tagging  
**Author**: Badshah Kumar  
**Supervisor / Reviewer**: Senior NLP Research Engineer & Experimentation Specialist  
**Evaluation Phase**: Phase 1 — Pure Unaugmented Baseline Benchmark  
**Date**: November 19, 2026  

---

## Executive Summary
This report establishes the official, reproducible Phase 1 experimental baseline across five models for Sanskrit POS tagging under in-domain, zero-shot cross-domain, and few-shot adaptation settings. In strict adherence to Phase 1 constraints, all models are evaluated as **pure baselines without Sanskrit-specific feature engineering or morphological augmentation**.

---

## 1. Experimental Setup & Protocol
- **Target POS Tagset (24 Tags)**: , , , , , , , , , , , , , , , , , , , , , , , 
- **Source Domain**: Classical Sanskrit Literature (Panchatantra, Epics, Kāvya prose)
- **Target Domain**: Modern Sanskrit Newspaper Prose (Contemporary journalistic discourse)
- **Few-Shot Budgets**: 0 (zero-shot), 50, 100, 200, 500, 1,000 sentences (strictly nested subsets)
- **Random Seeds**: 42, 123, 2026 (for statistical robustness)

---

## 2. In-Domain Performance Benchmark (Experiment 1 & Experiment 2)

### Table 1: Literature In-Domain (Source Benchmark)
| Model ID | Model Name | Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | BiLSTM-CRF | Literature Test | **68.84%** | 0.4339 | 0.3485 | **36.91%** | 67.43% | **Completed** |
| **M2** | XLM-RoBERTa | Literature Test | — | — | — | — | — | Pending GPU |
| **M3** | IndicBERT-v2 | Literature Test | — | — | — | — | — | Pending GPU |
| **M4** | MuRIL | Literature Test | — | — | — | — | — | Pending GPU |
| **M5** | Sanskrit-BERT | Literature Test | — | — | — | — | — | **Ready (Option A Transliteration)** |

### Table 2: News In-Domain (Target Reference Upper Bound)
| Model ID | Model Name | Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | BiLSTM-CRF | Locked News Test | **66.17%** | 0.2842 | 0.2319 | **23.19%** | 62.46% | **Completed** |
| **M2** | XLM-RoBERTa | Locked News Test | — | — | — | — | — | Pending GPU |
| **M3** | IndicBERT-v2 | Locked News Test | — | — | — | — | — | Pending GPU |
| **M4** | MuRIL | Locked News Test | — | — | — | — | — | Pending GPU |
| **M5** | Sanskrit-BERT | Locked News Test | — | — | — | — | — | **Ready (Option A Transliteration)** |

---

## 3. Zero-Shot Cross-Domain Generalization (Experiment 3)
Measures direct domain transfer: **Sanskrit Literature $	o$ Sanskrit News** without any target training data.

| Model ID | Model Name | Literature Macro F1 | News Zero-Shot Macro F1 | Absolute F1 Drop | Relative Degradation (%) | Zero-Shot Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | BiLSTM-CRF | **0.3691** | **0.1015** | **-0.2676** | **-72.49%** | **23.58%** |
| **M2** | XLM-RoBERTa | — | — | — | — | Pending GPU |
| **M3** | IndicBERT-v2 | — | — | — | — | Pending GPU |
| **M4** | MuRIL | — | — | — | — | Pending GPU |
| **M5** | Sanskrit-BERT | — | — | — | — | **Ready (Option A Transliteration)** |

> [!CAUTION]
> **The 72.49% Performance Collapse**:
> The word-level BiLSTM-CRF suffers a massive 72.49% drop in Macro F1 and accuracy collapses from 68.8% to 23.6%. Our token audit proves that **81.47% of all words in the News test set are Out-Of-Vocabulary (OOV)** relative to Literature.

---

## 4. Few-Shot Target Domain Adaptation (Experiment 4)
Models fine-tuned from Literature-trained checkpoints using nested target News subsets:

| Budget (Sentences) | Approx. Tokens | M1 Accuracy (Mean ± SD) | M1 Macro F1 (Mean ± SD) | Weighted F1 (Mean ± SD) | F1 Improvement (pp) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 (Zero-Shot)** | 0 | 23.58% | 10.15% | 28.58% | 0.00 |
| **50** | ~456 | 57.44% ± 4.57% | 13.00% ± 1.37% | 50.63% ± 3.86% | +2.85 |
| **100** | ~912 | 63.55% ± 0.55% | 15.04% ± 0.13% | 56.16% ± 0.25% | +4.89 |
| **200** | ~1,824 | 64.87% ± 0.29% | 15.76% ± 0.18% | 57.73% ± 0.29% | +5.61 |
| **500** | ~4,560 | 65.81% ± 0.34% | 17.20% ± 1.58% | 59.24% ± 0.39% | +7.05 |
| **1000** | ~9,120 | **66.64% ± 0.42%** | **19.65% ± 0.20%** | **60.38% ± 0.25%** | **+9.50** |

---

## 5. M5 (Sanskrit-BERT) Option A Integration & Verification
- **Issue Discovered**: `Matej/bert-base-buddhist-sanskrit-v2` tokenizer was pre-trained purely on Latin/IAST script. Passing raw Devanagari resulted in 100% `[UNK]` tokens.
- **Solution Executed (Option A)**:
  - Designed and tested an automated, zero-external-dependency `devanagari_to_iast` transliteration layer.
  - Integrated into `WordTokenClassificationDataset` and `TransformerTagger.predict_words`.
  - Transliterates words to IAST before subword tokenization, preserving strict 1:1 word-to-token index alignment.
- **Verification Results**:
  - Sample sentence tokenized into 12 subwords without a single `[UNK]` token.
  - Exactly 5 first-subword positions mapped to the 5 gold word POS labels.
  - M1, M2, M3, and M4 remain 100% in native Devanagari script.

---

## 6. Structured Data Deliverables & File Index
All results have been generated and preserved in:
- : [literature_in_domain_results.csv](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/literature_in_domain_results.csv)
- : [news_in_domain_results.csv](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/news_in_domain_results.csv)
- : [cross_domain_zero_shot_results.csv](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/cross_domain_zero_shot_results.csv)
- : [few_shot_results.csv](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/few_shot_results.csv)
- : [per_tag_results.csv](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/per_tag_results.csv)
- : [master_results.csv](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/master_results.csv)
- : [baseline_experiment_results.xlsx](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/baseline_experiment_results.xlsx)
- Figures: [figures/](file:///home/badshah/Desktop/Cross_Domain/outputs/baseline_experiments/figures/)
