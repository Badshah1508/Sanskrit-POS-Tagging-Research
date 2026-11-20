# Sanskrit POS Tagging: Cross-Domain Generalization and Few-Shot Adaptation

**M.Tech Data Science Dissertation Project**  
**Repository / Workspace**: `/home/badshah/Desktop/Cross_Domain`  
**Date**: November 2026  

---

## 1. Dissertation Overview

This research explores part-of-speech (POS) tagging for the low-resource, morphologically rich language **Sanskrit**, with a specific focus on:
1. **In-Domain Generalization**: Performance in the source domain (**Classical Sanskrit Literature**).
2. **Cross-Domain Generalization**: Performance loss under zero-shot transfer from literature to the target domain (**Modern Sanskrit News**).
3. **Few-Shot Adaptation**: Performance recovery when fine-tuning literature-trained models with limited target-domain budgets (**0, 50, 100, 200, 500, 1,000 sentences**).
4. **Target In-Domain Benchmark**: Upper-bound reference model trained and tested natively on news.
5. **Unannotated Target Inference**: Prediction and linguistic analysis on **3,000 unannotated Sanskrit news sentences**.

---

## 2. Directory Structure

```text
Cross_Domain/
├── configs/                             # Model hyperparameter configurations
│   ├── bilstm_crf.json
│   ├── xlm_roberta.json
│   └── indicbert_v2.json
├── data/
│   └── raw/                             # Verified raw CSV datasets
│       ├── Literature_Corpus.csv        # Source domain (1,461 sentences, 11,899 rows)
│       ├── News_Annotated.csv           # Target domain gold (2,645 sentences, 24,299 rows)
│       └── News_Unannotated.csv         # Target domain unannotated (3,000 sentences)
├── notebooks/
│   ├── Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb # Primary research notebook
│   └── optimized_sanskrit_pos_notebook.ipynb                # Synchronized notebook copy
├── outputs/
│   ├── datasets/                        # Standardized leak-free CSV & PKL splits
│   │   ├── literature_train.csv
│   │   ├── literature_validation.csv
│   │   ├── literature_test_unlabeled.csv
│   │   ├── literature_test_gold.csv
│   │   ├── news_adaptation_pool.csv
│   │   ├── news_validation.csv
│   │   ├── news_test_unlabeled.csv
│   │   ├── news_test_gold.csv
│   │   └── split_manifest.json
│   ├── mapping_reviews/                 # Complete mapping audit trail
│   │   ├── final_mapping_table.csv
│   │   ├── mapping_coverage_report.csv
│   │   ├── unmapped_occurrences.csv
│   │   ├── mapping_audit_log.csv
│   │   └── mapping_summary.json
│   ├── experiments/                     # Append-only experiment tracking
│   │   ├── BiLSTM_CRF/
│   │   ├── experiment_registry.csv      # Registry of all runs
│   │   └── best_model_index.csv         # Best checkpoint index based on Val Macro F1
│   ├── evaluations/                     # Consolidated tables and reports
│   │   ├── table1_in_domain.csv
│   │   ├── table2_zero_shot.csv
│   │   ├── table3_few_shot.csv
│   │   ├── table4_experiment_history.csv
│   │   └── FINAL_EXPERIMENT_REPORT.md
│   ├── figures/                         # Publication-grade evaluation figures
│   │   ├── few_shot_macro_f1_curve.png
│   │   ├── bilstm_lit_in_domain_confusion.png
│   │   ├── bilstm_news_in_domain_confusion.png
│   │   └── bilstm_zero_shot_confusion.png
│   ├── predictions/                     # Downstream inference outputs
│   │   └── unannotated_news/
│   └── logs/                            # Project audit reports
│       └── PROJECT_AUDIT_REPORT.md
├── src/                                 # Modular, testable Python package
│   ├── data_preparation/
│   │   └── dataset.py                   # Split loading, validation, nested few-shot sampling
│   ├── models/
│   │   ├── bilstm_crf.py                # Padding-aware BiLSTM-CRF sequence tagger
│   │   └── transformer_tagger.py        # XLM-R & IndicBERT v2 with subword-to-word alignment
│   ├── training/
│   │   └── trainer.py                   # Unified trainer, early stopping, checkpointing
│   ├── evaluation/
│   │   ├── evaluator.py                 # Macro/Weighted F1, 24-tag confusion matrix
│   │   ├── visualizer.py                # Table generation and publication plotting
│   │   └── inference.py                 # Unannotated news inference runner
│   └── experiments/
│       ├── run_experiments.py          # Master experimental pipeline CLI
│       ├── tracking.py                  # Collision-resistant run dirs & registry
│       └── kaggle_sync.py               # Kaggle bundle packaging & sync
├── requirements.txt
└── README.md
```

---

## 3. Standardized 24-Tag Sanskrit POS Tagset

All models evaluate against the exact same 24 tags:
`N`, `NS`, `NV`, `KNV`, `SN`, `SNV`, `SNS`, `SAM`, `SAMY`, `SAMP`, `KP`, `UP`, `AVN`, `AVP`, `AVUP`, `SAMB`, `AVU`, `K`, `AKV`, `AB`, `AVC`, `DVI`, `A`, `PU`.

---

## 4. Empirical Research Findings (BiLSTM-CRF)

### 4.1 In-Domain Performance
- **Literature In-Domain**: Accuracy = **68.84%**, Macro F1 = **0.3691**, Weighted F1 = **0.6743** (Tokens = 1,807).
- **News In-Domain**: Accuracy = **66.17%**, Macro F1 = **0.2319**, Weighted F1 = **0.6246** (Tokens = 7,324).

### 4.2 Cross-Domain Degradation (Literature $\rightarrow$ News Zero-Shot)
- **Zero-Shot News Evaluation**: Accuracy = **23.58%**, Macro F1 = **0.1015**, Weighted F1 = **0.2858**.
- **Absolute Macro F1 Drop**: **-0.2676** (-72.49% relative drop).
- *Linguistic analysis*: Classical literature contains distinct syntactic structures, archaic compounds, and poetic word order that do not transfer directly to contemporary news reporting, resulting in severe degradation.

### 4.3 Few-Shot Adaptation (Averaged across Seeds 42, 123, 2026)

| Annotation Budget | Mean Macro F1 | Std Dev | Macro F1 Gain vs Zero-Shot | Recovery of In-Domain Gap |
|---|---|---|---|---|
| **0 Sentences (Zero-Shot)** | 0.1015 | - | - | 0.0% |
| **50 Sentences** | 0.1300 | $\pm$ 0.0140 | +0.0285 | 21.9% |
| **100 Sentences** | 0.1512 | $\pm$ 0.0002 | +0.0496 | 38.1% |
| **200 Sentences** | 0.1576 | $\pm$ 0.0018 | +0.0561 | 43.0% |
| **500 Sentences** | 0.1720 | $\pm$ 0.0157 | +0.0704 | 54.1% |
| **1,000 Sentences** | **0.1965** | $\pm$ 0.0019 | **+0.0950** | **56.6%** |

---

## 5. Execution and Reproduction

### Local Execution (Virtual Environment)
```bash
# Activate virtual environment
source .venv/bin/activate

# Execute all BiLSTM-CRF experiments, few-shot adaptation, and unannotated news inference
python src/experiments/run_experiments.py --model bilstm_crf
```

### Kaggle GPU Acceleration (Transformers)
To train transformer models (`FacebookAI/xlm-roberta-base`, `ai4bharat/IndicBERTv2-MLM-only`) on remote GPU:
1. Generate the Kaggle bundle zip:
   ```bash
   python -c "from src.experiments.kaggle_sync import create_kaggle_bundle; create_kaggle_bundle('.')"
   ```
2. Upload `outputs/kaggle_bundle.zip` to your Kaggle Notebook / Dataset.
3. Run the training script on Kaggle T4/P100 GPU:
   ```bash
   python src/experiments/run_experiments.py --model all
   ```
4. Download the Kaggle output zip and unpack locally:
   ```bash
   python -c "from src.experiments.kaggle_sync import unpack_kaggle_results; unpack_kaggle_results('kaggle_output.zip', '.')"
   ```
