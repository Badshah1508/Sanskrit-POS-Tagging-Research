# Panchatantra-Based Cross-Domain Generalization and Few-Shot Adaptation for Sanskrit POS Tagging

**Generated:** 2026-11-26T06:44:19.170192+00:00  
**Source Domain:** `Panchatantra_clean.csv` (3,928 train / 763 val / 695 test sentences)  
**Target Domain:** `News_Annotated.csv` (1,484 adaptation pool / 370 val / 791 locked test sentences)  
**Target Tagset:** 24 Sanskrit POS tags (`N, NS, NV, KNV, SN, SNV, SNS, SAM, SAMY, SAMP, KP, UP, AVN, AVP, AVUP, SAMB, AVU, K, AKV, AB, AVC, DVI, A, PU`)

---

## 1. Objective 1: Cross-Domain Generalization (Panchatantra → News)

### Stage 1: Panchatantra In-Domain Performance
| model_id | model_name | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | training_time_sec | inference_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | BiLSTM-CRF | 0.8202 | 0.4319 | 0.3756 | 0.3875 | 0.8070 | 50.1700 | 0.1400 |
| M2 | XLM-RoBERTa | 0.8962 | 0.4823 | 0.4540 | 0.4665 | 0.8954 | 454.7100 | 0.6400 |
| M3 | IndicBERT v2 | 0.9077 | 0.4910 | 0.4985 | 0.4919 | 0.9086 | 476.2700 | 0.5800 |
| M4 | MuRIL | 0.9018 | 0.4685 | 0.4109 | 0.4268 | 0.8959 | 529.8500 | 0.5100 |
| M5 | Sanskrit-BERT | 0.8898 | 0.4649 | 0.4385 | 0.4494 | 0.8883 | 333.5300 | 0.5700 |

### Stage 2: Zero-Shot Cross-Domain Degradation
| model_id | model_name | panchatantra_in_domain_macro_f1 | zero_shot_news_macro_f1 | absolute_f1_drop_pp | relative_f1_drop_pct | panchatantra_accuracy | zero_shot_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | BiLSTM-CRF | 0.3875 | 0.1514 | 23.6200 | 60.9400 | 0.8202 | 0.5102 |
| M2 | XLM-RoBERTa | 0.4665 | 0.2374 | 22.9100 | 49.1100 | 0.8962 | 0.6614 |
| M3 | IndicBERT v2 | 0.4919 | 0.2282 | 26.3600 | 53.6000 | 0.9077 | 0.6537 |
| M4 | MuRIL | 0.4268 | 0.2313 | 19.5500 | 45.8100 | 0.9018 | 0.6614 |
| M5 | Sanskrit-BERT | 0.4494 | 0.1988 | 25.0600 | 55.7600 | 0.8898 | 0.6348 |

**Key Finding (Objective 1):**  
All five models experience substantial cross-domain degradation when evaluated zero-shot on Sanskrit News without target-domain fine-tuning. The strongest zero-shot model is **XLM-RoBERTa** (Macro F1 = 0.2374, Accuracy = 0.6614).

---

## 2. Objective 2: Few-Shot Cross-Domain Adaptation

### Stage 3: Few-Shot Adaptation Summary (Mean ± Std across seeds 42, 123, 2026)
| model_id | model_name | annotation_budget | macro_f1_mean | macro_f1_std | accuracy_mean | weighted_f1_mean | improvement_pp_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | BiLSTM-CRF | 0 | 0.1514 | 0.0000 | 0.5102 | 0.4907 | 0.0000 |
| M1 | BiLSTM-CRF | 50 | 0.1887 | 0.0029 | 0.6386 | 0.5809 | 3.7300 |
| M1 | BiLSTM-CRF | 100 | 0.1875 | 0.0070 | 0.6410 | 0.5826 | 3.6100 |
| M1 | BiLSTM-CRF | 200 | 0.1907 | 0.0019 | 0.6513 | 0.5871 | 3.9367 |
| M1 | BiLSTM-CRF | 500 | 0.1925 | 0.0003 | 0.6628 | 0.5950 | 4.1100 |
| M1 | BiLSTM-CRF | 1000 | 0.1953 | 0.0025 | 0.6662 | 0.5994 | 4.4000 |
| M2 | XLM-RoBERTa | 0 | 0.2374 | 0.0000 | 0.6614 | 0.6778 | 0.0000 |
| M2 | XLM-RoBERTa | 50 | 0.2534 | 0.0127 | 0.7039 | 0.7022 | 1.6000 |
| M2 | XLM-RoBERTa | 100 | 0.2687 | 0.0088 | 0.7329 | 0.7254 | 3.1333 |
| M2 | XLM-RoBERTa | 200 | 0.3029 | 0.0222 | 0.7551 | 0.7484 | 6.5500 |
| M2 | XLM-RoBERTa | 500 | 0.3251 | 0.0013 | 0.7697 | 0.7668 | 8.7700 |
| M2 | XLM-RoBERTa | 1000 | 0.3286 | 0.0045 | 0.7792 | 0.7757 | 9.1300 |
| M3 | IndicBERT v2 | 0 | 0.2282 | 0.0000 | 0.6537 | 0.6675 | 0.0000 |
| M3 | IndicBERT v2 | 50 | 0.2365 | 0.0009 | 0.7003 | 0.6910 | 0.8300 |
| M3 | IndicBERT v2 | 100 | 0.2452 | 0.0033 | 0.7162 | 0.7044 | 1.6933 |
| M3 | IndicBERT v2 | 200 | 0.2615 | 0.0025 | 0.7329 | 0.7213 | 3.3233 |
| M3 | IndicBERT v2 | 500 | 0.3130 | 0.0027 | 0.7626 | 0.7590 | 8.4800 |
| M3 | IndicBERT v2 | 1000 | 0.3231 | 0.0008 | 0.7798 | 0.7734 | 9.4867 |
| M4 | MuRIL | 0 | 0.2313 | 0.0000 | 0.6614 | 0.6670 | 0.0000 |
| M4 | MuRIL | 50 | 0.2368 | 0.0023 | 0.6934 | 0.6781 | 0.5567 |
| M4 | MuRIL | 100 | 0.2376 | 0.0041 | 0.7056 | 0.6839 | 0.6333 |
| M4 | MuRIL | 200 | 0.2338 | 0.0052 | 0.7246 | 0.6869 | 0.2533 |
| M4 | MuRIL | 500 | 0.2397 | 0.0354 | 0.7499 | 0.7201 | 0.8400 |
| M4 | MuRIL | 1000 | 0.3046 | 0.0089 | 0.7818 | 0.7711 | 7.3300 |
| M5 | Sanskrit-BERT | 0 | 0.1988 | 0.0000 | 0.6348 | 0.6319 | 0.0000 |
| M5 | Sanskrit-BERT | 50 | 0.2267 | 0.0042 | 0.6801 | 0.6686 | 2.7867 |
| M5 | Sanskrit-BERT | 100 | 0.2342 | 0.0008 | 0.7030 | 0.6864 | 3.5333 |
| M5 | Sanskrit-BERT | 200 | 0.2694 | 0.0138 | 0.7216 | 0.7105 | 7.0600 |
| M5 | Sanskrit-BERT | 500 | 0.2989 | 0.0063 | 0.7469 | 0.7400 | 10.0100 |
| M5 | Sanskrit-BERT | 1000 | 0.3139 | 0.0046 | 0.7581 | 0.7527 | 11.5033 |

**Key Finding (Objective 2):**  
Independent few-shot fine-tuning from each model's Panchatantra checkpoint consistently improves News POS tagging performance as annotated news sentences increase from 50 to 1,000. At 1,000 sentences, **XLM-RoBERTa** achieves a mean Macro F1 of **0.3286** (+9.13 pp over zero-shot).

---

## 3. Reproducibility & Artifacts
- **Primary Notebook:** `notebooks/Sanskrit_Panchatantra_CrossDomain_Experiments.ipynb`
- **All Outputs:** `outputs/Panchatantra_output/`
- **Evaluations:** `evaluations/panchatantra_in_domain_results.csv`, `zero_shot_cross_domain_results.csv`, `few_shot_results.csv`, `per_tag_results.csv`, `performance_drop_results.csv`, `master_results.csv`
- **Figures:** `figures/panchatantra_pos_distribution.png`, `news_pos_distribution.png`, `in_domain_vs_zero_shot.png`, `few_shot_learning_curves.png`, `per_tag_f1_comparison.png`, `figures/confusion_matrices/`
