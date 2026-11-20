# Project Pre-Experiment Audit Report
**Master Research Project**: Cross-Domain Generalization and Few-Shot Adaptation for Sanskrit POS Tagging  
**Evaluation Phase**: Phase 1 — Five-Model Baseline Experimental Evaluation  
**Date**: November 19, 2026  

---

## 1. Project Directory & Artifact Audit
- **Project Root**: 
- **Target Tagset**: 24 Target POS Tags (, , , , , , , , , , , , , , , , , , , , , , , )
- **Source Domain**: Classical Sanskrit Literature (, 2.58 MB)
- **Target Domain**: Modern Sanskrit News (, 11.68 MB; , 877 KB)
- **Data Integrity & Leakage Verification**: Verified document-level split with 0% sentence/document overlap.

---

## 2. Dataset Split Statistics
| Dataset Split | Role | Sentences | Tokens | Leakage Audit |
| :--- | :--- | :--- | :--- | :--- |
| **Literature Train** | Source In-Domain Training | 1,131 | 9,099 | Document-disjoint |
| **Literature Validation** | Source Model Selection | 279 | 2,246 | Document-disjoint |
| **Literature Held-Out Test** | Source Benchmark Test | 227 | 1,807 | Strictly locked |
| **News Adaptation Pool** | Target Few-Shot Pool | 1,484 | 13,544 | Partitioned |
| **News Validation** | Target Model Selection | 370 | 3,431 | Partitioned |
| **News Held-Out Test** | Target Locked Evaluation | 791 | 7,324 | **Strictly locked** |
| **Unannotated News** | Downstream Field Inference | 3,000 | ~27,000 | Raw text |

---

## 3. Five Model Verification & Checkpoint Audit
| ID | Model Name | Checkpoint Identifier | Model Type | Devanagari Tokenizer Status | Compatibility Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | BiLSTM-CRF | Project implementation | Word-level BiLSTM + CRF | Custom word vocab (3,669 tokens) | **Ready / Completed** |
| **M2** | XLM-RoBERTa |  | Multilingual Transformer | SentencePiece (Devanagari supported) | **Verified / Pending GPU** |
| **M3** | IndicBERT v2 |  | Indic Multilingual BERT | WordPiece (Devanagari native) | **Verified / Pending GPU** |
| **M4** | MuRIL |  | Multilingual Representation | WordPiece (Devanagari native) | **Verified / Pending GPU** |
| **M5** | Sanskrit-BERT |  | Buddhist Sanskrit BERT | **Latin/IAST Script ONLY** | **Blocked / Incompatible** |

> [!WARNING]
> **Critical Audit Finding on M5 (Sanskrit-BERT)**:
> Tokenizer inspection reveals  was trained strictly on Romanized Latin Sanskrit (IAST: ). Passing Devanagari script yields 100%  tokens. As per Part 2 of the Master Instruction (*"If a checkpoint is unavailable or incompatible, document the issue and stop that model's experiment until it is resolved"*), M5 baseline execution is documented and halted pending user approval on transliteration.

---

## 4. Mapping & POS Tagset Verification
- **Confirmed Target Tags**: 24 tags. All observed tokens across Literature and News map to these 24 tags with 0 unknown labels.
- **Ambiguous Prefixes**: 425 ambiguous tokens have been isolated into  and , guaranteeing 100% clean supervision data.
