import json
import ast
from pathlib import Path

CELL_8_TEXT = """import os
import shutil
import zipfile
from pathlib import Path

PROJECT_NAME = "Cross_Domain"

# --------------------------------------------------
# 1. Environment Detection (Mutually Exclusive)
# --------------------------------------------------
is_colab = Path("/content").is_dir()
is_kaggle = not is_colab and Path("/kaggle/working").is_dir()

cwd = Path.cwd().resolve()

if is_colab:
    PROJECT_DIR = Path("/content/Cross_Domain")
    print("Google Colab environment detected. Project root:", PROJECT_DIR)
elif is_kaggle:
    PROJECT_DIR = Path("/kaggle/working/Cross_Domain")
    print("Kaggle environment detected. Project root:", PROJECT_DIR)
elif cwd.name == PROJECT_NAME:
    PROJECT_DIR = cwd
elif cwd.parent.name == PROJECT_NAME:
    PROJECT_DIR = cwd.parent
else:
    candidates = [
        cwd,
        cwd.parent if cwd.name in ["notebooks", "src", "data", "outputs", "configs"] else cwd,
        Path.home() / "Desktop" / "Cross_Domain",
        Path("/home/badshah/Desktop/Cross_Domain").resolve(),
        Path("/root/Desktop/Cross_Domain"),
    ]
    try:
        for u in Path("/home").glob("*"):
            candidates.append(u / "Desktop" / "Cross_Domain")
    except Exception:
        pass
    valid = next((c for c in candidates if (c / "data" / "raw" / "Literature_Corpus.csv").is_file()), None)
    if valid is not None:
        PROJECT_DIR = valid
    else:
        PROJECT_DIR = Path.home() / "Desktop" / PROJECT_NAME
    print("Local environment detected. Project root:", PROJECT_DIR)

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_OUTPUT_DIR = OUTPUT_DIR / "datasets"
MAPPING_REVIEW_DIR = OUTPUT_DIR / "mapping_reviews"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
BILSTM_OUTPUT_DIR = MODEL_OUTPUT_DIR / "bilstm_crf"
XLMR_OUTPUT_DIR = MODEL_OUTPUT_DIR / "xlm_roberta"
INDICBERT_OUTPUT_DIR = MODEL_OUTPUT_DIR / "indicbert"
EVALUATION_DIR = OUTPUT_DIR / "evaluations"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"

for directory in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    DATASET_OUTPUT_DIR,
    MAPPING_REVIEW_DIR,
    MODEL_OUTPUT_DIR,
    BILSTM_OUTPUT_DIR,
    XLMR_OUTPUT_DIR,
    INDICBERT_OUTPUT_DIR,
    EVALUATION_DIR,
    FIGURE_DIR,
    LOG_DIR,
    EXPERIMENT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = MAPPING_REVIEW_DIR
OUT_DIR = MAPPING_REVIEW_DIR
SOURCE_FILE = RAW_DATA_DIR / "Literature_Corpus.csv"
TARGET_FILE = RAW_DATA_DIR / "News_Annotated.csv"
UNLABELED_NEWS_FILE = RAW_DATA_DIR / "News_Unannotated.csv"

# --------------------------------------------------
# 2. Check local directories first if on local machine
# --------------------------------------------------
if not SOURCE_FILE.is_file():
    search_roots = [
        Path("/home/badshah/Desktop/Cross_Domain"),
        *Path("/home").glob("*/Desktop/Cross_Domain"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for sroot in search_roots:
        s_raw = sroot / "data" / "raw"
        if (s_raw / "Literature_Corpus.csv").is_file():
            print(f"Copying missing datasets from {s_raw} -> {RAW_DATA_DIR}...")
            for f in s_raw.glob("*.csv"):
                shutil.copy2(f, RAW_DATA_DIR / f.name)
            for sub in ["outputs/datasets", "configs", "src"]:
                if (sroot / sub).is_dir():
                    shutil.copytree(sroot / sub, PROJECT_DIR / sub, dirs_exist_ok=True)
            break

# --------------------------------------------------
# 3. If running on Kaggle and attached under /kaggle/input, copy them
# --------------------------------------------------
if is_kaggle and not SOURCE_FILE.is_file():
    for f in Path("/kaggle/input").rglob("*"):
        if f.is_file() and f.name in ["Literature_Corpus.csv", "News_Annotated.csv", "News_Unannotated.csv"]:
            shutil.copy2(f, RAW_DATA_DIR / f.name)

# --------------------------------------------------
# 4. Universal Cloud Auto-Download: Fetch dataset directly into PROJECT_DIR
# --------------------------------------------------
if not SOURCE_FILE.is_file():
    print(f"Cloud environment detected! Fetching project datasets directly into {PROJECT_DIR}...")
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_c595dea7dbc740b795edcc95f32ed945"
    k_dir = Path.home() / ".kaggle"
    k_dir.mkdir(parents=True, exist_ok=True)
    (k_dir / "access_token").write_text("KGAT_c595dea7dbc740b795edcc95f32ed945", encoding="utf-8")
    try:
        (k_dir / "access_token").chmod(0o600)
    except Exception:
        pass
    os.system(f"pip install -q kaggle && kaggle datasets download -d badshahkumar/sanskrit-pos-crossdomain --unzip -p {PROJECT_DIR}")
    for b_src, b_dst in [
        (PROJECT_DIR / "kaggle_bundle/outputs/datasets", DATASET_OUTPUT_DIR),
        (PROJECT_DIR / "kaggle_bundle/configs", PROJECT_DIR / "configs"),
        (PROJECT_DIR / "kaggle_bundle/src", PROJECT_DIR / "src"),
        (PROJECT_DIR / "kaggle_bundle/data/raw", RAW_DATA_DIR),
        (PROJECT_DIR / "outputs/datasets", DATASET_OUTPUT_DIR),
    ]:
        if b_src.is_dir() and b_src != b_dst:
            shutil.copytree(b_src, b_dst, dirs_exist_ok=True)

print("Local project:", PROJECT_DIR)
print("Source dataset:", SOURCE_FILE, SOURCE_FILE.is_file())
print("Target dataset:", TARGET_FILE, TARGET_FILE.is_file())
print("Unannotated news:", UNLABELED_NEWS_FILE, UNLABELED_NEWS_FILE.is_file())
"""

CELL_404_TEXT = """# ============================================================
# C.1: LOAD LEAK-FREE DATASET SPLITS AND VERIFY LEAKAGE
# ============================================================

import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Standard target 24 POS tags
TARGET_TAGS = [
    "N", "NS", "NV", "KNV", "SN", "SNV", "SNS", "SAM", "SAMY", "SAMP",
    "KP", "UP", "AVN", "AVP", "AVUP", "SAMB", "AVU", "K", "AKV", "AB",
    "AVC", "DVI", "A", "PU"
]
TAG2ID = {tag: idx for idx, tag in enumerate(TARGET_TAGS)}
ID2TAG = {idx: tag for idx, tag in enumerate(TARGET_TAGS)}

# Ensure PROJECT_DIR is defined
if "PROJECT_DIR" not in globals():
    if Path("/content").is_dir():
        PROJECT_DIR = Path("/content/Cross_Domain")
    elif Path("/kaggle/working").is_dir():
        PROJECT_DIR = Path("/kaggle/working/Cross_Domain")
    else:
        PROJECT_DIR = Path.cwd().resolve()
        if PROJECT_DIR.name != "Cross_Domain" and (PROJECT_DIR / "Cross_Domain").is_dir():
            PROJECT_DIR = PROJECT_DIR / "Cross_Domain"

lit_dir = PROJECT_DIR / "outputs" / "datasets" / "Final_Literature_Data"
datasets_dir = PROJECT_DIR / "outputs" / "datasets"
lit_dir.mkdir(parents=True, exist_ok=True)
datasets_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Self-healing dataset check: ensures files exist before loading
# ------------------------------------------------------------
def ensure_splits_available():
    \"\"\"Checks if split files exist; if not, searches local machine or auto-downloads from Kaggle.\"\"\"
    check_files = [
        lit_dir / "literature_train.json",
        lit_dir / "literature_train.pkl",
        lit_dir / "literature_train.csv",
        datasets_dir / "literature_train.csv",
    ]
    if any(p.is_file() for p in check_files):
        return

    # 1. Search locally in /content, /kaggle, PROJECT_DIR, home
    search_roots = [
        PROJECT_DIR,
        PROJECT_DIR.parent,
        Path("/content"),
        Path("/kaggle"),
        Path.cwd(),
        Path.home() / "Desktop" / "Cross_Domain",
    ]
    for root in search_roots:
        if root.is_dir():
            found = list(root.rglob("literature_train.*"))
            if found:
                found_dir = found[0].parent
                print(f"Found existing dataset splits in {found_dir}. Synchronizing...")
                if found_dir.name == "Final_Literature_Data":
                    shutil.copytree(found_dir, lit_dir, dirs_exist_ok=True)
                    if found_dir.parent.is_dir():
                        shutil.copytree(found_dir.parent, datasets_dir, dirs_exist_ok=True)
                else:
                    shutil.copytree(found_dir, datasets_dir, dirs_exist_ok=True)
                    if (found_dir / "Final_Literature_Data").is_dir():
                        shutil.copytree(found_dir / "Final_Literature_Data", lit_dir, dirs_exist_ok=True)
                return

    # 2. Auto-fetch from Kaggle directly
    print("Split files not found on remote VM. Automatically downloading from Kaggle...")
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_c595dea7dbc740b795edcc95f32ed945"
    k_dir = Path.home() / ".kaggle"
    k_dir.mkdir(parents=True, exist_ok=True)
    (k_dir / "access_token").write_text("KGAT_c595dea7dbc740b795edcc95f32ed945", encoding="utf-8")
    try:
        (k_dir / "access_token").chmod(0o600)
    except Exception:
        pass
    os.system(f"pip install -q kaggle && kaggle datasets download -d badshahkumar/sanskrit-pos-crossdomain --unzip -p {PROJECT_DIR}")
    for b_src, b_dst in [
        (PROJECT_DIR / "kaggle_bundle/outputs/datasets", datasets_dir),
        (PROJECT_DIR / "kaggle_bundle/outputs/datasets/Final_Literature_Data", lit_dir),
        (PROJECT_DIR / "kaggle_bundle/configs", PROJECT_DIR / "configs"),
    ]:
        if b_src.is_dir():
            shutil.copytree(b_src, b_dst, dirs_exist_ok=True)

ensure_splits_available()

def sentence_key(row):
    doc_id = getattr(row, "doc_id", None)
    sentence_id = getattr(row, "sentence_id", None)
    if doc_id is None and isinstance(row, (dict, pd.Series)):
        doc_id = row.get("doc_id", row.get("sentence_id"))
        sentence_id = row.get("sentence_id")
    return f"{doc_id}::{sentence_id}"

def validate_split_leakage(splits_dict):
    # Literature domain check
    lit_seen = {}
    for sname in ["literature_train", "literature_validation", "literature_test"]:
        if sname in splits_dict:
            for row in splits_dict[sname].itertuples():
                key = sentence_key(row)
                if key in lit_seen:
                    raise AssertionError(f"Literature leakage: {key} in {lit_seen[key]} and {sname}")
                lit_seen[key] = sname

    # News domain document check
    news_docs = {}
    for sname in ["news_adaptation_pool", "news_validation", "news_test"]:
        if sname in splits_dict:
            for doc in splits_dict[sname]["doc_id"].unique():
                if doc in news_docs:
                    raise AssertionError(f"News document leakage: {doc} in {news_docs[doc]} and {sname}")
                news_docs[doc] = sname

def load_split_dataset(name, folder, basename, csv_fallback=None):
    \"\"\"Resilient multi-tier dataset loader supporting JSON, Pickle, and direct CSV reconstruction.
    Completely immune to pandas/pickle version incompatibilities across Colab and local environments.\"\"\"
    folder = Path(folder)
    
    # Check primary candidates
    candidates = [
        folder / f"{basename}.json",
        datasets_dir / f"{basename}.json",
        lit_dir / f"{basename}.json",
        folder / f"{basename}.pkl",
        datasets_dir / f"{basename}.pkl",
        lit_dir / f"{basename}.pkl",
    ]
    if csv_fallback:
        candidates.extend([
            folder / csv_fallback,
            datasets_dir / csv_fallback,
            lit_dir / csv_fallback,
        ])

    for p in candidates:
        if p.is_file():
            if p.suffix == ".json":
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return pd.DataFrame(json.load(f))
                except Exception:
                    pass
            elif p.suffix == ".pkl":
                try:
                    return pd.read_pickle(p)
                except Exception:
                    pass
            elif p.suffix == ".csv":
                try:
                    df_csv = pd.read_csv(p)
                    tag_col = "tag" if "tag" in df_csv.columns else "gold_sanskrit_tag"
                    return df_csv.groupby(["doc_id", "sentence_id"], sort=False).agg({
                        "token": list,
                        tag_col: list
                    }).reset_index().rename(columns={"token": "tokens", tag_col: "tags"})
                except Exception:
                    pass

    # Recursive fallback search across PROJECT_DIR and parent directories
    for pattern in [f"{basename}.json", f"{basename}.pkl", csv_fallback]:
        if not pattern:
            continue
        for search_root in [PROJECT_DIR, Path("/content"), Path("/kaggle")]:
            if search_root.is_dir():
                matches = list(search_root.rglob(pattern))
                for m in matches:
                    if m.is_file():
                        if m.suffix == ".json":
                            try:
                                with open(m, "r", encoding="utf-8") as f:
                                    return pd.DataFrame(json.load(f))
                            except Exception:
                                pass
                        elif m.suffix == ".pkl":
                            try:
                                return pd.read_pickle(m)
                            except Exception:
                                pass
                        elif m.suffix == ".csv":
                            try:
                                df_csv = pd.read_csv(m)
                                tag_col = "tag" if "tag" in df_csv.columns else "gold_sanskrit_tag"
                                return df_csv.groupby(["doc_id", "sentence_id"], sort=False).agg({
                                    "token": list,
                                    tag_col: list
                                }).reset_index().rename(columns={"token": "tokens", tag_col: "tags"})
                            except Exception:
                                pass

    raise FileNotFoundError(f"Could not load split '{name}' from JSON, PKL, or CSV.")

splits = {
    "literature_train": load_split_dataset("literature_train", lit_dir, "literature_train", "literature_train.csv"),
    "literature_validation": load_split_dataset("literature_validation", lit_dir, "literature_validation", "literature_validation.csv"),
    "literature_test": load_split_dataset("literature_test", lit_dir, "literature_test_locked", "literature_test_gold.csv"),
    "news_adaptation_pool": load_split_dataset("news_adaptation_pool", datasets_dir, "news_train_pool", "news_adaptation_pool.csv"),
    "news_validation": load_split_dataset("news_validation", datasets_dir, "news_validation", "news_validation.csv"),
    "news_test": load_split_dataset("news_test", datasets_dir, "news_test_locked", "news_test_gold.csv"),
}

validate_split_leakage(splits)
print("SUCCESS: Zero sentence and zero document leakage verified across all splits!")
print()

for name, df in splits.items():
    total_tokens = sum(len(seq) for seq in df["tokens"])
    print(f"{name:25s}: {len(df):4d} sentences, {total_tokens:5d} tokens")

unann_path = PROJECT_DIR / "data" / "raw" / "News_Unannotated.csv" if "PROJECT_DIR" in globals() else Path("data/raw/News_Unannotated.csv")
if not unann_path.is_file():
    cand_unann = list(PROJECT_DIR.rglob("News_Unannotated.csv"))
    if cand_unann:
        unann_path = cand_unann[0]
news_unannotated = pd.read_csv(unann_path)
print(f"news_unannotated         : {len(news_unannotated):4d} sentences")
"""

CELL_419_TEXT = """# ============================================================
# C.16: MASTER EXPERIMENT RUNNER FOR M2 -> M5 (ONE BY ONE)
#       Executes all 4 experimental stages sequentially:
#         Stage 1: Literature In-Domain (Source Training)
#         Stage 2: News In-Domain Reference Benchmark
#         Stage 3: Zero-Shot Cross-Domain Generalization
#         Stage 4: Few-Shot Target Domain Adaptation (50-1000 sents)
#       And saves checkpoints, predictions, tables, and Excel workbook.
# ============================================================

import copy
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Select models to execute:
# To run one model at a time, uncomment one:
#   MODELS_TO_RUN = ["xlm_roberta"]    # M2: XLM-RoBERTa
#   MODELS_TO_RUN = ["indicbert_v2"]   # M3: IndicBERT v2
#   MODELS_TO_RUN = ["muril"]          # M4: MuRIL
#   MODELS_TO_RUN = ["sanskrit_bert"]  # M5: Sanskrit-BERT (Option A Transliteration)
# Or run all remaining models sequentially one by one:
MODELS_TO_RUN = ["xlm_roberta", "indicbert_v2", "muril", "sanskrit_bert"]

lit_train = splits["literature_train"]
lit_val = splits["literature_validation"]
lit_test = splits["literature_test"]
news_pool = splits["news_adaptation_pool"]
news_val = splits["news_validation"]
news_test = splits["news_test"]

def run_full_pipeline_for_model(m_key, seeds=[42], budgets=[50, 100, 200, 500, 1000]):
    spec = MODEL_SPECS[m_key]
    display_name = spec["display_name"]
    model_id_tag = spec["id"]
    
    print()
    print("=" * 75)
    print(f"STARTING FULL EXPERIMENTAL SUITE: {display_name} ({model_id_tag})")
    print(f"   Model: {spec.get('hf_model_id', m_key)}")
    print(f"   IAST Transliteration: {spec.get('is_iast', False)}")
    print("=" * 75)

    config = load_model_config(spec)
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # ----------------------------------------------------
    # STAGE 1: Literature In-Domain Training & Evaluation
    # ----------------------------------------------------
    print(f"\\n>>> [STAGE 1/4] Literature In-Domain: {display_name}...")
    exp_id, run_dir = create_hierarchical_run_directory(EXP_ROOT, model=display_name, setting="literature_in_domain", seed=seed)
    started_at = datetime.now(timezone.utc).isoformat()

    tagger = TransformerTagger(model_id=spec["hf_model_id"], is_iast=spec.get("is_iast", False)).to(DEVICE)
    best_tagger, val_metrics, _ = train_transformer(
        tagger=tagger, train_df=lit_train, val_df=lit_val, device=DEVICE, run_dir=run_dir, config=config
    )
    test_preds = best_tagger.predict_words(list(lit_test["tokens"]), device=DEVICE, batch_size=config.get("batch_size", 16))
    summary_lit, per_tag_lit, report_lit, conf_mat_lit = evaluate_predictions(lit_test, test_preds)
    save_evaluation_results(run_dir, summary_lit, per_tag_lit, report_lit, conf_mat_lit, test_preds, lit_test)
    append_registry(
        REGISTRY_PATH, experiment_id=exp_id, model=display_name, setting="literature_in_domain",
        annotation_budget=0, seed=seed, status="SUCCESS", run_dir=run_dir,
        validation_macro_f1=val_metrics.get("macro_f1"), test_macro_f1=summary_lit["macro_f1"],
        test_accuracy=summary_lit["accuracy"], test_weighted_f1=summary_lit["weighted_f1"], started_at=started_at
    )
    lit_chk_dir = run_dir / "checkpoint"
    print(f"Stage 1 Done: Lit Val F1={val_metrics.get('macro_f1', 0)*100:.2f}%, Lit Test Acc={summary_lit['accuracy']*100:.2f}%, Test Macro F1={summary_lit['macro_f1']*100:.2f}%")

    # Free GPU memory
    del tagger, best_tagger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # STAGE 2: News In-Domain Reference Benchmark
    # ----------------------------------------------------
    print(f"\\n>>> [STAGE 2/4] News In-Domain Reference: {display_name}...")
    exp_id, run_dir = create_hierarchical_run_directory(EXP_ROOT, model=display_name, setting="news_in_domain", seed=seed)
    started_at = datetime.now(timezone.utc).isoformat()

    news_tagger = TransformerTagger(model_id=spec["hf_model_id"], is_iast=spec.get("is_iast", False)).to(DEVICE)
    best_news_tagger, news_val_metrics, _ = train_transformer(
        tagger=news_tagger, train_df=news_pool, val_df=news_val, device=DEVICE, run_dir=run_dir, config=config
    )
    test_preds = best_news_tagger.predict_words(list(news_test["tokens"]), device=DEVICE, batch_size=config.get("batch_size", 16))
    summary_news, per_tag_news, report_news, conf_mat_news = evaluate_predictions(news_test, test_preds)
    save_evaluation_results(run_dir, summary_news, per_tag_news, report_news, conf_mat_news, test_preds, news_test)
    append_registry(
        REGISTRY_PATH, experiment_id=exp_id, model=display_name, setting="news_in_domain",
        annotation_budget=0, seed=seed, status="SUCCESS", run_dir=run_dir,
        validation_macro_f1=news_val_metrics.get("macro_f1"), test_macro_f1=summary_news["macro_f1"],
        test_accuracy=summary_news["accuracy"], test_weighted_f1=summary_news["weighted_f1"], started_at=started_at
    )
    print(f"Stage 2 Done: News Val F1={news_val_metrics.get('macro_f1', 0)*100:.2f}%, News Test Acc={summary_news['accuracy']*100:.2f}%, Test Macro F1={summary_news['macro_f1']*100:.2f}%")

    # Free GPU memory
    del news_tagger, best_news_tagger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # STAGE 3: Zero-Shot Cross-Domain Generalization
    # ----------------------------------------------------
    print(f"\\n>>> [STAGE 3/4] Zero-Shot Cross-Domain (Literature -> News Locked Test): {display_name}...")
    exp_id, run_dir = create_hierarchical_run_directory(EXP_ROOT, model=display_name, setting="zero_shot", seed=seed)
    started_at = datetime.now(timezone.utc).isoformat()

    lit_loaded = TransformerTagger.load(lit_chk_dir, device=DEVICE)
    test_preds = lit_loaded.predict_words(list(news_test["tokens"]), device=DEVICE, batch_size=config.get("batch_size", 16))
    summary_zs, per_tag_zs, report_zs, conf_mat_zs = evaluate_predictions(news_test, test_preds)
    save_evaluation_results(run_dir, summary_zs, per_tag_zs, report_zs, conf_mat_zs, test_preds, news_test)
    append_registry(
        REGISTRY_PATH, experiment_id=exp_id, model=display_name, setting="zero_shot",
        annotation_budget=0, seed=seed, status="SUCCESS", run_dir=run_dir,
        validation_macro_f1=None, test_macro_f1=summary_zs["macro_f1"],
        test_accuracy=summary_zs["accuracy"], test_weighted_f1=summary_zs["weighted_f1"], started_at=started_at
    )
    abs_drop = summary_lit["macro_f1"] - summary_zs["macro_f1"]
    rel_drop = (abs_drop / summary_lit["macro_f1"]) * 100 if summary_lit["macro_f1"] else 0
    print(f"Stage 3 Done: Zero-Shot News Acc={summary_zs['accuracy']*100:.2f}%, Test Macro F1={summary_zs['macro_f1']*100:.2f}%")
    print(f"   Domain Degradation: Absolute Drop = -{abs_drop*100:.2f} pp, Relative Drop = -{rel_drop:.2f}%")

    # Free GPU memory
    del lit_loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # STAGE 4: Few-Shot Target Domain Adaptation
    # ----------------------------------------------------
    print(f"\\n>>> [STAGE 4/4] Few-Shot Adaptation (Budgets: {budgets}): {display_name}...")
    for s in seeds:
        nested_subsets = get_nested_few_shot_subsets(news_pool, budgets=budgets, seed=s)
        for b in budgets:
            sub_df = nested_subsets[b]
            token_count = sum(len(t) for t in sub_df["tokens"])
            exp_id, run_dir = create_hierarchical_run_directory(EXP_ROOT, model=display_name, setting="few_shot", annotation_budget=b, seed=s)
            started_at = datetime.now(timezone.utc).isoformat()

            torch.manual_seed(s)
            ft_tagger = TransformerTagger.load(lit_chk_dir, device=DEVICE)
            ft_config = copy.deepcopy(config)
            ft_config["learning_rate"] = config.get("learning_rate", 3e-5) * 0.7
            ft_config["epochs"] = 6
            ft_config["early_stopping_patience"] = 2

            best_ft, ft_val, _ = train_transformer(tagger=ft_tagger, train_df=sub_df, val_df=news_val, device=DEVICE, run_dir=run_dir, config=ft_config)
            test_preds = best_ft.predict_words(list(news_test["tokens"]), device=DEVICE, batch_size=config.get("batch_size", 16))
            summary_fs, per_tag_fs, report_fs, conf_mat_fs = evaluate_predictions(news_test, test_preds)
            save_evaluation_results(run_dir, summary_fs, per_tag_fs, report_fs, conf_mat_fs, test_preds, news_test)
            append_registry(
                REGISTRY_PATH, experiment_id=exp_id, model=display_name, setting="few_shot",
                annotation_budget=b, seed=s, status="SUCCESS", run_dir=run_dir,
                validation_macro_f1=ft_val.get("macro_f1"), test_macro_f1=summary_fs["macro_f1"],
                test_accuracy=summary_fs["accuracy"], test_weighted_f1=summary_fs["weighted_f1"], started_at=started_at
            )
            gain = (summary_fs["macro_f1"] - summary_zs["macro_f1"]) * 100
            print(f"   --> Budget {b:4d} sent ({token_count:5d} tokens): Val F1={ft_val.get('macro_f1', 0)*100:.2f}%, Test Macro F1={summary_fs['macro_f1']*100:.2f}%, Acc={summary_fs['accuracy']*100:.2f}% (Gain: +{gain:.2f} pp)")

            # Clean GPU memory per budget run
            del ft_tagger, best_ft
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\\nALL 4 EXPERIMENTAL STAGES COMPLETED FOR {display_name}!")

# Execute models sequentially
for model_key in MODELS_TO_RUN:
    run_full_pipeline_for_model(model_key)

print()
print("=" * 75)
print("ALL REQUESTED MODELS FINISHED! GENERATING CONSOLIDATED BENCHMARK TABLES")
print("=" * 75)

BASELINE_OUT_DIR = PROJECT_DIR / "outputs" / "baseline_experiments"
BASELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)
EVALS_DIR = PROJECT_DIR / "outputs" / "evaluations"
EVALS_DIR.mkdir(parents=True, exist_ok=True)

if REGISTRY_PATH.is_file():
    reg_df = pd.read_csv(REGISTRY_PATH)

    # 1. In-Domain Table
    in_domain_df = reg_df[reg_df["setting"].isin(["literature_in_domain", "news_in_domain"]) & (reg_df["status"] == "SUCCESS")].copy()
    in_domain_summary = []
    for m_name in in_domain_df["model"].unique():
        m_runs = in_domain_df[in_domain_df["model"] == m_name]
        lit_run = m_runs[m_runs["setting"] == "literature_in_domain"]
        news_run = m_runs[m_runs["setting"] == "news_in_domain"]
        in_domain_summary.append({
            "Model": m_name,
            "Literature_Val_Macro_F1": lit_run["validation_macro_f1"].values[0] if not lit_run.empty else None,
            "Literature_Test_Macro_F1": lit_run["test_macro_f1"].values[0] if not lit_run.empty else None,
            "Literature_Test_Accuracy": lit_run["test_accuracy"].values[0] if not lit_run.empty else None,
            "News_Val_Macro_F1": news_run["validation_macro_f1"].values[0] if not news_run.empty else None,
            "News_Test_Macro_F1": news_run["test_macro_f1"].values[0] if not news_run.empty else None,
            "News_Test_Accuracy": news_run["test_accuracy"].values[0] if not news_run.empty else None,
        })
    t1_df = pd.DataFrame(in_domain_summary)
    t1_df.to_csv(BASELINE_OUT_DIR / "literature_in_domain_results.csv", index=False)
    t1_df.to_csv(EVALS_DIR / "table1_in_domain.csv", index=False)

    # 2. Zero-Shot Cross-Domain Table
    zs_df = reg_df[(reg_df["setting"] == "zero_shot") & (reg_df["status"] == "SUCCESS")].copy()
    zs_summary = []
    for _, row in zs_df.iterrows():
        m_name = row["model"]
        lit_match = in_domain_df[(in_domain_df["model"] == m_name) & (in_domain_df["setting"] == "literature_in_domain")]
        lit_f1 = lit_match["test_macro_f1"].values[0] if not lit_match.empty else None
        zs_f1 = row["test_macro_f1"]
        abs_drop = (lit_f1 - zs_f1) if (lit_f1 is not None and zs_f1 is not None) else None
        rel_drop = (abs_drop / lit_f1 * 100) if (lit_f1 and abs_drop is not None) else None
        zs_summary.append({
            "Model": m_name,
            "Literature_InDomain_Macro_F1": lit_f1,
            "News_ZeroShot_Macro_F1": zs_f1,
            "Absolute_F1_Drop": abs_drop,
            "Relative_F1_Drop_Pct": rel_drop,
            "ZeroShot_Accuracy": row["test_accuracy"],
            "ZeroShot_Weighted_F1": row["test_weighted_f1"],
        })
    t2_df = pd.DataFrame(zs_summary)
    t2_df.to_csv(BASELINE_OUT_DIR / "cross_domain_zero_shot_results.csv", index=False)
    t2_df.to_csv(EVALS_DIR / "table2_zero_shot.csv", index=False)

    # 3. Few-Shot Adaptation Progression Table
    fs_df = reg_df[reg_df["setting"].isin(["few_shot", "zero_shot"]) & (reg_df["status"] == "SUCCESS")].copy()
    fs_df.loc[fs_df["setting"] == "zero_shot", "annotation_budget"] = 0
    t3_pivot = fs_df.pivot_table(
        index="model",
        columns="annotation_budget",
        values="test_macro_f1",
        aggfunc="mean"
    ).reset_index()
    t3_pivot.to_csv(BASELINE_OUT_DIR / "few_shot_results.csv", index=False)
    t3_pivot.to_csv(EVALS_DIR / "table3_few_shot.csv", index=False)

    # 4. Master Excel Workbook
    try:
        import openpyxl
        excel_path = BASELINE_OUT_DIR / "baseline_experiment_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            t1_df.to_excel(writer, sheet_name="In_Domain_Benchmarks", index=False)
            t2_df.to_excel(writer, sheet_name="Zero_Shot_Degradation", index=False)
            t3_pivot.to_excel(writer, sheet_name="Few_Shot_Progression", index=False)
            reg_df.to_excel(writer, sheet_name="Experiment_Registry", index=False)
        print(f"Generated comparison tables and Excel workbook at: {excel_path}\\n")
    except Exception as ex:
        print("Note: Excel export skipped (openpyxl):", ex)

    print("=== Table 1: In-Domain Performance ===")
    display(t1_df)
    print("\\n=== Table 2: Zero-Shot Cross-Domain Generalization ===")
    display(t2_df)
    print("\\n=== Table 3: Few-Shot Adaptation Progression ===")
    display(t3_pivot)
"""

# Verify AST
ast.parse(CELL_8_TEXT)
ast.parse(CELL_404_TEXT)
ast.parse(CELL_419_TEXT)
print("AST validation passed for CELL_8, CELL_404, CELL_419!")

notebook_paths = [
    Path("notebooks/optimized_sanskrit_pos_notebook.ipynb"),
    Path("notebooks/Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb"),
    Path("outputs/kaggle_dataset_upload/kaggle_bundle/notebooks/optimized_sanskrit_pos_notebook.ipynb"),
    Path("outputs/kaggle_dataset_upload/kaggle_bundle/notebooks/Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb"),
    Path("outputs/kaggle_kernel/optimized_sanskrit_pos_notebook.ipynb"),
]

for nb_path in notebook_paths:
    if not nb_path.is_file():
        continue
    print(f"Updating {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 8
    nb["cells"][8]["source"] = [l + "\n" for l in CELL_8_TEXT.split("\n")]
    if nb["cells"][8]["source"] and nb["cells"][8]["source"][-1].endswith("\n\n"):
        nb["cells"][8]["source"][-1] = nb["cells"][8]["source"][-1][:-1]

    # Cell 404
    nb["cells"][404]["source"] = [l + "\n" for l in CELL_404_TEXT.split("\n")]
    if nb["cells"][404]["source"] and nb["cells"][404]["source"][-1].endswith("\n\n"):
        nb["cells"][404]["source"][-1] = nb["cells"][404]["source"][-1][:-1]

    # Cell 407 (append load_model_config if not present)
    c407_src = "".join(nb["cells"][407]["source"])
    if "def load_model_config(" not in c407_src:
        load_config_code = '''

def load_model_config(spec):
    \"\"\"Resiliently load model hyperparameter configuration, falling back to verified defaults.\"\"\"
    candidates = [
        PROJECT_DIR / spec.get("config_path", ""),
        PROJECT_DIR / "kaggle_bundle" / spec.get("config_path", ""),
        Path("configs") / Path(spec.get("config_path", "")).name,
    ]
    for c in candidates:
        if c and c.is_file():
            try:
                with open(c, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    if spec.get("type") == "bilstm":
        return {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "early_stopping_patience": 3, "seed": 42}
    return {"epochs": 10, "batch_size": 16, "learning_rate": 3e-5, "weight_decay": 0.01, "warmup_ratio": 0.1, "early_stopping_patience": 3, "seed": 42}
'''
        nb["cells"][407]["source"] = [l + "\n" for l in (c407_src + load_config_code).split("\n")]

    # Cell 411, 412, 413, 414
    for c_idx in [411, 412, 413]:
        src = "".join(nb["cells"][c_idx]["source"])
        target = 'with open(PROJECT_DIR / spec["config_path"], encoding="utf-8") as f:\n        config = json.load(f)'
        if target in src:
            nb["cells"][c_idx]["source"] = [l + "\n" for l in src.replace(target, "config = load_model_config(spec)").split("\n")]

    c414_src = "".join(nb["cells"][414]["source"])
    c414_target = 'with open(PROJECT_DIR / spec["config_path"], encoding="utf-8") as f:\n        base_config = json.load(f)'
    if c414_target in c414_src:
        nb["cells"][414]["source"] = [l + "\n" for l in c414_src.replace(c414_target, "base_config = load_model_config(spec)").split("\n")]

    # Cell 419
    nb["cells"][419]["source"] = [l + "\n" for l in CELL_419_TEXT.split("\n")]
    if nb["cells"][419]["source"] and nb["cells"][419]["source"][-1].endswith("\n\n"):
        nb["cells"][419]["source"][-1] = nb["cells"][419]["source"][-1][:-1]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved {nb_path}")

print("All notebooks processed successfully.")
