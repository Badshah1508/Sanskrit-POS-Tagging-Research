"""Kaggle environment synchronization and dataset packaging utilities.

Provides reliable bundling of datasets, configs, and scripts for Kaggle GPU execution,
and automated unpacking and synchronization of remote experiment outputs back to the local project.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


def create_kaggle_bundle(
    project_root: Path,
    output_bundle_path: Path | None = None,
) -> Path:
    """Bundle local datasets, configs, and src into a single zip archive for Kaggle upload."""
    root = Path(project_root).resolve()
    if output_bundle_path is None:
        output_bundle_path = root / "outputs" / "kaggle_bundle.zip"
    else:
        output_bundle_path = Path(output_bundle_path).resolve()

    output_bundle_path.parent.mkdir(parents=True, exist_ok=True)

    items_to_bundle = [
        ("data/raw/Literature_Corpus.csv", "data/raw/Literature_Corpus.csv"),
        ("data/raw/News_Annotated.csv", "data/raw/News_Annotated.csv"),
        ("data/raw/News_Unannotated.csv", "data/raw/News_Unannotated.csv"),
        ("outputs/datasets/Final_Literature_Data", "outputs/datasets/Final_Literature_Data"),
        ("outputs/datasets/news_train_pool.pkl", "outputs/datasets/news_train_pool.pkl"),
        ("outputs/datasets/news_validation.pkl", "outputs/datasets/news_validation.pkl"),
        ("outputs/datasets/news_test_locked.pkl", "outputs/datasets/news_test_locked.pkl"),
        ("outputs/datasets/split_manifest.json", "outputs/datasets/split_manifest.json"),
        ("configs", "configs"),
        ("src", "src"),
        ("notebooks/optimized_sanskrit_pos_notebook.ipynb", "optimized_sanskrit_pos_notebook.ipynb"),
        ("notebooks/Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb", "Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb"),
    ]

    with zipfile.ZipFile(output_bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_src, rel_dst in items_to_bundle:
            src_path = root / rel_src
            if src_path.is_file():
                zf.write(src_path, arcname=rel_dst)
            elif src_path.is_dir():
                for file in src_path.rglob("*"):
                    if file.is_file() and "__pycache__" not in file.parts and not file.name.endswith(".pyc"):
                        arcname = str(Path(rel_dst) / file.relative_to(src_path))
                        zf.write(file, arcname=arcname)

    print(f"Kaggle bundle created successfully at: {output_bundle_path} ({output_bundle_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_bundle_path


def generate_kaggle_dataset_metadata(project_root: Path, slug: str = "sanskrit-pos-crossdomain") -> Path:
    """Generate dataset-metadata.json for the official Kaggle CLI."""
    meta = {
        "title": "Sanskrit Cross-Domain POS Datasets and Configs",
        "id": f"your-kaggle-username/{slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    meta_path = Path(project_root) / "outputs" / "kaggle_dataset_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def unpack_kaggle_results(
    kaggle_results_archive: Path,
    project_root: Path,
) -> None:
    """Unpack downloaded Kaggle experiment artifacts into local outputs/experiments/."""
    root = Path(project_root).resolve()
    archive = Path(kaggle_results_archive).resolve()
    
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    extract_tmp = root / "outputs" / "_kaggle_tmp"
    extract_tmp.mkdir(parents=True, exist_ok=True)
    
    shutil.unpack_archive(str(archive), str(extract_tmp))
    
    # Merge into outputs/experiments
    target_exp = root / "outputs" / "experiments"
    target_exp.mkdir(parents=True, exist_ok=True)

    # Walk and copy
    for item in extract_tmp.rglob("*"):
        if item.is_file():
            rel = item.relative_to(extract_tmp)
            dst = target_exp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            
    shutil.rmtree(extract_tmp, ignore_errors=True)
    print(f"Successfully synchronized Kaggle results into {target_exp}")
