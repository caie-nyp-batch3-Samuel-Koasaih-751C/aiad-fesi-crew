"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.19.14
"""
from __future__ import annotations
import os, random, shutil
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _is_img(name: str, exts: Iterable[str] = VALID_EXTS) -> bool:
    return any(name.lower().endswith(e) for e in exts)

def _copy_many(files: List[Path], src_class: Path, dst_class: Path):
    dst_class.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(src_class / f.name, dst_class / f.name)

def stratified_split_folders(
    src_root: str,
    dst_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,           # test gets the rest
    seed: int = 42,
    limit_per_class: int | None = None # optional cap for faster experiments
) -> Dict[str, int]:
    """
    Reads class subfolders from src_root and writes:
      dst_root/train/<class>/...
      dst_root/val/<class>/...
      dst_root/test/<class>/...
    Keeps class balance by splitting within each class.
    """

    src = Path(src_root)
    dst = Path(dst_root)
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "val").mkdir(parents=True, exist_ok=True)
    (dst / "test").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0, "classes": 0}

    for class_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        cls = class_dir.name
        files = [f for f in class_dir.iterdir() if f.is_file() and _is_img(f.name)]
        if not files:
            continue

        if limit_per_class:
            files = files[:limit_per_class]

        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files   = files[n_train:n_train + n_val]
        test_files  = files[n_train + n_val:]

        _copy_many(train_files, class_dir, dst / "train" / cls)
        _copy_many(val_files,   class_dir, dst / "val" / cls)
        _copy_many(test_files,  class_dir, dst / "test" / cls)

        counts["train"] += len(train_files)
        counts["val"]   += len(val_files)
        counts["test"]  += len(test_files)
        counts["classes"] += 1

    counts["dst_root"] = str(dst)
    return counts
