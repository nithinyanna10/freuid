#!/usr/bin/env python3
"""Check competition data layout and image path resolution."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.dataset import load_train_metadata
from src.paths import (
    DATA_RAW,
    find_sample_submission,
    find_test_csv,
    find_train_csv,
    get_data_root,
    resolve_image_path,
)


def main() -> int:
    root = get_data_root()
    print(f"Data root: {root}")
    if DATA_RAW.is_symlink():
        print(f"Symlink: data/raw -> {DATA_RAW.readlink()}")
    elif not DATA_RAW.exists():
        print("Tip: run scripts/setup_data.sh to symlink data/raw -> Kaggle folder")

    try:
        train_csv = find_train_csv(root)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"Train CSV: {train_csv.name}")
    metadata = load_train_metadata(root)
    print(f"Rows: {len(metadata)}, columns: {list(metadata.columns)}")
    print(f"Labels:\n{metadata['label'].value_counts().sort_index()}")

    if "is_digital" in metadata.columns:
        print(f"is_digital:\n{metadata['is_digital'].value_counts().sort_index()}")
    if "type" in metadata.columns:
        print(f"Document types ({metadata['type'].nunique()}):\n{metadata['type'].value_counts()}")

    missing = 0
    for _, row in metadata.iterrows():
        try:
            resolve_image_path(root, row["image_path"])
        except FileNotFoundError:
            missing += 1
    print(f"Resolvable images: {len(metadata) - missing}/{len(metadata)}")
    if missing:
        return 1

    test_csv = find_test_csv(root)
    if test_csv:
        import pandas as pd

        test = pd.read_csv(test_csv)
        print(f"test.csv: {len(test)} rows")
    else:
        print("test.csv: not present (expected for sample release)")

    try:
        sample = find_sample_submission(root)
        import pandas as pd

        sub = pd.read_csv(sample)
        print(f"sample_submission.csv: {len(sub)} rows, columns={list(sub.columns)}")
    except FileNotFoundError:
        print("sample_submission.csv: missing")

    for dirname in ("train", "test", "train_sample"):
        d = root / dirname
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file() and not _.name.startswith("."))
            print(f"{dirname}/: {n} files (recursive)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
