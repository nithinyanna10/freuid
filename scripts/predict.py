#!/usr/bin/env python3
"""Run inference and write a competition-style submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.dataset import FreuidDataset, load_train_metadata
from src.device import get_device
from src.model import build_model
from src.paths import SUBMISSIONS_DIR, find_sample_submission, find_test_csv, get_data_root
from src.submission import write_submission
from src.transforms import build_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FREUID inference + submission")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data" / "processed" / "baseline.pt",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=SUBMISSIONS_DIR / "submission.csv",
    )
    p.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="train = sample train ids (for pipeline check); test = test.csv when available",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def load_metadata(split: str) -> pd.DataFrame:
    root = get_data_root()
    if split == "test":
        test_csv = find_test_csv(root)
        if test_csv is None:
            raise FileNotFoundError("test.csv not found — use --split train for sample data")
        return pd.read_csv(test_csv)
    return load_train_metadata(root)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(ckpt.get("image_size", 384))
    backbone = str(ckpt.get("backbone", "resnet18"))

    metadata = load_metadata(args.split)
    if "image_path" not in metadata.columns:
        raise ValueError(f"Expected image_path column, got {list(metadata.columns)}")

    dataset = FreuidDataset(metadata, transform=build_transforms(image_size, train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids: list[str] = []
    scores: list[float] = []
    for images, _, batch_ids in tqdm(loader, disable=len(loader) < 2):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        ids.extend(batch_ids)
        scores.extend(probs)

    out = write_submission(pd.Series(ids), pd.Series(scores), args.output)
    print(f"Wrote {len(ids)} rows to {out}")

    # Compare shape to sample submission when present
    try:
        sample = pd.read_csv(find_sample_submission())
        print(f"sample_submission.csv has {len(sample)} rows (reference)")
    except FileNotFoundError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
