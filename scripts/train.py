#!/usr/bin/env python3
"""Minimal baseline training loop for FREUID (sample or full train set)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.dataset import FreuidDataset, load_train_metadata
from src.device import get_device
from src.model import build_model
from src.paths import DATA_PROCESSED, get_data_root
from src.transforms import build_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FREUID fraud baseline")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=DATA_PROCESSED / "baseline.pt")
    return p.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, leave=False, disable=len(loader) < 2):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Data root: {get_data_root()}")
    print(f"Device: {device}")

    metadata = load_train_metadata()
    train_idx, val_idx = train_test_split(
        np.arange(len(metadata)),
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=metadata["label"] if metadata["label"].nunique() > 1 else None,
    )

    train_tf = build_transforms(args.image_size, train=True)
    val_tf = build_transforms(args.image_size, train=False)
    full_train = FreuidDataset(metadata, transform=train_tf)
    full_val = FreuidDataset(metadata, transform=val_tf)

    train_loader = DataLoader(
        Subset(full_train, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.backbone, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                },
                args.checkpoint,
            )

    print(f"Saved checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
