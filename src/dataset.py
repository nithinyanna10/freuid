from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.paths import find_train_csv, get_data_root, resolve_image_path


def load_train_metadata(root: Path | None = None) -> pd.DataFrame:
    root = root or get_data_root()
    df = pd.read_csv(find_train_csv(root))
    df["label"] = df["label"].astype(int)
    if "is_digital" in df.columns:
        df["is_digital"] = df["is_digital"].map(_to_bool_int)
    return df


def _to_bool_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Cannot parse is_digital value: {value!r}")


class FreuidDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        data_root: Path | None = None,
        transform=None,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.data_root = (data_root or get_data_root()).resolve()
        self.transform = transform
        self._paths = [
            resolve_image_path(self.data_root, row["image_path"])
            for _, row in self.metadata.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.metadata.iloc[index]
        image = Image.open(self._paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label, str(row["id"])
