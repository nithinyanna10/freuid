from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

KAGGLE_DIR_NAME = "the-freuid-challenge-2026-ijcai-ecai"


def get_data_root() -> Path:
    """Resolved competition data directory (symlink or Kaggle folder)."""
    if DATA_RAW.is_dir() or DATA_RAW.is_symlink():
        return DATA_RAW.resolve()
    fallback = PROJECT_ROOT / "data" / KAGGLE_DIR_NAME
    if fallback.is_dir():
        return fallback.resolve()
    return DATA_RAW.resolve()


def find_train_csv(root: Path | None = None) -> Path:
    root = root or get_data_root()
    for name in ("train.csv", "train_sample_labels.csv"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No train CSV under {root}")


def find_sample_submission(root: Path | None = None) -> Path:
    root = root or get_data_root()
    path = root / "sample_submission.csv"
    if path.is_file():
        return path
    raise FileNotFoundError(f"No sample_submission.csv under {root}")


def find_test_csv(root: Path | None = None) -> Path | None:
    root = root or get_data_root()
    path = root / "test.csv"
    return path if path.is_file() else None


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    """Resolve CSV image_path to an on-disk file (handles nested train_sample/)."""
    rel = Path(image_path)
    name = rel.name
    parent = rel.parent

    candidates = [
        data_root / rel,
        data_root / parent / name,
        data_root / "train_sample" / name,
        data_root / "train_sample" / "train_sample" / name,
        data_root / "train" / name,
        data_root / name,
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"Image not found for {image_path!r} under {data_root}")
