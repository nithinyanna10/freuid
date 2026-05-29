"""Build a Kaggle-ready submission CSV from test ids and fraud scores."""

from pathlib import Path

import pandas as pd


def write_submission(
    ids: pd.Series,
    scores: pd.Series,
    output_path: Path | str,
) -> Path:
    """Write id,score CSV. Scores must be in [0, 1]."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": ids.astype(str), "score": scores.astype(float).clip(0.0, 1.0)})
    df.to_csv(out, index=False)
    return out
