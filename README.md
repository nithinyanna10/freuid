# FREUID Challenge 2026 (IJCAI-ECAI)

Binary fraud detection on identity document images: **0 = bona-fide**, **1 = attack/fraud**.

Competition focus: physical manipulations, GenAI-driven digital edits, print-and-capture forgeries, and mixed attack types. Metric: **FREUID Score** (AuDET + APCER at 1% BPCER). Submit calibrated probabilities in `[0, 1]`; higher = more likely fraud.

## Data layout

Download from Kaggle into `data/the-freuid-challenge-2026-ijcai-ecai/`, then symlink:

```bash
bash scripts/setup_data.sh   # creates data/raw -> Kaggle folder
```

**Sample release (current):**

```
data/raw/
  train_sample_labels.csv
  sample_submission.csv
  train_sample/train_sample/*.jpeg
```

**Full release (expected ~June):**

```
data/raw/
  train.csv, test.csv, sample_submission.csv
  train/, test/
```

`train.csv` columns: `id`, `image_path`, `label`, `is_digital`, `type`  
`type` format: `<country>/<document-type>` (e.g. `USA/DL`, `SWITZERLAND/ID`)

## Quick start

```bash
cd freuid-2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_data.sh
python scripts/verify_data.py
python scripts/train.py
python scripts/predict.py --split train
```

Outputs: `data/processed/baseline.pt`, `submissions/submission.csv` (`id,score`).

## Project structure

| Path | Purpose |
|------|---------|
| `data/raw/` | Competition download (gitignored) |
| `data/processed/` | Cached features / splits |
| `notebooks/` | EDA and experiments |
| `src/` | Training, inference, submission helpers |
| `submissions/` | Generated `id,score` CSVs |
| `configs/` | Experiment configs |
| `scripts/` | CLI utilities |

## Submission format

```csv
id,score
000001,0.0123
```

One row per `test.csv` id. Positive class = fraud (`label=1`).

## Notes

- Sample training data may be available before the full release (~June).
- Treat cross-document generalization (country, script, layout) as a first-class goal, not only template-specific artifacts.
