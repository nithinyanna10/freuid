#!/usr/bin/env bash
# Symlink data/raw -> Kaggle download folder (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/data/the-freuid-challenge-2026-ijcai-ecai"
LINK="$ROOT/data/raw"

if [[ ! -d "$TARGET" ]]; then
  echo "Expected Kaggle data at: $TARGET"
  echo "Download the competition data into data/ first."
  exit 1
fi

if [[ -L "$LINK" ]]; then
  echo "Symlink already exists: $LINK -> $(readlink "$LINK")"
  exit 0
fi

if [[ -e "$LINK" && ! -L "$LINK" ]]; then
  if [[ "$(ls -A "$LINK" 2>/dev/null | grep -v '^\.gitkeep$' || true)" == "" ]]; then
    rmdir "$LINK" 2>/dev/null || rm -rf "$LINK"
  else
    echo "data/raw exists and is not empty. Move or remove it, then re-run."
    exit 1
  fi
fi

ln -s "$(basename "$TARGET")" "$LINK"
echo "Created: $LINK -> $(readlink "$LINK")"
