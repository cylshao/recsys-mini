#!/usr/bin/env bash
# Download and extract MovieLens-1M.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
TARGET_DIR="${RAW_DIR}/ml-1m"
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"

mkdir -p "${RAW_DIR}"

if [[ -d "${TARGET_DIR}" && -f "${TARGET_DIR}/ratings.dat" ]]; then
  echo "✓ MovieLens-1M already exists: ${TARGET_DIR}"
  exit 0
fi

echo "▶ Downloading MovieLens-1M..."
cd "${RAW_DIR}"
if command -v curl >/dev/null 2>&1; then
  curl -L -o ml-1m.zip "${URL}"
else
  wget -O ml-1m.zip "${URL}"
fi

echo "▶ Extracting..."
unzip -o ml-1m.zip
rm ml-1m.zip

echo "✓ Done. Files located at: ${TARGET_DIR}"
ls -lh "${TARGET_DIR}"
