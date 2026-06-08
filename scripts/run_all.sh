#!/usr/bin/env bash
# Run the whole pipeline end to end.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

bash scripts/download_data.sh
python scripts/01_eda.py
python scripts/02_prepare_samples.py
python scripts/03_train_recall.py
python scripts/04_train_rank.py
python scripts/05_offline_eval.py
