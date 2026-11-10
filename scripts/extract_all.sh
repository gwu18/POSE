#!/usr/bin/env bash
# Extract landmarks for every video in data/raw/<pid>/, then QA the labels.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m posetcn.extract --all "$@"
python tools/qa_labels.py
