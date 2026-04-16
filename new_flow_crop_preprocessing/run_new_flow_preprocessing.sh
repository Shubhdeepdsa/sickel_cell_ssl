#!/usr/bin/env bash
set -euo pipefail

uv run python build_masked_resized_dataset.py "$@"
