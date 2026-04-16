# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python research pipeline for sickle-cell image analysis. Core scripts live at the repo root, grouped by stage:
- `eda.py`, `split_protocol.py`, `crop_creation.py` for data prep
- `train_*.py` for model training
- `evaluate_*.py` and `visualize_*.py` for evaluation and reporting
- `prototype_*.py` for experimental variants

Generated artifacts are written to stage/output folders such as `split_outputs/`, `crop_outputs/`, `stage5_outputs/`, `stage6_outputs/`, `stage7_outputs/`, and `visual_reports/`. Keep source code changes separate from regenerated figures, CSVs, and cached data.

## Build, Test, and Development Commands
Use `uv` with the checked-in `uv.lock` when possible:
- `uv sync` installs the locked dependencies from `pyproject.toml`
- `uv run python split_protocol.py` regenerates split manifests
- `uv run python crop_creation.py` rebuilds crop datasets
- `uv run python train_stage5.py` runs the full-label training stage
- `uv run python train_stage6_low_label.py` runs low-label experiments
- `uv run python visualize_stage5_stage7.py` or `uv run python visualize_grayscale_truth_test.py` regenerates comparison figures and report outputs

If a script exposes CLI flags, check `uv run python <script>.py --help`.

## Coding Style & Naming Conventions
Follow modern Python style: 4-space indentation, `snake_case` for functions/files, `UPPER_SNAKE_CASE` for constants, and type hints for new code. Prefer `pathlib.Path`, explicit dataclasses for records, and small pure helpers over deeply nested script logic. Keep comments brief and only for non-obvious processing steps.

## Testing Guidelines
There is no dedicated `pytest` suite yet. Treat `visualize_grayscale_truth_test.py` and similar stage scripts as smoke tests: they should run without errors and reproduce the expected artifacts. When adding tests, name them `test_*.py` and place them alongside the code or under a `tests/` directory.

## Commit & Pull Request Guidelines
Recent commits use short, stage-oriented prefixes such as `feat(...)`, `Updated`, and `Implemented`. Keep commit subjects concise and imperative, and mention the affected pipeline stage when relevant. Pull requests should summarize the scientific change, list commands run, and include updated figures or tables when outputs change. Note any data or configuration assumptions explicitly.

## Configuration & Data Tips
Keep dataset paths and experiment settings in `config.yaml` when practical. Do not commit raw private data or large transient outputs unless they are intended as final artifacts.
