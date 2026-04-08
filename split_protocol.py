from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from eda import IMAGE_EXTENSIONS, iter_files, load_dataset_config


TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ("train", "val", "test")
KNOWN_STAINS = {"field", "leishman"}
STAIN_METADATA_PLACEHOLDER = Path("path/to/stains.csv")


@dataclass(frozen=True)
class SourceImage:
    source_id: str
    label: str
    stain: str
    stain_source: str
    raw_path: Path
    annotation_path: Path | None = None


def infer_stain_from_path(path: Path) -> tuple[str, str]:
    tokens = []
    for part in path.parts:
        tokens.extend(part.lower().replace("-", "_").split("_"))
    for token in tokens:
        if token in KNOWN_STAINS:
            return token, "path"
    return "unknown", "not_available"


def load_stain_metadata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path == STAIN_METADATA_PLACEHOLDER:
        print(
            "Warning: `path/to/stains.csv` is an example placeholder, not a real metadata file. "
            "Proceeding with stain=unknown."
        )
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"Stain metadata file does not exist: {path}. "
            "Remove `--stain-metadata` to run without Field/Leishman labels, or pass a real CSV "
            "with columns `source_key,stain`."
        )

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Stain metadata CSV has no header: {path}")

        source_column = next(
            (key for key in ("source_key", "source_id", "stem", "file_name", "filename") if key in reader.fieldnames),
            None,
        )
        if source_column is None or "stain" not in reader.fieldnames:
            raise ValueError(
                "Stain metadata CSV must include a `stain` column and one of "
                "`source_key`, `source_id`, `stem`, `file_name`, or `filename`."
            )

        metadata: dict[str, str] = {}
        for row in reader:
            source_id = row[source_column].strip() if source_column == "source_key" else Path(row[source_column]).stem
            stain = row["stain"].strip().lower()
            if stain:
                metadata[source_id] = stain
    return metadata


def build_sources(config_path: Path, stain_metadata_path: Path | None) -> tuple[list[SourceImage], Path]:
    config = load_dataset_config(config_path)
    stain_metadata = load_stain_metadata(stain_metadata_path)
    sources: list[SourceImage] = []

    for raw_path in iter_files(config.unlabelled, IMAGE_EXTENSIONS):
        source_id = raw_path.stem
        source_key = f"positive:{source_id}"
        annotation_path = config.labelled / raw_path.name
        if source_key in stain_metadata:
            stain, stain_source = stain_metadata[source_key], "metadata"
        elif source_id in stain_metadata:
            stain, stain_source = stain_metadata[source_id], "metadata"
        else:
            stain, stain_source = infer_stain_from_path(raw_path.relative_to(config.root))
        sources.append(
            SourceImage(
                source_id=source_id,
                label="positive",
                stain=stain,
                stain_source=stain_source,
                raw_path=raw_path,
                annotation_path=annotation_path if annotation_path.exists() else None,
            )
        )

    for raw_path in iter_files(config.negative_clear, IMAGE_EXTENSIONS):
        source_id = raw_path.stem
        source_key = f"negative:{source_id}"
        if source_key in stain_metadata:
            stain, stain_source = stain_metadata[source_key], "metadata"
        elif source_id in stain_metadata:
            stain, stain_source = stain_metadata[source_id], "metadata"
        else:
            stain, stain_source = infer_stain_from_path(raw_path.relative_to(config.root))
        sources.append(
            SourceImage(
                source_id=source_id,
                label="negative",
                stain=stain,
                stain_source=stain_source,
                raw_path=raw_path,
            )
        )

    return sources, config.root.parent


def target_counts(n_items: int) -> dict[str, int]:
    raw_counts = {split: n_items * TARGET_RATIOS[split] for split in SPLIT_ORDER}
    counts = {split: int(raw_counts[split]) for split in SPLIT_ORDER}
    remaining = n_items - sum(counts.values())

    remainders = sorted(
        SPLIT_ORDER,
        key=lambda split: (raw_counts[split] - counts[split], TARGET_RATIOS[split]),
        reverse=True,
    )
    for split in remainders[:remaining]:
        counts[split] += 1

    non_train_splits = [split for split in ("val", "test") if TARGET_RATIOS[split] > 0]
    if n_items >= len(non_train_splits) + 1:
        for split in non_train_splits:
            if counts[split] == 0:
                donor = max(SPLIT_ORDER, key=lambda candidate: counts[candidate])
                counts[donor] -= 1
                counts[split] += 1
    return counts


def split_sources(sources: list[SourceImage], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[SourceImage]] = defaultdict(list)
    for source in sources:
        strata[(source.label, source.stain)].append(source)

    assignments: dict[str, str] = {}
    for stratum_sources in strata.values():
        ordered = sorted(stratum_sources, key=lambda source: source.source_id)
        rng.shuffle(ordered)
        counts = target_counts(len(ordered))
        cursor = 0
        for split in SPLIT_ORDER:
            for source in ordered[cursor : cursor + counts[split]]:
                assignments[f"{source.label}:{source.source_id}"] = split
            cursor += counts[split]
    return assignments


def source_rows(sources: list[SourceImage], assignments: dict[str, str], project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in sorted(sources, key=lambda item: (item.label, item.source_id)):
        key = f"{source.label}:{source.source_id}"
        rows.append(
            {
                "split": assignments[key],
                "source_key": key,
                "source_id": source.source_id,
                "label": source.label,
                "stain": source.stain,
                "stain_source": source.stain_source,
                "raw_path": str(source.raw_path.relative_to(project_root)),
                "annotation_path": str(source.annotation_path.relative_to(project_root)) if source.annotation_path else "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_stain_template(path: Path, rows: list[dict[str, str]]) -> None:
    template_rows = [
        {
            "source_key": row["source_key"],
            "source_id": row["source_id"],
            "label": row["label"],
            "stain": "",
            "raw_path": row["raw_path"],
        }
        for row in rows
    ]
    write_csv(path, template_rows)


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    label_counts = Counter((row["split"], row["label"]) for row in rows)
    stain_counts = Counter((row["split"], row["label"], row["stain"]) for row in rows)

    label_rows = [
        {"split": split, "label": label, "count": str(count)}
        for (split, label), count in sorted(label_counts.items())
    ]
    stain_rows = [
        {"split": split, "label": label, "stain": stain, "count": str(count)}
        for (split, label, stain), count in sorted(stain_counts.items())
    ]
    return label_rows, stain_rows


def write_report(path: Path, rows: list[dict[str, str]], label_rows: list[dict[str, str]], stain_rows: list[dict[str, str]]) -> None:
    total = len(rows)
    split_counts = Counter(row["split"] for row in rows)
    stain_sources = Counter(row["stain_source"] for row in rows)
    unknown_stain = sum(1 for row in rows if row["stain"] == "unknown")

    def lines(table: list[dict[str, str]], keys: tuple[str, ...]) -> str:
        if not table:
            return "- none"
        return "\n".join("- " + ", ".join(f"{key}={row[key]}" for key in keys) for row in table)

    report = f"""# Stage 2 Split Protocol

## Protocol
- Split unit: source image, before crop generation.
- Target split: 70% train, 15% val, 15% test.
- Stratification key: label plus stain.
- Seed: recorded in `source_image_splits.csv`.

## Split counts
- train: {split_counts.get("train", 0)} / {total}
- val: {split_counts.get("val", 0)} / {total}
- test: {split_counts.get("test", 0)} / {total}

## Label balance
{lines(label_rows, ("split", "label", "count"))}

## Stain balance
{lines(stain_rows, ("split", "label", "stain", "count"))}

## Stain metadata note
- stain_source counts: {dict(stain_sources)}
- unknown stain rows: {unknown_stain}

If Field vs Leishman labels are available later, provide a CSV with columns `source_key,stain` or `source_id,stain` and rerun this script with `--stain-metadata path/to/file.csv`. This script will then stratify by those labels instead of `unknown`.

## Generated files
- `source_image_splits.csv`: one row per source image. Crops must inherit this split.
- `split_label_summary.csv`: split counts by positive/negative.
- `split_stain_summary.csv`: split counts by label and stain.
- `crop_annotation_splits.csv`: annotation-box rows joined to the source split when `eda_outputs/annotation_boxes.csv` exists.
- `stain_metadata_template.csv`: optional template to fill if Field/Leishman labels become available.
"""
    path.write_text(report)


def write_crop_annotation_manifest(
    annotation_boxes_path: Path,
    source_split_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    if not annotation_boxes_path.exists():
        return

    positive_split_by_source = {
        row["source_key"]: row["split"] for row in source_split_rows if row["label"] == "positive"
    }
    positive_stain_by_source = {
        row["source_key"]: row["stain"] for row in source_split_rows if row["label"] == "positive"
    }

    rows: list[dict[str, str]] = []
    with annotation_boxes_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = Path(row["file_name"]).stem
            source_key = f"positive:{source_id}"
            split = positive_split_by_source.get(source_key)
            if split is None:
                continue
            rows.append(
                {
                    "split": split,
                    "source_key": source_key,
                    "source_id": source_id,
                    "label": "positive",
                    "stain": positive_stain_by_source.get(source_key, "unknown"),
                    **row,
                }
            )
    write_csv(output_path, rows)


def run_split(
    config_path: Path,
    output_dir: Path,
    seed: int,
    stain_metadata_path: Path | None,
    annotation_boxes_path: Path,
) -> None:
    sources, project_root = build_sources(config_path, stain_metadata_path)
    assignments = split_sources(sources, seed)
    rows = source_rows(sources, assignments, project_root)
    for row in rows:
        row["seed"] = str(seed)

    label_rows, stain_rows = summarize(rows)
    write_csv(output_dir / "source_image_splits.csv", rows)
    write_csv(output_dir / "split_label_summary.csv", label_rows)
    write_csv(output_dir / "split_stain_summary.csv", stain_rows)
    write_stain_template(output_dir / "stain_metadata_template.csv", rows)
    write_crop_annotation_manifest(annotation_boxes_path, rows, output_dir / "crop_annotation_splits.csv")
    write_report(output_dir / "stage2_split_protocol.md", rows, label_rows, stain_rows)

    print(f"Wrote {output_dir / 'source_image_splits.csv'}")
    print(f"Wrote {output_dir / 'stage2_split_protocol.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create source-image-level train/val/test splits before crop generation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("split_outputs"), help="Directory for split outputs")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument(
        "--stain-metadata",
        type=Path,
        default=None,
        help=(
            "Optional real CSV with columns source_key,stain for Field/Leishman balancing. "
            "Omit this option if the stain labels are not available."
        ),
    )
    parser.add_argument(
        "--annotation-boxes",
        type=Path,
        default=Path("eda_outputs/annotation_boxes.csv"),
        help="Optional annotation box CSV to join onto source-level splits",
    )
    args = parser.parse_args()
    run_split(args.config, args.output_dir, args.seed, args.stain_metadata, args.annotation_boxes)


if __name__ == "__main__":
    main()
