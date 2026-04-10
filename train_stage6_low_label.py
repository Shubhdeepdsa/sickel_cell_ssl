from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import train_stage5 as stage5


def fraction_tag(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}"


def fraction_label(fraction: float) -> str:
    return f"{int(round(fraction * 100))}%"


def validate_fraction(fraction: float) -> None:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"Fractions must be in (0, 1]. Received: {fraction}")


def source_labels(records: list[stage5.CropRecord]) -> dict[str, str]:
    labels_by_source: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.label is not None:
            labels_by_source[record.source_key].add(record.label)
    resolved: dict[str, str] = {}
    for source_key, labels in labels_by_source.items():
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for source {source_key}: {sorted(labels)}")
        resolved[source_key] = next(iter(labels))
    return resolved


def build_nested_source_subsets(
    train_records: list[stage5.CropRecord],
    fractions: list[float],
    seed: int,
) -> dict[float, set[str]]:
    label_by_source = source_labels(train_records)
    shuffled_by_label: dict[str, list[str]] = {}
    for label in stage5.CLASS_NAMES:
        source_keys = sorted(source_key for source_key, source_label in label_by_source.items() if source_label == label)
        rng = stage5.np.random.default_rng(seed + (1000 if label == "sickled" else 0))
        shuffled = list(rng.permutation(source_keys))
        shuffled_by_label[label] = shuffled

    subsets: dict[float, set[str]] = {}
    for fraction in sorted(set(fractions)):
        selected: set[str] = set()
        for label in stage5.CLASS_NAMES:
            label_sources = shuffled_by_label[label]
            if not label_sources:
                continue
            n_select = len(label_sources) if fraction >= 1.0 else max(1, int(round(len(label_sources) * fraction)))
            n_select = min(n_select, len(label_sources))
            selected.update(label_sources[:n_select])
        subsets[fraction] = selected
    return subsets


def subset_records_by_sources(records: list[stage5.CropRecord], selected_sources: set[str]) -> list[stage5.CropRecord]:
    return [record for record in records if record.source_key in selected_sources]


def summarize_subset(train_records: list[stage5.CropRecord], selected_sources: set[str]) -> dict[str, object]:
    selected_records = subset_records_by_sources(train_records, selected_sources)
    source_counts = Counter(source_labels(selected_records).values())
    crop_counts = Counter(record.label for record in selected_records)
    return {
        "n_train_sources": len(selected_sources),
        "train_source_counts": dict(source_counts),
        "n_train_crops": len(selected_records),
        "train_crop_counts": dict(crop_counts),
    }


def write_subset_manifest(
    output_dir: Path,
    fraction: float,
    train_records: list[stage5.CropRecord],
    selected_sources: set[str],
) -> None:
    label_by_source = source_labels(train_records)
    crop_count_by_source = Counter(record.source_key for record in train_records if record.source_key in selected_sources)
    rows = [
        {
            "label_fraction": fraction,
            "label_fraction_percent": int(round(fraction * 100)),
            "source_key": source_key,
            "label": label_by_source[source_key],
            "n_train_crops": crop_count_by_source[source_key],
        }
        for source_key in sorted(selected_sources)
    ]
    stage5.write_csv(output_dir / f"fraction_{fraction_tag(fraction)}" / "train_subset_sources.csv", rows)


def write_stage6_summary(output_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    stage5.write_csv(output_dir / "stage6_low_label_comparison.csv", rows)
    report_lines = [
        "# Stage 6 Low-Label Protocol",
        "",
        "## Protocol",
        "- Training labels are reduced only within the train split.",
        "- Source-image subsets are sampled first, then all crops from the selected train sources are used.",
        "- Validation and test splits stay fixed across all low-label runs.",
        "- Source-level sampling is stratified by class and nested by fraction, so 10% is contained within 20%, 20% within 50%, and 50% within 100%.",
        f"- Fractions: `{', '.join(fraction_label(value) for value in args.fractions)}`.",
        f"- Inputs converted to grayscale before training/evaluation: `{args.grayscale}`.",
        f"- SimCLR SSL splits: `{','.join(args.ssl_splits)}`.",
        "- By default, SimCLR still sees the full train-split unlabeled pool. That is intentional: SSL should help when labeled data is scarce but unlabeled in-domain crops are still available.",
        "",
        "## Comparison",
    ]
    for row in rows:
        report_lines.extend(
            [
                f"### {row['setup']} at {row['label_fraction_display']}",
                f"- train sources: {row['n_train_sources']} ({row['train_source_counts']})",
                f"- train crops: {row['n_train_crops']} ({row['train_crop_counts']})",
                f"- best val balanced accuracy: {row['best_val_balanced_accuracy']}",
                f"- test crop balanced accuracy: {row['test_crop_balanced_accuracy']}",
                f"- test crop ROC AUC: {row['test_crop_roc_auc']}",
                f"- test source balanced accuracy: {row['test_source_balanced_accuracy']}",
            ]
        )
    (output_dir / "stage6_report.md").write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: low-label protocol for supervised, ImageNet, and SimCLR baselines.")
    parser.add_argument("--pool-a-manifest", type=Path, default=Path("crop_outputs/pool_a_labeled_sickled_manifest.csv"))
    parser.add_argument("--pool-b-manifest", type=Path, default=Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv"))
    parser.add_argument("--pool-c-manifest", type=Path, default=Path("crop_outputs/pool_c_unlabeled_candidates_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("stage6_outputs"))
    parser.add_argument(
        "--setup",
        choices=["all", "supervised_scratch", "imagenet_finetune", "simclr_finetune"],
        default="all",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--simclr-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--supervised-epochs", type=int, default=20)
    parser.add_argument("--simclr-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--simclr-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert every crop to grayscale before training and evaluation while keeping the 3-channel model interface unchanged.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.2, 0.1],
        help="Fractions of train-label source images to keep. Sampled at source-image level.",
    )
    parser.add_argument(
        "--simclr-checkpoint",
        type=Path,
        default=None,
        help="Reuse an existing SimCLR encoder checkpoint instead of rerunning SimCLR pretraining.",
    )
    parser.add_argument(
        "--ssl-splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Crops allowed for SimCLR pretraining. Default avoids validation/test leakage.",
    )
    args = parser.parse_args()

    fractions = [float(value) for value in args.fractions]
    for value in fractions:
        validate_fraction(value)
    args.fractions = fractions

    stage5.set_seed(args.seed)
    device = stage5.select_device(args.device)
    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_records = stage5.load_labeled_records(args.pool_a_manifest, args.pool_b_manifest, project_root)
    stage5.validate_records(labeled_records)
    stage5.validate_source_split_integrity(labeled_records)
    records_by_split = {
        split: [record for record in labeled_records if record.split == split]
        for split in ("train", "val", "test")
    }

    selected_by_fraction = build_nested_source_subsets(records_by_split["train"], args.fractions, args.seed)
    requested = (
        ["supervised_scratch", "imagenet_finetune", "simclr_finetune"]
        if args.setup == "all"
        else [args.setup]
    )

    simclr_checkpoint: Path | None = None
    if "simclr_finetune" in requested:
        simclr_dir = output_dir / "simclr_pretrain"
        if args.simclr_checkpoint is not None:
            simclr_checkpoint = args.simclr_checkpoint.resolve()
            if not simclr_checkpoint.exists():
                raise FileNotFoundError(f"SimCLR checkpoint does not exist: {simclr_checkpoint}")
        else:
            ssl_records = stage5.load_unlabeled_records(args.pool_c_manifest, labeled_records, project_root, set(args.ssl_splits))
            stage5.validate_records(ssl_records)
            simclr_checkpoint = stage5.train_simclr(ssl_records, args, device, simclr_dir)

    comparison_rows: list[dict[str, object]] = []
    for fraction in sorted(selected_by_fraction.keys(), reverse=True):
        selected_sources = selected_by_fraction[fraction]
        write_subset_manifest(output_dir, fraction, records_by_split["train"], selected_sources)
        subset_summary = summarize_subset(records_by_split["train"], selected_sources)
        low_label_records = {
            "train": subset_records_by_sources(records_by_split["train"], selected_sources),
            "val": records_by_split["val"],
            "test": records_by_split["test"],
        }

        for setup_name in requested:
            setup_output_dir = output_dir / f"fraction_{fraction_tag(fraction)}" / setup_name
            if setup_name == "supervised_scratch":
                summary = stage5.train_supervised_classifier(
                    f"{setup_name}_{fraction_tag(fraction)}",
                    pretrained=False,
                    records_by_split=low_label_records,
                    args=args,
                    device=device,
                    output_dir=setup_output_dir,
                )
            elif setup_name == "imagenet_finetune":
                summary = stage5.train_supervised_classifier(
                    f"{setup_name}_{fraction_tag(fraction)}",
                    pretrained=True,
                    records_by_split=low_label_records,
                    args=args,
                    device=device,
                    output_dir=setup_output_dir,
                )
            else:
                if simclr_checkpoint is None:
                    raise RuntimeError("SimCLR checkpoint was not prepared")
                summary = stage5.train_supervised_classifier(
                    f"{setup_name}_{fraction_tag(fraction)}",
                    pretrained=False,
                    records_by_split=low_label_records,
                    args=args,
                    device=device,
                    output_dir=setup_output_dir,
                    simclr_checkpoint=simclr_checkpoint,
                )

            summary["label_fraction"] = fraction
            summary["label_fraction_display"] = fraction_label(fraction)
            summary.update(subset_summary)
            (setup_output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
            comparison_rows.append(
                {
                    "setup": setup_name,
                    "input_mode": "grayscale" if args.grayscale else "rgb",
                    "label_fraction": fraction,
                    "label_fraction_display": fraction_label(fraction),
                    **subset_summary,
                    "best_val_balanced_accuracy": summary.get("best_val_balanced_accuracy"),
                    "test_crop_balanced_accuracy": summary.get("test", {}).get("crop_metrics", {}).get("balanced_accuracy"),
                    "test_crop_accuracy": summary.get("test", {}).get("crop_metrics", {}).get("accuracy"),
                    "test_crop_roc_auc": summary.get("test", {}).get("crop_metrics", {}).get("roc_auc"),
                    "test_source_balanced_accuracy": summary.get("test", {}).get("source_metrics", {}).get("balanced_accuracy"),
                    "test_source_accuracy": summary.get("test", {}).get("source_metrics", {}).get("accuracy"),
                }
            )

    write_stage6_summary(output_dir, comparison_rows, args)
    print(f"Wrote {output_dir / 'stage6_report.md'}")


if __name__ == "__main__":
    main()
