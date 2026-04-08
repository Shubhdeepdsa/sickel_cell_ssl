from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("non_sickled", "sickled")
LABEL_TO_ID = {"non_sickled": 0, "sickled": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows(pool_a_manifest: Path, pool_b_manifest: Path, project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(pool_a_manifest):
        row = row.copy()
        row["binary_label"] = "sickled"
        row["label_id"] = "1"
        rows.append(row)
    for row in read_csv(pool_b_manifest):
        row = row.copy()
        row["binary_label"] = "non_sickled"
        row["label_id"] = "0"
        rows.append(row)

    missing = [row["crop_path"] for row in rows if not (project_root / row["crop_path"]).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} crop files. First missing paths: {preview}")
    return rows


def validate_source_split_integrity(rows: list[dict[str, str]]) -> dict[str, object]:
    split_by_source: dict[str, set[str]] = defaultdict(set)
    label_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_by_source[row["source_key"]].add(row["split"])
        label_by_source[row["source_key"]].add(row["binary_label"])

    sources_with_multiple_splits = {
        source_key: sorted(splits) for source_key, splits in split_by_source.items() if len(splits) > 1
    }
    sources_with_multiple_labels = {
        source_key: sorted(labels) for source_key, labels in label_by_source.items() if len(labels) > 1
    }
    if sources_with_multiple_splits:
        raise ValueError(f"Source leakage detected across splits: {sources_with_multiple_splits}")
    if sources_with_multiple_labels:
        raise ValueError(f"Source has conflicting labels: {sources_with_multiple_labels}")

    return {
        "n_sources": len(split_by_source),
        "sources_with_multiple_splits": 0,
        "sources_with_multiple_labels": 0,
    }


def rgb_histogram(image_array: np.ndarray, bins: int) -> np.ndarray:
    histograms = [
        np.histogram(image_array[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)[0]
        for channel in range(3)
    ]
    return np.concatenate(histograms).astype(np.float32)


def image_features(path: Path, image_size: int, hist_bins: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    gray = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    gradient_x = np.abs(np.diff(gray, axis=1))
    gradient_y = np.abs(np.diff(gray, axis=0))
    texture = np.array(
        [
            float(gray.mean()),
            float(gray.std()),
            float(gradient_x.mean()),
            float(gradient_y.mean()),
            float((gray < 0.18).mean()),
            float((gray > 0.92).mean()),
        ],
        dtype=np.float32,
    )

    thumbnail = array.reshape(-1).astype(np.float32)
    histogram = rgb_histogram(array, hist_bins)
    return np.concatenate([thumbnail, histogram, texture])


def build_matrix(rows: list[dict[str, str]], project_root: Path, image_size: int, hist_bins: int) -> tuple[np.ndarray, np.ndarray]:
    features = [image_features(project_root / row["crop_path"], image_size, hist_bins) for row in rows]
    labels = [int(row["label_id"]) for row in rows]
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def make_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "per_class": {
            ID_TO_LABEL[class_id]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_id in enumerate([0, 1])
        },
    }
    if len(set(y.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


def majority_baseline_metrics(y: np.ndarray) -> dict[str, object]:
    majority_label = Counter(y.tolist()).most_common(1)[0][0]
    predictions = np.full_like(y, fill_value=majority_label)
    return {
        "majority_label": ID_TO_LABEL[int(majority_label)],
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
    }


def source_level_metrics(rows: list[dict[str, str]], predictions: np.ndarray) -> dict[str, object]:
    by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_source[row["source_key"]].append((row["binary_label"], int(prediction)))

    actual_labels: list[int] = []
    predicted_labels: list[int] = []
    crop_counts: list[int] = []
    for source_key, values in by_source.items():
        source_actuals = {actual for actual, _ in values}
        if len(source_actuals) != 1:
            raise ValueError(f"Conflicting labels within source {source_key}: {source_actuals}")
        actual_label = LABEL_TO_ID[source_actuals.pop()]
        predicted_label = Counter(prediction for _, prediction in values).most_common(1)[0][0]
        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)
        crop_counts.append(len(values))

    actual = np.asarray(actual_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    return {
        "n_sources": len(by_source),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "source_label_counts": dict(Counter(ID_TO_LABEL[int(value)] for value in actual)),
        "prediction_counts": dict(Counter(ID_TO_LABEL[int(value)] for value in predicted)),
        "mean_crops_per_source": float(np.mean(crop_counts)) if crop_counts else 0.0,
        "max_crops_per_source": int(max(crop_counts)) if crop_counts else 0,
    }


def prediction_rows(model: Pipeline, rows: list[dict[str, str]], x: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    probabilities = model.predict_proba(x)[:, 1]
    predictions = model.predict(x)
    output_rows: list[dict[str, object]] = []
    for row, actual_id, predicted_id, probability in zip(rows, y, predictions, probabilities):
        output_rows.append(
            {
                "split": row["split"],
                "source_key": row["source_key"],
                "crop_path": row["crop_path"],
                "actual_label": ID_TO_LABEL[int(actual_id)],
                "predicted_label": ID_TO_LABEL[int(predicted_id)],
                "probability_sickled": round(float(probability), 6),
            }
        )
    return output_rows


def confusion_matrix_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    matrix = metrics["confusion_matrix"]
    return [
        {
            "actual_label": actual_label,
            "predicted_non_sickled": matrix[actual_id][0],
            "predicted_sickled": matrix[actual_id][1],
        }
        for actual_id, actual_label in enumerate(CLASS_NAMES)
    ]


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    val = summary["validation_metrics"]
    test = summary["test_metrics"]
    counts = summary["dataset_counts"]
    majority = summary["majority_baselines"]
    source_metrics = summary["source_level_metrics"]

    def fmt(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    report = f"""# Stage 4 Binary Cell Classifier

## Task
- Binary crop classification only: sickled vs non-sickled.
- Not cell counting.
- Not whole-image diagnosis.
- Training data: Pool A labeled sickled crops plus Pool B labeled non-sickled crops.
- Split protocol: crop rows inherit source-image splits from Stage 2.

## Dataset
- train: {counts["train"]}
- val: {counts["val"]}
- test: {counts["test"]}
- source leakage check: {summary["source_integrity"]}

## Majority Baseline
- val majority label: {majority["val"]["majority_label"]}; accuracy: {fmt(majority["val"]["accuracy"])}; balanced_accuracy: {fmt(majority["val"]["balanced_accuracy"])}
- test majority label: {majority["test"]["majority_label"]}; accuracy: {fmt(majority["test"]["accuracy"])}; balanced_accuracy: {fmt(majority["test"]["balanced_accuracy"])}

## Model
- Model type: logistic regression baseline on resized RGB thumbnail, RGB histograms, and simple texture features.
- Selected C from validation: {summary["selected_c"]}
- Candidate C values: {summary["candidate_c_values"]}

## Validation Metrics
- accuracy: {fmt(val["accuracy"])}
- balanced_accuracy: {fmt(val["balanced_accuracy"])}
- macro_f1: {fmt(val["f1_macro"])}
- roc_auc: {fmt(val["roc_auc"])}

## Test Metrics
- accuracy: {fmt(test["accuracy"])}
- balanced_accuracy: {fmt(test["balanced_accuracy"])}
- macro_f1: {fmt(test["f1_macro"])}
- roc_auc: {fmt(test["roc_auc"])}

## Source-Level Majority Vote
- val source-level balanced_accuracy: {fmt(source_metrics["val"]["balanced_accuracy"])} over {source_metrics["val"]["n_sources"]} source images
- test source-level balanced_accuracy: {fmt(source_metrics["test"]["balanced_accuracy"])} over {source_metrics["test"]["n_sources"]} source images
- test crop prediction counts: {summary["prediction_counts"]["test"]}

## Caveats
- Pool B was generated by connected components from negative images and should be manually spot-checked before treating this as a final medical model.
- This is a first downstream baseline, not an SSL model and not a whole-slide or whole-image diagnosis pipeline.
"""
    path.write_text(report)


def run_training(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    rows = load_manifest_rows(args.pool_a_manifest, args.pool_b_manifest, project_root)
    source_integrity = validate_source_split_integrity(rows)

    rows_by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    for split, split_rows in rows_by_split.items():
        if not split_rows:
            raise ValueError(f"No rows for split={split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "binary_dataset_manifest.csv", rows)

    print("Extracting image features", flush=True)
    matrices = {
        split: build_matrix(split_rows, project_root, args.image_size, args.hist_bins)
        for split, split_rows in rows_by_split.items()
    }

    x_train, y_train = matrices["train"]
    x_val, y_val = matrices["val"]
    x_test, y_test = matrices["test"]

    print("Selecting regularization on validation split", flush=True)
    validation_grid: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_c: float | None = None
    best_score = -1.0
    for c_value in args.c_values:
        model = make_model(c_value, args.seed)
        model.fit(x_train, y_train)
        val_metrics = evaluate_model(model, x_val, y_val)
        validation_grid.append({"C": c_value, **val_metrics})
        score = float(val_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_c = c_value
            best_model = model

    if best_model is None or best_c is None:
        raise RuntimeError("No model was trained")

    val_metrics = evaluate_model(best_model, x_val, y_val)
    test_metrics = evaluate_model(best_model, x_test, y_test)
    val_predictions = best_model.predict(x_val)
    test_predictions = best_model.predict(x_test)

    print("Writing Stage 4 outputs", flush=True)
    joblib.dump(best_model, args.output_dir / "binary_cell_classifier.joblib")
    write_csv(args.output_dir / "validation_predictions.csv", prediction_rows(best_model, rows_by_split["val"], x_val, y_val))
    write_csv(args.output_dir / "test_predictions.csv", prediction_rows(best_model, rows_by_split["test"], x_test, y_test))
    write_csv(args.output_dir / "validation_confusion_matrix.csv", confusion_matrix_rows(val_metrics))
    write_csv(args.output_dir / "test_confusion_matrix.csv", confusion_matrix_rows(test_metrics))
    write_csv(
        args.output_dir / "validation_model_selection.csv",
        [
            {
                "C": row["C"],
                "balanced_accuracy": row["balanced_accuracy"],
                "accuracy": row["accuracy"],
                "f1_macro": row["f1_macro"],
                "roc_auc": row["roc_auc"],
            }
            for row in validation_grid
        ],
    )

    summary = {
        "task": "binary_cell_classification",
        "classes": list(CLASS_NAMES),
        "selected_c": best_c,
        "candidate_c_values": list(args.c_values),
        "image_size": args.image_size,
        "hist_bins": args.hist_bins,
        "dataset_counts": {
            split: dict(Counter(row["binary_label"] for row in split_rows))
            for split, split_rows in rows_by_split.items()
        },
        "source_integrity": source_integrity,
        "majority_baselines": {
            "train": majority_baseline_metrics(y_train),
            "val": majority_baseline_metrics(y_val),
            "test": majority_baseline_metrics(y_test),
        },
        "prediction_counts": {
            "val": dict(Counter(ID_TO_LABEL[int(value)] for value in val_predictions)),
            "test": dict(Counter(ID_TO_LABEL[int(value)] for value in test_predictions)),
        },
        "source_level_metrics": {
            "val": source_level_metrics(rows_by_split["val"], val_predictions),
            "test": source_level_metrics(rows_by_split["test"], test_predictions),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classification_report_val": classification_report(
            y_val,
            best_model.predict(x_val),
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_test": classification_report(
            y_test,
            best_model.predict(x_test),
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_markdown_report(args.output_dir / "stage4_binary_classifier.md", summary)

    print(f"Selected C: {best_c}")
    print(f"Validation balanced accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Wrote {args.output_dir / 'stage4_binary_classifier.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 4 binary sickled/non-sickled crop classifier.")
    parser.add_argument(
        "--pool-a-manifest",
        type=Path,
        default=Path("crop_outputs/pool_a_labeled_sickled_manifest.csv"),
        help="Pool A labeled sickled manifest from Stage 3",
    )
    parser.add_argument(
        "--pool-b-manifest",
        type=Path,
        default=Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv"),
        help="Pool B labeled non-sickled manifest from Stage 3",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("stage4_outputs"), help="Directory for model outputs")
    parser.add_argument("--image-size", type=int, default=32, help="Thumbnail size used for classifier features")
    parser.add_argument("--hist-bins", type=int, default=16, help="Per-channel RGB histogram bins")
    parser.add_argument("--seed", type=int, default=42, help="Model random seed")
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_C_VALUES),
        help="Logistic regression C values to evaluate on validation split",
    )
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
