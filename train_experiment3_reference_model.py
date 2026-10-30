"""
Experiment 3 — ImageNet ResNet-18 Reference Model Training & Validation Gate

Trains an ImageNet-pretrained ResNet-18 reference model under the frozen Experiment 3 protocol:
1. Verifies Apple Silicon MPS compatibility and system resources.
2. Supports pre-flight smoke testing without test-set leakage.
3. Trains under locked seed (default 42) with inverse-frequency class weighting.
4. Selects best model strictly by validation balanced accuracy.
5. Saves best_model.pt with SHA-256 hash and logs complete training provenance.
6. Evaluates baseline performance on the unchanged test set before artifact interventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

CLASS_NAMES = ("non_sickled", "sickled")
LABEL_TO_ID = {"non_sickled": 0, "sickled": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    crop_path: Path
    label: str
    split: str
    source_key: str
    source_id: str
    pool: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_records(
    manifest_a: Path,
    manifest_b: Path,
    project_root: Path,
) -> list[CropRecord]:
    records: list[CropRecord] = []
    with manifest_a.open(newline="") as f:
        for r in csv.DictReader(f):
            records.append(
                CropRecord(
                    crop_id=r["crop_id"],
                    crop_path=project_root / r["crop_path"],
                    label="sickled",
                    split=r["split"],
                    source_key=r["source_key"],
                    source_id=r["source_id"],
                    pool="A",
                )
            )
    with manifest_b.open(newline="") as f:
        for r in csv.DictReader(f):
            records.append(
                CropRecord(
                    crop_id=r["crop_id"],
                    crop_path=project_root / r["crop_path"],
                    label="non_sickled",
                    split=r["split"],
                    source_key=r["source_key"],
                    source_id=r["source_id"],
                    pool="B",
                )
            )
    return records


def verify_split_integrity(records: list[CropRecord]) -> None:
    split_by_source: dict[str, set[str]] = defaultdict(set)
    label_by_source: dict[str, set[str]] = defaultdict(set)
    missing_files: list[str] = []

    for r in records:
        split_by_source[r["source_key"]].add(r["split"])
        label_by_source[r["source_key"]].add(r["label"])
        if not r.crop_path.exists():
            missing_files.append(str(r.crop_path))

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} crop files. First 3: {missing_files[:3]}")

    split_violations = {k: v for k, v in split_by_source.items() if len(v) > 1}
    label_violations = {k: v for k, v in label_by_source.items() if len(v) > 1}

    if split_violations:
        raise ValueError(f"CRITICAL: Split leakage across sources: {split_violations}")
    if label_violations:
        raise ValueError(f"CRITICAL: Conflicting labels across sources: {label_violations}")


class ReferenceCropDataset(Dataset):
    def __init__(self, records: list[CropRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
        r = self.records[index]
        with Image.open(r.crop_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        label_id = torch.tensor(LABEL_TO_ID[r.label], dtype=torch.long)
        return tensor, label_id, r.crop_id, r.source_key, str(r.crop_path)


def get_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_reference_resnet18() -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def compute_class_weights(records: list[CropRecord], device: torch.device) -> torch.Tensor:
    counts = Counter(r.label for r in records)
    weights = [1.0 / max(counts[lbl], 1) for lbl in CLASS_NAMES]
    norm_w = np.array(weights, dtype=np.float32)
    norm_w = norm_w / norm_w.sum() * len(CLASS_NAMES)
    return torch.tensor(norm_w, dtype=torch.float32, device=device)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    model.eval()
    all_targets = []
    all_scores = []  # P(sickled)
    all_preds = []
    all_ids = []
    all_sources = []
    all_paths = []

    with torch.no_grad():
        for tensors, targets, crop_ids, source_keys, paths in loader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Sickled prob

            all_targets.extend(targets.cpu().numpy().tolist())
            all_scores.extend(probs.cpu().numpy().tolist())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            all_ids.extend(crop_ids)
            all_sources.extend(source_keys)
            all_paths.extend(paths)

    return (
        np.array(all_targets, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_scores, dtype=float),
        all_ids,
        all_sources,
        all_paths,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sickled_recall_tpr": float(tpr),
        "non_sickled_spec_tnr": float(tnr),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_source_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sources: list[str],
    threshold: float = 0.50,
) -> dict[str, float]:
    src_groups: dict[str, list[float]] = defaultdict(list)
    src_labels: dict[str, int] = {}
    for yt, score, src in zip(y_true, y_score, sources):
        src_groups[src].append(score)
        src_labels[src] = yt

    src_true = []
    src_pred = []
    src_score = []
    for src, scores in src_groups.items():
        src_true.append(src_labels[src])
        mean_p = float(np.mean(scores))
        src_score.append(mean_p)
        src_pred.append(1 if mean_p >= threshold else 0)

    res = compute_metrics(np.array(src_true), np.array(src_pred), np.array(src_score))
    return res


def run_hardware_check(device: torch.device) -> dict[str, object]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": str(device),
    }
    print("--- Hardware & Environment Compatibility Check ---")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def run_training_experiment(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    smoke_test: bool = False,
    output_dir: Path = Path("experiment_3_reference_outputs"),
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = select_device("auto")
    hw_info = run_hardware_check(device)

    # 1. Load Dataset
    project_root = Path.cwd().resolve()
    manifest_a = Path("crop_outputs/pool_a_labeled_sickled_manifest.csv")
    manifest_b = Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv")

    records = load_dataset_records(manifest_a, manifest_b, project_root)
    verify_split_integrity(records)

    records_train = [r for r in records if r.split == "train"]
    records_val = [r for r in records if r.split == "val"]
    records_test = [r for r in records if r.split == "test"]

    print(f"\nDataset loaded: Train={len(records_train)}, Val={len(records_val)}, Test={len(records_test)} (Total: {len(records)})")

    train_tf, eval_tf = get_transforms(image_size=224)
    train_dataset = ReferenceCropDataset(records_train, train_tf)
    val_dataset = ReferenceCropDataset(records_val, eval_tf)
    test_dataset = ReferenceCropDataset(records_test, eval_tf)

    num_workers = 0 if device.type == "mps" else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 2. Build Reference ResNet-18
    model = build_reference_resnet18().to(device)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(records_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Smoke Test Mode (1 epoch, 2 batches, train & val only, zero test leakage)
    if smoke_test:
        print("\n--- Running Pre-Flight Smoke Test (1 epoch, 2 batches) ---")
        model.train()
        batch_count = 0
        total_loss = 0.0
        for tensors, targets, _, _, _ in train_loader:
            tensors, targets = tensors.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            if batch_count >= 2:
                break
        print(f"Smoke test forward/backward pass successful! Mean loss: {total_loss / batch_count:.4f}")
        return {"status": "smoke_test_passed", "device": str(device)}

    # 4. Full Reference Model Training
    print(f"\nStarting reference model training: {epochs} epochs, lr={lr}, AdamW on {device}...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_bal_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for tensors, targets, _, _, _ in train_loader:
            tensors, targets = tensors.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation Evaluation for Model Selection
        y_val_true, y_val_pred, y_val_score, _, _, _ = evaluate_loader(model, val_loader, device)
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_score)
        val_bal_acc = val_metrics["balanced_accuracy"]

        elapsed = time.time() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_balanced_accuracy": round(val_bal_acc, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_tpr": round(val_metrics["sickled_recall_tpr"], 4),
            "val_tnr": round(val_metrics["non_sickled_spec_tnr"], 4),
            "elapsed_seconds": round(elapsed, 2),
        })

        is_best = val_bal_acc > best_val_bal_acc
        if is_best:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "model_name": "Experiment 3 — ImageNet ResNet-18 Reference Model",
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_metrics": val_metrics,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "seed": seed,
                },
                "hardware": hw_info,
            }, output_dir / "best_model.pt")

        best_mark = " [*BEST*]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Bal Acc: {val_bal_acc:.1%} (TPR: {val_metrics['sickled_recall_tpr']:.1%}, TNR: {val_metrics['non_sickled_spec_tnr']:.1%}) | Time: {elapsed:.1f}s{best_mark}", flush=True)

    # 5. Freeze Model with Best Weights
    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    # Save History CSV
    history_csv = output_dir / "training_history.csv"
    with history_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Calculate Checkpoint SHA-256
    checkpoint_path = output_dir / "best_model.pt"
    ckpt_sha256 = compute_file_sha256(checkpoint_path)

    # 6. Tune Decision Threshold on Validation Split
    y_val_true, _, y_val_score, _, val_sources, _ = evaluate_loader(model, val_loader, device)
    best_th, best_th_val_ba = 0.50, 0.0
    for th in np.linspace(0.1, 0.9, 81):
        ba = balanced_accuracy_score(y_val_true, (y_val_score >= th).astype(int))
        if ba > best_th_val_ba:
            best_th_val_ba, best_th = ba, float(th)

    # 7. Baseline Evaluation on Test Set (One-Time Unchanged Benchmark)
    print("\n--- Evaluating Frozen Reference Model on Unchanged Test Set ---")
    y_test_true, y_test_pred_def, y_test_score, test_ids, test_sources, test_paths = evaluate_loader(model, test_loader, device)
    y_test_pred_tuned = (y_test_score >= best_th).astype(int)

    test_metrics_default = compute_metrics(y_test_true, y_test_pred_def, y_test_score)
    test_metrics_tuned = compute_metrics(y_test_true, y_test_pred_tuned, y_test_score)

    # Source-Level Aggregations
    test_src_default = compute_source_metrics(y_test_true, y_test_score, test_sources, threshold=0.50)
    test_src_tuned = compute_source_metrics(y_test_true, y_test_score, test_sources, threshold=best_th)

    # Save Test Predictions CSV
    test_pred_rows = []
    for cid, path, src, yt, p, s in zip(test_ids, test_paths, test_sources, y_test_true, y_test_pred_def, y_test_score):
        test_pred_rows.append({
            "crop_id": cid,
            "crop_path": str(Path(path).relative_to(project_root)),
            "source_key": src,
            "split": "test",
            "y_true": yt,
            "p_sickled": round(s, 6),
            "y_pred_def50": p,
            "y_pred_tuned": 1 if s >= best_th else 0,
        })

    test_csv = output_dir / "test_baseline_predictions.csv"
    with test_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_pred_rows)

    # Save Metrics JSON
    metrics_summary = {
        "model_name": "Experiment 3 — ImageNet ResNet-18 Reference Model",
        "checkpoint_sha256": ckpt_sha256,
        "best_epoch": best_epoch,
        "validation_metrics": {
            "best_val_bal_acc_default": float(best_val_bal_acc),
            "best_val_bal_acc_tuned": float(best_th_val_ba),
            "tuned_threshold": float(best_th),
        },
        "test_crop_metrics_default_th50": test_metrics_default,
        "test_crop_metrics_tuned_th": test_metrics_tuned,
        "test_source_metrics_default_th50": test_src_default,
        "test_source_metrics_tuned_th": test_src_tuned,
        "dataset_split_counts": {
            "train": len(records_train),
            "val": len(records_val),
            "test": len(records_test),
        },
        "hardware": hw_info,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

    # 8. Generate Comprehensive Markdown Report
    generate_reference_model_report(
        output_dir / "EXPERIMENT_3_REFERENCE_MODEL_REPORT.md",
        metrics_summary,
        history,
    )

    print(f"\nTraining and baseline evaluation complete!")
    print(f"  Best Validation Bal Acc: {best_val_bal_acc:.1%} (Epoch {best_epoch})")
    print(f"  Test Crop Bal Acc (th=0.50): {test_metrics_default['balanced_accuracy']:.1%} (TPR: {test_metrics_default['sickled_recall_tpr']:.1%}, TNR: {test_metrics_default['non_sickled_spec_tnr']:.1%})")
    print(f"  Test Source Bal Acc (th=0.50): {test_src_default['balanced_accuracy']:.1%}")
    print(f"  Checkpoint SHA-256: {ckpt_sha256}")
    return metrics_summary


def generate_reference_model_report(
    report_path: Path,
    summary: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    test_def = summary["test_crop_metrics_default_th50"]
    test_src = summary["test_source_metrics_default_th50"]
    val_m = summary["validation_metrics"]
    hw = summary["hardware"]

    # Baseline Quality Gate Check
    passed_gate = test_def["balanced_accuracy"] >= 0.75
    gate_status = "**PASSED (>= 75.0%)**" if passed_gate else "**FAILED (< 75.0%)**"

    md = [
        "# Experiment 3 — ImageNet ResNet-18 Reference Model Report",
        "",
        "## 1. Executive Summary & Provenance",
        "This report documents the training, validation selection, and baseline evaluation of the primary reference model for **Experiment 3: Controlled Intervention & Shortcut Reliance Analysis**.",
        "",
        "### Critical Lineage & Methodology Statement",
        "- **Model Identity**: `Experiment 3 — ImageNet ResNet-18 Reference Model`.",
        "- **Distinct from Historical Models**: This model is a newly trained reference architecture initialized from Torchvision ImageNet weights (`ResNet18_Weights.DEFAULT`). It is **not** the historical model, nor is it assumed to replicate historical 98–99% figures unless separately demonstrated.",
        "- **Protocol Status**: Frozen exploratory reference model prior to Stage 3 artifact interventions.",
        f"- **Checkpoint SHA-256**: `{summary['checkpoint_sha256']}`",
        f"- **Best Validation Epoch**: Epoch **{summary['best_epoch']}** (Val Balanced Accuracy: **{val_m['best_val_bal_acc_default']:.1%}**)",
        "",
        "### Baseline Quality Gate Assessment",
        f"- **Baseline Evaluation Gate Criterion**: Balanced accuracy $\\ge 75.0\\%$ on unchanged test crops.",
        f"- **Status**: {gate_status} — The reference model achieved **{test_def['balanced_accuracy']:.1%} balanced accuracy** on test crops (**{test_src['balanced_accuracy']:.1%} at source level**).",
        f"- **Decision**: {'Eligible to proceed to Stage 3 artifact interventions.' if passed_gate else 'Investigation of baseline classifier required before interpreting interventions.'}",
        "",
        "---",
        "",
        "## 2. Unchanged Baseline Test Performance",
        "",
        "| Evaluation Level | Decision Threshold ($th$) | Balanced Accuracy | Accuracy | Sickled Recall (TPR) | Non-Sickled Spec (TNR) | ROC AUC |",
        "|---|:---:|---:|---:|---:|---:|---:|",
        f"| **Crop Level (Standard)** | $th = 0.50$ | **{test_def['balanced_accuracy']:.1%}** | {test_def['accuracy']:.1%} | {test_def['sickled_recall_tpr']:.1%} | {test_def['non_sickled_spec_tnr']:.1%} | {test_def['roc_auc']:.3f} |",
        f"| **Crop Level (Val-Tuned)** | $th^* = {val_m['tuned_threshold']:.2f}$ | **{summary['test_crop_metrics_tuned_th']['balanced_accuracy']:.1%}** | {summary['test_crop_metrics_tuned_th']['accuracy']:.1%} | {summary['test_crop_metrics_tuned_th']['sickled_recall_tpr']:.1%} | {summary['test_crop_metrics_tuned_th']['non_sickled_spec_tnr']:.1%} | {summary['test_crop_metrics_tuned_th']['roc_auc']:.3f} |",
        f"| **Source Level (Standard)** | $th = 0.50$ | **{test_src['balanced_accuracy']:.1%}** | {test_src['accuracy']:.1%} | {test_src['sickled_recall_tpr']:.1%} | {test_src['non_sickled_spec_tnr']:.1%} | {test_src['roc_auc']:.3f} |",
        f"| **Source Level (Val-Tuned)** | $th^* = {val_m['tuned_threshold']:.2f}$ | **{summary['test_source_metrics_tuned_th']['balanced_accuracy']:.1%}** | {summary['test_source_metrics_tuned_th']['accuracy']:.1%} | {summary['test_source_metrics_tuned_th']['sickled_recall_tpr']:.1%} | {summary['test_source_metrics_tuned_th']['non_sickled_spec_tnr']:.1%} | {summary['test_source_metrics_tuned_th']['roc_auc']:.3f} |",
        "",
        "### Crop-Level Confusion Matrix (Standard $th = 0.50$)",
        f"- **True Positives (Sickled correctly identified)**: {test_def['tp']} / {test_def['tp'] + test_def['fn']} ({test_def['sickled_recall_tpr']:.1%})",
        f"- **True Negatives (Non-Sickled correctly identified)**: {test_def['tn']} / {test_def['tn'] + test_def['fp']} ({test_def['non_sickled_spec_tnr']:.1%})",
        f"- **False Positives (Non-Sickled predicted as Sickled)**: {test_def['fp']}",
        f"- **False Negatives (Sickled predicted as Non-Sickled)**: {test_def['fn']}",
        "",
        "---",
        "",
        "## 3. Training Dynamics & Convergence History",
        "",
        "| Epoch | Train Loss | Val Bal. Acc. | Val Accuracy | Val TPR | Val TNR | Time (s) | Best Checkpoint |",
        "|:---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for h in history:
        is_b = "Yes" if h["epoch"] == summary["best_epoch"] else ""
        md.append(f"| {h['epoch']:02d} | {h['train_loss']:.4f} | {h['val_balanced_accuracy']:.1%} | {h['val_accuracy']:.1%} | {h['val_tpr']:.1%} | {h['val_tnr']:.1%} | {h['elapsed_seconds']:.1f}s | {is_b} |")

    md.extend([
        "",
        "---",
        "",
        "## 4. Software & Hardware Environment",
        f"- **Operating Platform**: `{hw['platform']}` (`{hw['machine']}`)",
        f"- **Python Version**: `{hw['python_version']}`",
        f"- **PyTorch Version**: `{hw['torch_version']}`",
        f"- **Apple MPS Acceleration**: `{hw['mps_available']}` (Device: `{hw['selected_device']}`)",
        f"- **Training Seeds**: All random seeds locked to `42`.",
        "",
        "---",
        "",
        "## 5. Next Steps: Stage 3 Artifact Interventions",
        "With the baseline evaluation gate passed and the model weights frozen in `best_model.pt` (`" + summary["checkpoint_sha256"][:16] + "...`), Stage 3 will execute:",
        "1. **Subgroup Stratification**: Measure natural performance disparity across border-positive vs clean negative crops.",
        "2. **Border Removal ($T_{\\text{border}}$)** vs **Matched Negative Control ($T_{\\text{control}}$)**.",
        "3. **Plasma Equalization ($T_{\\text{plasma}}$)** vs **Matched Luminance Control ($T_{\\text{photo-control}}$)**.",
        "4. **Paired Shift Analysis**: Track continuous change in predicted probability $\\Delta p = p(\\text{sickled} \\mid x_{\\text{intervened}}) - p(\\text{sickled} \\mid x_{\\text{orig}})$ with 2,000 source-cluster bootstrap CIs.",
    ])

    report_path.write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Experiment 3 Reference ResNet-18 Model")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch / 2-batch smoke test")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_3_reference_outputs"))
    args = parser.parse_args()

    run_training_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
