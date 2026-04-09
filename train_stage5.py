from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm.auto import tqdm


CLASS_NAMES = ("non_sickled", "sickled")
LABEL_TO_ID = {"non_sickled": 0, "sickled": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CropRecord:
    crop_path: Path
    label: str | None
    split: str
    source_key: str
    pool: str


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_labeled_records(pool_a_manifest: Path, pool_b_manifest: Path, project_root: Path) -> list[CropRecord]:
    records: list[CropRecord] = []
    for row in read_csv(pool_a_manifest):
        records.append(
            CropRecord(
                crop_path=project_root / row["crop_path"],
                label="sickled",
                split=row["split"],
                source_key=row["source_key"],
                pool="A",
            )
        )
    for row in read_csv(pool_b_manifest):
        records.append(
            CropRecord(
                crop_path=project_root / row["crop_path"],
                label="non_sickled",
                split=row["split"],
                source_key=row["source_key"],
                pool="B",
            )
        )
    return records


def load_unlabeled_records(
    pool_c_manifest: Path,
    labeled_records: list[CropRecord],
    project_root: Path,
    splits: set[str],
) -> list[CropRecord]:
    records: list[CropRecord] = []
    for row in read_csv(pool_c_manifest):
        if row["split"] in splits:
            records.append(
                CropRecord(
                    crop_path=project_root / row["crop_path"],
                    label=None,
                    split=row["split"],
                    source_key=row["source_key"],
                    pool="C",
                )
            )
    for record in labeled_records:
        if record.split in splits:
            records.append(
                CropRecord(
                    crop_path=record.crop_path,
                    label=None,
                    split=record.split,
                    source_key=record.source_key,
                    pool=record.pool,
                )
            )
    return records


def validate_records(records: list[CropRecord]) -> None:
    missing = [str(record.crop_path) for record in records if not record.crop_path.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} crop images. First missing paths: {preview}")


def validate_source_split_integrity(records: list[CropRecord]) -> dict[str, object]:
    split_by_source: dict[str, set[str]] = defaultdict(set)
    label_by_source: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_by_source[record.source_key].add(record.split)
        if record.label is not None:
            label_by_source[record.source_key].add(record.label)
    multi_split = {source: sorted(splits) for source, splits in split_by_source.items() if len(splits) > 1}
    multi_label = {source: sorted(labels) for source, labels in label_by_source.items() if len(labels) > 1}
    if multi_split:
        raise ValueError(f"Source leakage detected across splits: {multi_split}")
    if multi_label:
        raise ValueError(f"Source has conflicting labels: {multi_label}")
    return {
        "n_sources": len(split_by_source),
        "sources_with_multiple_splits": 0,
        "sources_with_multiple_labels": 0,
    }


class LabeledCropDataset(Dataset):
    def __init__(self, records: list[CropRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        record = self.records[index]
        if record.label is None:
            raise ValueError("LabeledCropDataset received an unlabeled record")
        with Image.open(record.crop_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(LABEL_TO_ID[record.label], dtype=torch.long), record.source_key, str(record.crop_path)


class SimCLRDataset(Dataset):
    def __init__(self, records: list[CropRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.crop_path) as image:
            image = image.convert("RGB")
            view_1 = self.transform(image)
            view_2 = self.transform(image)
        return view_1, view_2


def train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def simclr_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
            transforms.RandomGrayscale(p=0.10),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_resnet18_classifier(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


class SimCLRModel(nn.Module):
    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)


def json_safe_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


def load_torch_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError:
        # This script only falls back for checkpoints it created locally.
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_simclr_encoder_into_classifier(classifier: nn.Module, simclr_checkpoint: Path) -> None:
    checkpoint = load_torch_checkpoint(simclr_checkpoint)
    encoder_state = checkpoint["encoder_state_dict"]
    classifier_state = classifier.state_dict()
    transferred = {
        key: value
        for key, value in encoder_state.items()
        if key in classifier_state and not key.startswith("fc.")
    }
    classifier_state.update(transferred)
    classifier.load_state_dict(classifier_state)


def class_weights(records: list[CropRecord], device: torch.device) -> torch.Tensor:
    counts = Counter(record.label for record in records)
    weights = [1.0 / max(counts[label], 1) for label in CLASS_NAMES]
    normalized = np.asarray(weights, dtype=np.float32)
    normalized = normalized / normalized.sum() * len(CLASS_NAMES)
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_one_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_items = 0
    for images, targets, _, _ in tqdm(loader, desc="supervised train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        n_items += images.size(0)
    return total_loss / max(n_items, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_source_keys: list[str] = []
    all_paths: list[str] = []
    for images, targets, source_keys, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())
        all_source_keys.extend(source_keys)
        all_paths.extend(paths)
    return (
        np.concatenate(all_targets),
        np.concatenate(all_probs),
        all_source_keys,
        all_paths,
    )


def metric_dict(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    y_pred = (y_score >= 0.5).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_class": {
            ID_TO_LABEL[class_id]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_id in enumerate([0, 1])
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) == 2 else None
    return metrics


def source_level_metrics(y_true: np.ndarray, y_score: np.ndarray, source_keys: list[str]) -> dict[str, object]:
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for target, score, source_key in zip(y_true, y_score, source_keys):
        by_source[source_key].append((int(target), float(score)))
    source_targets: list[int] = []
    source_scores: list[float] = []
    for values in by_source.values():
        target_values = {target for target, _ in values}
        if len(target_values) != 1:
            raise ValueError(f"Conflicting labels in one source during source-level metrics: {target_values}")
        source_targets.append(target_values.pop())
        source_scores.append(float(np.mean([score for _, score in values])))
    return metric_dict(np.asarray(source_targets, dtype=np.int64), np.asarray(source_scores, dtype=np.float64))


def prediction_rows(
    split: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    source_keys: list[str],
    paths: list[str],
) -> list[dict[str, object]]:
    y_pred = (y_score >= 0.5).astype(np.int64)
    return [
        {
            "split": split,
            "source_key": source_key,
            "crop_path": path,
            "actual_label": ID_TO_LABEL[int(target)],
            "predicted_label": ID_TO_LABEL[int(predicted)],
            "probability_sickled": round(float(score), 6),
        }
        for target, predicted, score, source_key, path in zip(y_true, y_pred, y_score, source_keys, paths)
    ]


def train_supervised_classifier(
    setup_name: str,
    pretrained: bool,
    records_by_split: dict[str, list[CropRecord]],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    simclr_checkpoint: Path | None = None,
) -> dict[str, object]:
    train_dataset = LabeledCropDataset(records_by_split["train"], train_transform(args.image_size))
    val_dataset = LabeledCropDataset(records_by_split["val"], eval_transform(args.image_size))
    test_dataset = LabeledCropDataset(records_by_split["test"], eval_transform(args.image_size))
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_resnet18_classifier(pretrained=pretrained)
    if simclr_checkpoint is not None:
        load_simclr_encoder_into_classifier(model, simclr_checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(records_by_split["train"], device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.supervised_epochs, 1))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_balanced_accuracy = -1.0
    history: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.supervised_epochs + 1):
        start = time.time()
        train_loss = train_one_supervised_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        y_val, val_score, _, _ = predict(model, val_loader, device)
        val_metrics = metric_dict(y_val, val_score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "elapsed_seconds": round(time.time() - start, 3),
            }
        )
        if float(val_metrics["balanced_accuracy"]) > best_val_balanced_accuracy:
            best_val_balanced_accuracy = float(val_metrics["balanced_accuracy"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "setup": setup_name,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "val_metrics": val_metrics,
                    "args": json_safe_value(vars(args)),
                },
                output_dir / "best_model.pt",
            )
        print(
            f"{setup_name} epoch {epoch}/{args.supervised_epochs} "
            f"loss={train_loss:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    split_outputs = {}
    for split, loader in {"val": val_loader, "test": test_loader}.items():
        y_true, y_score, source_keys, paths = predict(model, loader, device)
        split_outputs[split] = {
            "crop_metrics": metric_dict(y_true, y_score),
            "source_metrics": source_level_metrics(y_true, y_score, source_keys),
        }
        write_csv(output_dir / f"{split}_predictions.csv", prediction_rows(split, y_true, y_score, source_keys, paths))

    write_csv(output_dir / "history.csv", history)
    summary = {
        "setup": setup_name,
        "pretrained": pretrained,
        "simclr_checkpoint": str(simclr_checkpoint) if simclr_checkpoint else None,
        "dataset_counts": {
            split: dict(Counter(record.label for record in records))
            for split, records in records_by_split.items()
        },
        "best_val_balanced_accuracy": best_val_balanced_accuracy,
        **split_outputs,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(mask, -9e15)
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
    return F.cross_entropy(similarity, positives)


def train_simclr(
    unlabeled_records: list[CropRecord],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Path:
    dataset = SimCLRDataset(unlabeled_records, simclr_transform(args.image_size))
    loader = make_loader(dataset, args.simclr_batch_size, shuffle=True, num_workers=args.num_workers)
    model = SimCLRModel(projection_dim=args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.simclr_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.simclr_epochs, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []

    for epoch in range(1, args.simclr_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start = time.time()
        for view_1, view_2 in tqdm(loader, desc="simclr pretrain", leave=False):
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(view_1)
            z2 = model(view_2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "simclr_loss": mean_loss, "elapsed_seconds": round(time.time() - start, 3)})
        print(f"simclr epoch {epoch}/{args.simclr_epochs} loss={mean_loss:.4f}", flush=True)

    checkpoint_path = output_dir / "simclr_encoder.pt"
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "projection_dim": args.projection_dim,
            "temperature": args.temperature,
            "n_pretrain_crops": len(unlabeled_records),
            "args": json_safe_value(vars(args)),
        },
        checkpoint_path,
    )
    write_csv(output_dir / "simclr_history.csv", history)
    return checkpoint_path


def load_existing_summary(output_dir: Path, setup_name: str) -> dict[str, object] | None:
    metrics_path = output_dir / setup_name / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def write_stage5_summary(output_dir: Path, summaries: dict[str, dict[str, object]], args: argparse.Namespace) -> None:
    rows = []
    for setup, summary in summaries.items():
        test_crop = summary.get("test", {}).get("crop_metrics", {})
        test_source = summary.get("test", {}).get("source_metrics", {})
        rows.append(
            {
                "setup": setup,
                "test_crop_balanced_accuracy": test_crop.get("balanced_accuracy"),
                "test_crop_accuracy": test_crop.get("accuracy"),
                "test_crop_roc_auc": test_crop.get("roc_auc"),
                "test_source_balanced_accuracy": test_source.get("balanced_accuracy"),
                "test_source_accuracy": test_source.get("accuracy"),
                "best_val_balanced_accuracy": summary.get("best_val_balanced_accuracy"),
            }
        )
    write_csv(output_dir / "stage5_baseline_comparison.csv", rows)
    report_lines = [
        "# Stage 5 Pretraining And Baselines",
        "",
        "## Setups",
        "- `supervised_scratch`: ResNet-18 trained from random initialization on Pool A vs Pool B.",
        "- `imagenet_finetune`: ResNet-18 initialized from ImageNet weights, then fine-tuned on Pool A vs Pool B.",
        "- `simclr_finetune`: ResNet-18 SimCLR-pretrained on in-domain crops, then fine-tuned on Pool A vs Pool B.",
        "",
        "## Leakage Policy",
        f"- SimCLR SSL splits: `{','.join(args.ssl_splits)}`.",
        "- Default is train-only SSL pretraining. Use `--ssl-splits train val test` only if you intentionally want transductive pretraining.",
        "",
        "## Comparison",
    ]
    for row in rows:
        report_lines.append(
            "- {setup}: test crop balanced_accuracy={test_crop_balanced_accuracy}, "
            "test source balanced_accuracy={test_source_balanced_accuracy}, test ROC AUC={test_crop_roc_auc}".format(**row)
        )
    (output_dir / "stage5_report.md").write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: supervised, ImageNet, and SimCLR baselines.")
    parser.add_argument("--pool-a-manifest", type=Path, default=Path("crop_outputs/pool_a_labeled_sickled_manifest.csv"))
    parser.add_argument("--pool-b-manifest", type=Path, default=Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv"))
    parser.add_argument("--pool-c-manifest", type=Path, default=Path("crop_outputs/pool_c_unlabeled_candidates_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("stage5_outputs"))
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

    set_seed(args.seed)
    device = select_device(args.device)
    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_records = load_labeled_records(args.pool_a_manifest, args.pool_b_manifest, project_root)
    validate_records(labeled_records)
    validate_source_split_integrity(labeled_records)
    records_by_split = {
        split: [record for record in labeled_records if record.split == split]
        for split in ("train", "val", "test")
    }

    requested = (
        ["supervised_scratch", "imagenet_finetune", "simclr_finetune"]
        if args.setup == "all"
        else [args.setup]
    )
    summaries: dict[str, dict[str, object]] = {}

    if "supervised_scratch" in requested:
        summaries["supervised_scratch"] = train_supervised_classifier(
            "supervised_scratch",
            pretrained=False,
            records_by_split=records_by_split,
            args=args,
            device=device,
            output_dir=output_dir / "supervised_scratch",
        )

    if "imagenet_finetune" in requested:
        summaries["imagenet_finetune"] = train_supervised_classifier(
            "imagenet_finetune",
            pretrained=True,
            records_by_split=records_by_split,
            args=args,
            device=device,
            output_dir=output_dir / "imagenet_finetune",
        )

    if "simclr_finetune" in requested:
        simclr_dir = output_dir / "simclr_pretrain"
        if args.simclr_checkpoint is not None:
            simclr_checkpoint = args.simclr_checkpoint.resolve()
            if not simclr_checkpoint.exists():
                raise FileNotFoundError(f"SimCLR checkpoint does not exist: {simclr_checkpoint}")
        else:
            ssl_records = load_unlabeled_records(args.pool_c_manifest, labeled_records, project_root, set(args.ssl_splits))
            validate_records(ssl_records)
            simclr_checkpoint = train_simclr(ssl_records, args, device, simclr_dir)
        summaries["simclr_finetune"] = train_supervised_classifier(
            "simclr_finetune",
            pretrained=False,
            records_by_split=records_by_split,
            args=args,
            device=device,
            output_dir=output_dir / "simclr_finetune",
            simclr_checkpoint=simclr_checkpoint,
        )

    for setup_name in ("supervised_scratch", "imagenet_finetune", "simclr_finetune"):
        if setup_name not in summaries:
            existing_summary = load_existing_summary(output_dir, setup_name)
            if existing_summary is not None:
                summaries[setup_name] = existing_summary

    write_stage5_summary(output_dir, summaries, args)
    print(f"Wrote {output_dir / 'stage5_report.md'}")


if __name__ == "__main__":
    main()
