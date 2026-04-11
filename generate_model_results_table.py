from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "paper_tables"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def pretty_setup_name(setup: str) -> str:
    mapping = {
        "handcrafted_baseline": "Handcrafted Baseline",
        "supervised_scratch": "Supervised Scratch",
        "imagenet_finetune": "ImageNet Fine-tune",
        "simclr_finetune": "SimCLR Fine-tune",
    }
    return mapping.get(setup, setup)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
    )


def add_row(
    rows: list[dict[str, object]],
    *,
    stage: str,
    model: str,
    input_mode: str,
    setup: str,
    label_fraction: str,
    epochs: str,
    batch_size: str,
    learn_rate: str,
    best_val_bal_acc: float,
    test_acc: float,
    test_bal_acc: float,
    precision: float,
    recall: float,
    f1_score: float,
    roc_auc: float,
) -> None:
    rows.append(
        {
            "Stage": stage,
            "Model": model,
            "Input": input_mode,
            "Setup": pretty_setup_name(setup),
            "Label %": label_fraction,
            "Epochs": epochs,
            "Batch": batch_size,
            "LR": learn_rate,
            "Best Val Bal Acc": best_val_bal_acc,
            "Test Acc": test_acc,
            "Test Bal Acc": test_bal_acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1_score,
            "ROC AUC": roc_auc,
        }
    )


def append_stage4(rows: list[dict[str, object]]) -> None:
    metrics = read_json(ROOT / "stage4_outputs" / "metrics.json")
    add_row(
        rows,
        stage="Stage 4",
        model="Logistic Regression",
        input_mode="RGB",
        setup="handcrafted_baseline",
        label_fraction="100%",
        epochs="-",
        batch_size="-",
        learn_rate="-",
        best_val_bal_acc=metrics["validation_metrics"]["balanced_accuracy"],
        test_acc=metrics["test_metrics"]["accuracy"],
        test_bal_acc=metrics["test_metrics"]["balanced_accuracy"],
        precision=metrics["classification_report_test"]["macro avg"]["precision"],
        recall=metrics["classification_report_test"]["macro avg"]["recall"],
        f1_score=metrics["classification_report_test"]["macro avg"]["f1-score"],
        roc_auc=metrics["test_metrics"]["roc_auc"],
    )


def training_details(setup: str) -> tuple[str, str, str]:
    if setup == "simclr_finetune":
        return "50+20", "64+32", "3e-4 + 1e-4"
    return "20", "32", "1e-4"


def append_stage5(rows: list[dict[str, object]], output_root: Path, input_mode: str) -> None:
    for setup in ("supervised_scratch", "imagenet_finetune", "simclr_finetune"):
        metrics = read_json(output_root / setup / "metrics.json")
        epochs, batch_size, learn_rate = training_details(setup)
        add_row(
            rows,
            stage="Stage 5",
            model="ResNet-18",
            input_mode=input_mode,
            setup=setup,
            label_fraction="100%",
            epochs=epochs,
            batch_size=batch_size,
            learn_rate=learn_rate,
            best_val_bal_acc=metrics["best_val_balanced_accuracy"],
            test_acc=metrics["test"]["crop_metrics"]["accuracy"],
            test_bal_acc=metrics["test"]["crop_metrics"]["balanced_accuracy"],
            precision=metrics["test"]["crop_metrics"]["classification_report"]["macro avg"]["precision"],
            recall=metrics["test"]["crop_metrics"]["classification_report"]["macro avg"]["recall"],
            f1_score=metrics["test"]["crop_metrics"]["classification_report"]["macro avg"]["f1-score"],
            roc_auc=metrics["test"]["crop_metrics"]["roc_auc"],
        )


def append_stage6(rows: list[dict[str, object]]) -> None:
    for fraction in ("100", "050", "020", "010"):
        fraction_dir = ROOT / "stage6_grayscale_outputs" / f"fraction_{fraction}"
        for setup in ("supervised_scratch", "imagenet_finetune", "simclr_finetune"):
            metrics = read_json(fraction_dir / setup / "metrics.json")
            epochs, batch_size, learn_rate = training_details(setup)
            add_row(
                rows,
                stage="Stage 6",
                model="ResNet-18",
                input_mode="Grayscale",
                setup=setup,
                label_fraction=f"{int(fraction)}%",
                epochs=epochs,
                batch_size=batch_size,
                learn_rate=learn_rate,
                best_val_bal_acc=metrics["best_val_balanced_accuracy"],
                test_acc=metrics["test"]["crop_metrics"]["accuracy"],
                test_bal_acc=metrics["test"]["crop_metrics"]["balanced_accuracy"],
                precision=metrics["test"]["crop_metrics"]["classification_report"]["macro avg"]["precision"],
                recall=metrics["test"]["crop_metrics"]["classification_report"]["macro avg"]["recall"],
                f1_score=metrics["test"]["crop_metrics"]["classification_report"]["macro avg"]["f1-score"],
                roc_auc=metrics["test"]["crop_metrics"]["roc_auc"],
            )


def write_csv_table(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: list[dict[str, object]], path: Path) -> None:
    headers = list(rows[0].keys())
    lines = [
        "# All Model Results",
        "",
        "Macro precision, recall, and F1 are reported on the test crop split for consistency across all trained models.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex_table(rows: list[dict[str, object]], path: Path) -> None:
    headers = list(rows[0].keys())
    column_spec = "llllrlllrrrrrrr"
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Summary of all trained models across Stages 4--6. Precision, recall, and F1 are macro-averaged test crop metrics.}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [latex_escape(fmt(row[header])) for header in headers]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\label{tab:all-model-results}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    append_stage4(rows)
    append_stage5(rows, ROOT / "stage5_outputs", "RGB")
    append_stage5(rows, ROOT / "stage5_grayscale_outputs", "Grayscale")
    append_stage6(rows)

    write_csv_table(rows, OUTPUT_DIR / "all_model_results_table.csv")
    write_markdown_table(rows, OUTPUT_DIR / "all_model_results_table.md")
    write_latex_table(rows, OUTPUT_DIR / "all_model_results_table.tex")

    print(f"Wrote {OUTPUT_DIR / 'all_model_results_table.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'all_model_results_table.md'}")
    print(f"Wrote {OUTPUT_DIR / 'all_model_results_table.tex'}")


if __name__ == "__main__":
    main()
