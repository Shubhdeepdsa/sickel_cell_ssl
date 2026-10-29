from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KAGGLE_DATASET_ID = "florencetushabe/sickle-cell-disease-dataset"
EXPECTED_COUNTS = {
    "Positive/Labelled": 422,
    "Positive/Unlabelled": 422,
    "Negative/Clear": 147,
}


def verify_dataset(dataset_dir: Path) -> bool:
    all_ok = True
    total_found = 0
    print(f"Checking dataset directory at: {dataset_dir}")
    for subpath, expected_count in EXPECTED_COUNTS.items():
        folder = dataset_dir / subpath
        if not folder.exists():
            print(f"  [MISSING] {subpath} does not exist.")
            all_ok = False
            continue
        found = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        total_found += found
        status = "OK" if found == expected_count else "COUNT MISMATCH"
        print(f"  [{status}] {subpath}: found {found} images (expected {expected_count})")
        if found != expected_count:
            all_ok = False
    print(f"Total image count: {total_found} (expected {sum(EXPECTED_COUNTS.values())})")
    return all_ok


def download_and_setup(dataset_dir: Path, force: bool = False) -> None:
    if dataset_dir.exists() and not force:
        if verify_dataset(dataset_dir):
            print("Dataset is already present and complete. Use --force to re-download.")
            return

    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required to download the dataset. "
            "Please install it using: pip install kagglehub or uv pip install kagglehub"
        )

    print(f"Downloading '{KAGGLE_DATASET_ID}' from Kaggle via kagglehub...")
    downloaded_path_str = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    downloaded_path = Path(downloaded_path_str)
    print(f"Dataset downloaded to cache: {downloaded_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in ["Positive", "Negative"]:
        src = downloaded_path / folder_name
        dest = dataset_dir / folder_name
        if src.exists():
            print(f"Syncing {src} -> {dest}...")
            if dest.exists() and force:
                shutil.rmtree(dest)
            shutil.copytree(src, dest, dirs_exist_ok=True)

    print("\nVerifying copied dataset...")
    if verify_dataset(dataset_dir):
        print("\nDataset successfully installed and verified in `./datasets`!")
    else:
        print("\nWARNING: Some files might be missing or counts did not match expected numbers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and set up the Uganda Sickle Cell Disease Dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets"),
        help="Target directory for dataset (default: ./datasets)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-copy even if target directory exists",
    )
    args = parser.parse_args()
    download_and_setup(args.dataset_dir, force=args.force)


if __name__ == "__main__":
    main()
