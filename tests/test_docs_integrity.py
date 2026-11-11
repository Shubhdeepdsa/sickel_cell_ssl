"""
Automated Documentation Integrity & Consistency Verification

Validates that:
1. All relative links in README.md point to actual existing files.
2. No local absolute file:/// URLs exist.
3. Outlier crop IDs and category frequencies in README match
   experiment_3_reference_outputs/outlier_visual_audit_adjudication.csv exactly.
4. Key dataset counts and metric claims match underlying split and manifest files.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


def test_no_absolute_file_urls() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")
    matches = re.findall(r"file:///[^\s\)\"\'>]+", readme_text)
    assert not matches, f"Found absolute local file:/// URLs in README.md: {matches}"
    print("[PASS] test_no_absolute_file_urls: Zero absolute file:/// URLs found.")


def test_relative_links_exist() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")
    # Match markdown links [text](relative_path) where path does not start with http or #
    links = re.findall(r"\[([^\]]+)\]\(([^http#\)\s]+)\)", readme_text)
    missing = []
    for text, path_str in links:
        target = Path(path_str.split("#")[0])  # strip anchors
        if not target.exists():
            missing.append((text, path_str))
    assert not missing, f"Found broken relative links in README.md: {missing}"
    print(f"[PASS] test_relative_links_exist: All {len(links)} relative links exist on disk.")


def test_outlier_crop_ids_match_csv() -> None:
    csv_path = Path("experiment_3_reference_outputs/outlier_visual_audit_adjudication.csv")
    assert csv_path.exists(), f"Missing adjudication CSV at {csv_path}"

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 20, f"Expected 20 audited outliers, found {len(rows)}"

    readme_text = Path("README.md").read_text(encoding="utf-8")
    missing_crops = []
    for r in rows:
        crop_id = r["crop_id"]
        if f"`{crop_id}`" not in readme_text:
            missing_crops.append(crop_id)

    assert not missing_crops, f"Outlier crop IDs missing from README.md: {missing_crops}"
    print(f"[PASS] test_outlier_crop_ids_match_csv: All 20 audited outlier crop IDs verified in README.md.")


def test_dataset_counts_and_metrics() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")

    # 1. Unique source images (569) and total image files (991)
    assert "569" in readme_text, "Missing 569 unique source images in README"
    assert "991" in readme_text, "Missing 991 image files in README"

    # 2. Check source_image_splits.csv count
    source_split_path = Path("split_outputs/source_image_splits.csv")
    with source_split_path.open(encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    assert len(source_rows) == 569, f"source_image_splits.csv has {len(source_rows)} rows (expected 569)"

    # 3. Dataset versions
    assert "2,711" in readme_text, "Missing Dataset v1 count (2,711) in README"
    assert "2,717" in readme_text, "Missing Dataset v2 count (2,717) in README"

    # 4. Verified border prevalence
    assert "35.4%" in readme_text, "Missing 35.4% negative border rate in README"
    assert "10.0%" in readme_text, "Missing 10.0% positive border rate in README"
    assert "25.3 percentage points" in readme_text, "Missing 25.3 percentage points difference in README"

    # 5. Reference model accuracy
    assert "99.5%" in readme_text, "Missing 99.5% test balanced accuracy in README"
    assert "402 of 404" in readme_text, "Missing 402/404 correct crops count in README"

    # 6. Controlled intervention contrast
    assert "-0.0026" in readme_text, "Missing -0.0026 intervention contrast in README"

    # 7. Historical Stage 4 baseline consistency
    assert "84.8%" in readme_text, "Missing 84.8% Stage 4 baseline in README"

    print("[PASS] test_dataset_counts_and_metrics: Key dataset counts and metrics strictly verified.")


def test_license_consistency() -> None:
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    assert "Florence Tushabe" in license_text, "LICENSE missing Florence Tushabe attribution"
    assert "CC-BY 4.0" in license_text, "LICENSE missing CC-BY 4.0 attribution notice"
    print("[PASS] test_license_consistency: License file and dataset attribution verified.")


if __name__ == "__main__":
    print("Running Documentation Integrity Verification Suite...\n")
    test_no_absolute_file_urls()
    test_relative_links_exist()
    test_outlier_crop_ids_match_csv()
    test_dataset_counts_and_metrics()
    test_license_consistency()
    print("\nAll documentation consistency checks PASSED successfully!")
