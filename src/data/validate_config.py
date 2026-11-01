"""
ForenSightX - Dataset Configuration and Label Map Validator
Authoritative Reference: ForenSightX_Project_Architecture.docx Section 13

Validates `datasets.yaml` and `label_map.yaml` against schema requirements:
- Dataset name, local path, license, task modality, ID column, label column,
  class names, text column, file path column, min test cases.
- Path existence and non-empty status.
- Label map alignment with specified classes.

Usage:
  python -m src.data.validate_config [--datasets datasets.yaml] [--label-map label_map.yaml]
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import yaml


REQUIRED_DATASET_FIELDS = [
    "name",
    "path",
    "license",
    "task",
    "id_column",
    "label_column",
    "class_names",
    "min_test_cases",
]

ALLOWED_TASKS = {"image", "text", "multimodal"}


def validate_configs(
    datasets_path: str = "datasets.yaml",
    label_map_path: str = "label_map.yaml",
    check_paths_exist: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validates datasets and label mapping configurations.
    Returns:
      (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []

    # 1. File existence checks
    if not os.path.exists(datasets_path):
        errors.append(f"Missing required dataset configuration file: '{datasets_path}'")
    if not os.path.exists(label_map_path):
        errors.append(f"Missing required label mapping configuration file: '{label_map_path}'")

    if errors:
        return False, errors

    # 2. YAML parsing
    datasets_data: Optional[Dict[str, Any]] = None
    label_map_data: Optional[Dict[str, Any]] = None

    try:
        with open(datasets_path, "r", encoding="utf-8") as f:
            datasets_data = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"YAML parsing error in '{datasets_path}': {e}")

    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map_data = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"YAML parsing error in '{label_map_path}': {e}")

    if errors or datasets_data is None or label_map_data is None:
        return False, errors

    # 3. Datasets schema validation
    if not isinstance(datasets_data, dict) or "datasets" not in datasets_data:
        errors.append(f"Root object in '{datasets_path}' must be a dictionary containing 'datasets' list.")
        return False, errors

    datasets_list = datasets_data.get("datasets")
    if not isinstance(datasets_list, list) or len(datasets_list) == 0:
        errors.append(f"'datasets' list in '{datasets_path}' must contain at least one dataset entry.")
        return False, errors

    label_mapping = label_map_data.get("label_mapping", {})
    if not isinstance(label_mapping, dict):
        errors.append(f"'label_mapping' in '{label_map_path}' must be a dictionary.")

    for idx, ds in enumerate(datasets_list):
        if not isinstance(ds, dict):
            errors.append(f"Dataset entry #{idx + 1} must be a dictionary.")
            continue

        ds_name = ds.get("name", f"dataset_{idx + 1}")

        # Check required fields
        for field in REQUIRED_DATASET_FIELDS:
            if field not in ds or ds[field] is None:
                errors.append(f"Dataset '{ds_name}' is missing required field: '{field}'")

        # Modality checks
        task = ds.get("task")
        if task not in ALLOWED_TASKS:
            errors.append(
                f"Dataset '{ds_name}' has invalid task '{task}'. Allowed tasks: {sorted(list(ALLOWED_TASKS))}"
            )

        if task in ("text", "multimodal") and ("text_column" not in ds or not ds["text_column"]):
            errors.append(f"Dataset '{ds_name}' with task '{task}' is missing required 'text_column'")

        if task in ("image", "multimodal") and ("file_column" not in ds or not ds["file_column"]):
            errors.append(f"Dataset '{ds_name}' with task '{task}' is missing required 'file_column'")

        # Min test cases check
        min_cases = ds.get("min_test_cases")
        if isinstance(min_cases, int):
            if min_cases < 1:
                errors.append(f"Dataset '{ds_name}' min_test_cases must be >= 1 (got {min_cases})")
        else:
            errors.append(f"Dataset '{ds_name}' min_test_cases must be an integer")

        # Path check
        ds_path = ds.get("path")
        if check_paths_exist and ds_path:
            if not os.path.exists(ds_path):
                errors.append(
                    f"Dataset '{ds_name}' path does not exist on filesystem: '{ds_path}'"
                )

        # Label mapping alignment check
        class_names = ds.get("class_names")
        if isinstance(class_names, list) and task in label_mapping:
            task_map = label_mapping[task]
            for c_name in class_names:
                if isinstance(task_map, dict) and c_name not in task_map:
                    errors.append(
                        f"Dataset '{ds_name}' class '{c_name}' not defined in label_mapping['{task}']"
                    )

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ForenSightX dataset configuration files")
    parser.add_argument("--datasets", default="datasets.yaml", help="Path to datasets.yaml")
    parser.add_argument("--label-map", default="label_map.yaml", help="Path to label_map.yaml")
    parser.add_argument("--ignore-missing-paths", action="store_true", help="Do not assert local paths exist")
    args = parser.parse_args()

    print("=" * 80)
    print("ForenSightX Real Dataset Configuration Validator")
    print(f"Checking: datasets='{args.datasets}', label_map='{args.label_map}'")
    print("=" * 80)

    is_valid, errors = validate_configs(
        datasets_path=args.datasets,
        label_map_path=args.label_map,
        check_paths_exist=not args.ignore_missing_paths,
    )

    if is_valid:
        print("[SUCCESS] All dataset and label configuration requirements verified cleanly!")
        return 0
    else:
        print(f"[VALIDATION FAILED] Found {len(errors)} error(s):")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

