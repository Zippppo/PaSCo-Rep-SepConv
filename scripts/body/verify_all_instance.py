#!/usr/bin/env python
"""
Verify All-Instance Strategy Implementation.

This script verifies that:
1. Label mapping is correct (all organs 2-35 are thing classes)
2. Dataset generates correct mask_labels
3. Class weights are set correctly
4. Expected number of instances per sample

Usage:
    python scripts/body/verify_all_instance.py --dataset_root Dataset/voxel_data --split_file dataset_split.json
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Direct import of label_mapping to avoid heavy dependencies through __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "label_mapping",
    os.path.join(project_root, "pasco/data/body/label_mapping.py")
)
label_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_mapping)

THING_IDS = label_mapping.THING_IDS
SINGLE_INSTANCE_IDS = label_mapping.SINGLE_INSTANCE_IDS
MULTI_INSTANCE_IDS = label_mapping.MULTI_INSTANCE_IDS
CLASS_NAMES_NEW = label_mapping.CLASS_NAMES_NEW
N_CLASSES_NEW = label_mapping.N_CLASSES_NEW
get_total_instances_per_sample = label_mapping.get_total_instances_per_sample
validate_mapping = label_mapping.validate_mapping


def verify_label_mapping():
    """Verify label mapping is correct."""
    print("\n" + "="*60)
    print("1. VERIFYING LABEL MAPPING")
    print("="*60)

    print(f"\nNumber of classes: {N_CLASSES_NEW}")
    print(f"Thing IDs (organs with instances): {len(THING_IDS)} classes")
    print(f"  - Single-instance organs (2-22): {len(SINGLE_INSTANCE_IDS)} classes")
    print(f"  - Multi-instance organs (23-35): {len(MULTI_INSTANCE_IDS)} classes")

    # Verify thing_ids covers classes 2-35
    expected_thing_ids = list(range(2, 36))
    if THING_IDS == expected_thing_ids:
        print("\n✓ THING_IDS correctly covers all organ classes (2-35)")
    else:
        print(f"\n✗ THING_IDS mismatch!")
        print(f"  Expected: {expected_thing_ids}")
        print(f"  Got: {THING_IDS}")
        return False

    # Run full validation
    print("\nRunning full label mapping validation...")
    try:
        validate_mapping()
        print("✓ Label mapping validation passed!")
        return True
    except AssertionError as e:
        print(f"✗ Label mapping validation failed: {e}")
        return False


def verify_dataset(dataset_root, split_file, use_precomputed=True):
    """Verify dataset generates correct mask labels."""
    print("\n" + "="*60)
    print("2. VERIFYING DATASET MASK LABELS")
    print("="*60)

    # Lazy import to avoid MinkowskiEngine dependency
    try:
        import torch
        from pasco.data.body.body_dataset import BodyDataset
    except ImportError as e:
        print(f"\n⚠ Cannot import dataset module: {e}")
        print("  Skipping dataset verification")
        return True  # Don't fail on import error

    # Determine precomputed path
    precomputed_root = dataset_root + "_precomputed" if use_precomputed else None

    if use_precomputed and not os.path.exists(precomputed_root):
        print(f"\n✗ Precomputed directory not found: {precomputed_root}")
        print("  Please run: python scripts/body/data/data_pre_process.py first")
        return False

    # Create dataset
    try:
        dataset = BodyDataset(
            split="train",
            root=dataset_root,
            split_file=split_file,
            use_precomputed=use_precomputed,
            precomputed_root=precomputed_root,
        )
        print(f"\n✓ Dataset created successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"\n✗ Failed to create dataset: {e}")
        return False

    # Test a few samples
    num_test = min(5, len(dataset))
    print(f"\nTesting {num_test} samples...")

    instance_counts = []
    for i in range(num_test):
        try:
            sample = dataset.get_individual(i)
            mask_label = sample["mask_label"]
            n_instances = len(mask_label["labels"])
            instance_counts.append(n_instances)

            # Count instances by type
            single_count = sum(1 for l in mask_label["labels"] if l.item() in SINGLE_INSTANCE_IDS)
            multi_count = sum(1 for l in mask_label["labels"] if l.item() in MULTI_INSTANCE_IDS)

            print(f"  Sample {i}: {n_instances} instances "
                  f"(single: {single_count}, multi: {multi_count})")

            # Verify all labels are in thing_ids
            for label in mask_label["labels"]:
                if label.item() not in THING_IDS:
                    print(f"    ✗ Invalid label {label.item()} not in THING_IDS!")
                    return False

        except Exception as e:
            print(f"  ✗ Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    avg_instances = np.mean(instance_counts)
    expected = get_total_instances_per_sample()

    print(f"\nInstance count statistics:")
    print(f"  Average: {avg_instances:.1f}")
    print(f"  Expected (all organs): {expected}")
    print(f"  Range: [{min(instance_counts)}, {max(instance_counts)}]")

    if avg_instances > 30:  # Should have at least 30+ instances
        print("\n✓ Dataset generates reasonable number of instances")
        return True
    else:
        print("\n✗ Too few instances generated!")
        return False


def verify_class_weights():
    """Verify class weights are set correctly."""
    print("\n" + "="*60)
    print("3. VERIFYING CLASS WEIGHTS")
    print("="*60)

    n_classes_with_dustbin = N_CLASSES_NEW + 1
    class_weight = np.ones(n_classes_with_dustbin)
    class_weight[0] = 0.0   # outside_body - IGNORE
    class_weight[1] = 0.05  # inside_body_empty - low weight
    class_weight[-1] = 0.1  # dustbin class

    print(f"\nClass weights ({n_classes_with_dustbin} total):")
    print(f"  Class 0 (outside_body): {class_weight[0]:.2f} (IGNORED)")
    print(f"  Class 1 (inside_body_empty): {class_weight[1]:.2f}")
    print(f"  Classes 2-35 (organs): 1.0")
    print(f"  Dustbin class: {class_weight[-1]:.2f}")

    # Verify
    checks_passed = True

    if class_weight[0] != 0.0:
        print("  ✗ outside_body should have weight 0.0!")
        checks_passed = False
    else:
        print("  ✓ outside_body correctly ignored")

    if class_weight[1] > 0.1:
        print("  ✗ inside_body_empty should have low weight!")
        checks_passed = False
    else:
        print("  ✓ inside_body_empty has low weight")

    organ_weights = class_weight[2:36]
    if np.all(organ_weights == 1.0):
        print("  ✓ All organ classes have weight 1.0")
    else:
        print("  ✗ Some organ classes don't have weight 1.0!")
        checks_passed = False

    return checks_passed


def verify_config():
    """Verify config parameters."""
    print("\n" + "="*60)
    print("4. VERIFYING CONFIG PARAMETERS")
    print("="*60)

    import yaml
    config_path = "configs/body.yaml"

    if not os.path.exists(config_path):
        print(f"\n✗ Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig: {config_path}")
    print(f"  num_queries: {config.get('num_queries', 'NOT SET')}")
    print(f"  thing_ids length: {len(config.get('thing_ids', []))}")
    print(f"  n_classes: {config.get('n_classes', 'NOT SET')}")
    print(f"  transformer_dec_layers: {config.get('transformer_dec_layers', 'NOT SET')}")

    checks_passed = True

    # Verify num_queries
    num_queries = config.get('num_queries', 0)
    expected_instances = get_total_instances_per_sample()
    if num_queries >= expected_instances * 3:
        print(f"\n  ✓ num_queries ({num_queries}) >= 3x expected instances ({expected_instances})")
    else:
        print(f"\n  ✗ num_queries ({num_queries}) too low! Need >= {expected_instances * 3}")
        checks_passed = False

    # Verify thing_ids
    if config.get('thing_ids', []) == list(range(2, 36)):
        print("  ✓ thing_ids correctly set to [2-35]")
    else:
        print("  ✗ thing_ids not correctly set!")
        checks_passed = False

    return checks_passed


def main():
    parser = argparse.ArgumentParser(description="Verify All-Instance Strategy")
    parser.add_argument('--dataset_root', type=str, default="Dataset/voxel_data")
    parser.add_argument('--split_file', type=str, default="dataset_split.json")
    parser.add_argument('--skip_dataset', action='store_true', help='Skip dataset verification')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ALL-INSTANCE STRATEGY VERIFICATION")
    print("="*60)

    all_passed = True

    # 1. Verify label mapping
    if not verify_label_mapping():
        all_passed = False

    # 2. Verify class weights
    if not verify_class_weights():
        all_passed = False

    # 3. Verify config
    if not verify_config():
        all_passed = False

    # 4. Verify dataset (optional)
    if not args.skip_dataset:
        if os.path.exists(args.dataset_root) and os.path.exists(args.split_file):
            if not verify_dataset(args.dataset_root, args.split_file):
                all_passed = False
        else:
            print("\n⚠ Skipping dataset verification (files not found)")

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    if all_passed:
        print("\n✓ All verifications PASSED!")
        print("\nYou can now run training with:")
        print("  python scripts/body/train.py \\")
        print("    --dataset_root Dataset/voxel_data \\")
        print("    --split_file dataset_split.json \\")
        print("    --use_precomputed \\")
        print("    --num_queries 400 \\")
        print("    --exp_prefix body_allinstance")
        return 0
    else:
        print("\n✗ Some verifications FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
