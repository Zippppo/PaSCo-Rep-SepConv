"""Dataset split script for LOOC project.

Generates a unified train/val/test split (8:1:1) that can be used
across different models and servers for fair comparison.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

RANDOM_SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "voxel-output", "merged_data")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "dataset_split.json")


def create_split(data_dir, output_path, train_ratio=0.8, val_ratio=0.1, seed=RANDOM_SEED):
    """Create and save dataset split."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    sample_ids = [os.path.basename(f).replace('.npz', '') for f in all_files]

    if not sample_ids:
        print(f"Error: No .npz files found in {data_dir}")
        sys.exit(1)

    np.random.seed(seed)
    indices = np.random.permutation(len(sample_ids))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_ids = [sample_ids[i] for i in indices[:n_train]]
    val_ids = [sample_ids[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [sample_ids[i] for i in indices[n_train + n_val:]]

    split_info = {
        "seed": seed,
        "total_samples": len(sample_ids),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 2),
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        },
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Dataset split saved to: {output_path}")
    print(f"  Total: {len(sample_ids)}")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/len(sample_ids):.1%})")
    print(f"  Val:   {len(val_ids)} ({len(val_ids)/len(sample_ids):.1%})")
    print(f"  Test:  {len(test_ids)} ({len(test_ids)/len(sample_ids):.1%})")

    return split_info


def load_split(split_path):
    """Load dataset split from JSON file."""
    with open(split_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset split for LOOC")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Path to data directory containing .npz files')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Output JSON file path')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    create_split(args.data_dir, args.output, args.train_ratio, args.val_ratio, args.seed)
