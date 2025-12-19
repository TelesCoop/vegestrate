"""Shared utilities for dataset manifest operations."""

import json
from pathlib import Path


def load_manifest(manifest_path: str) -> dict[str, list[dict]]:
    """Load dataset manifest from JSON file.

    Args:
        manifest_path: Path to dataset_manifest.json

    Returns:
        Dictionary with 'train' and 'test' lists of entries
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest


def save_manifest(manifest: dict[str, list[dict]], output_path: str):
    """Save dataset manifest to JSON file.

    Args:
        manifest: Dictionary with 'train' and 'test' lists
        output_path: Path to save manifest JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def setup_split_directories(
    base_dir: Path, splits: list[str] = None
) -> dict[str, Path]:
    """Create train/test/val directories.

    Args:
        base_dir: Base data directory
        splits: List of split names (default: ['train', 'test'])

    Returns:
        Dictionary mapping split name to directory path
    """
    if splits is None:
        splits = ["train", "test"]

    split_dirs = {}
    for split in splits:
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_dir

    return split_dirs
