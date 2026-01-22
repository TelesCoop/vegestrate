#!/usr/bin/env python3
"""Prepare training and testing data from LiDAR tiles.

This script reads data/dataset_manifest.json and processes tiles:
- Downloads LiDAR data (if not exists)
- Generates classification_map.tif
- Fetches corresponding orthophoto via WMS

Supports parallel processing using threads for faster execution.

Usage:
    # First generate manifest
    python update_manifest.py

    # Then prepare data
    python prepare_training_data.py --manifest data/dataset_manifest.json

    # Use more threads for faster processing
    python prepare_training_data.py --workers 8
"""

import argparse
import time
from functools import partial
from pathlib import Path

import laspy

from ..core import (
    build_tile_list,
    create_classification_map,
    download_file,
    filter_ground_vegetation,
    load_manifest,
    print_classification_info,
    print_processing_summary,
    process_tiles_parallel,
    setup_split_directories,
)
from .fetch_wms_from_raster import fetch_wms_for_raster

WMS_URL = "https://data.geopf.fr/wms-r"
WMS_LAYER = "HR.ORTHOIMAGERY.ORTHOPHOTOS"


def extract_tile_id(url):
    """Extract tile ID from URL filename"""
    filename = url.split("/")[-1]
    parts = filename.split("_")
    tile_id = f"{parts[2]}_{parts[3]}"
    return tile_id


def download_and_process_lidar(url, output_dir, resolution=0.2):
    """
    Download LiDAR data and create classification map

    Args:
        url: LiDAR download URL
        output_dir: Directory to save outputs
        resolution: Raster cell size in meters (default: 0.2)

    Returns:
        Path to classification_map.tif
    """
    filename = url.split("/")[-1]
    tile_id = extract_tile_id(url)
    laz_path = output_dir / filename

    print(f"\n{'=' * 70}")
    print(f"Processing tile: {tile_id}")
    print(f"{'=' * 70}")

    if not laz_path.exists():
        print(f"Downloading LiDAR data from {url}...")
        download_file(url, str(laz_path))
        print(f"✓ Downloaded to {laz_path}")
    else:
        print(f"✓ LiDAR data already exists: {laz_path}")

    print("Loading LAS data...")
    las = laspy.read(str(laz_path))
    print(f"✓ Loaded {len(las.points):,} points")

    print_classification_info(las)

    filtered_las = filter_ground_vegetation(las)

    classmap_path = output_dir / f"classification_map_{tile_id}.tif"
    create_classification_map(
        filtered_las, classmap_path, output_path=output_dir, resolution=resolution
    )

    return classmap_path


def fetch_orthophoto(classmap_path, output_dir, tile_id):
    """
    Fetch WMS orthophoto matching the classification map bounds

    Args:
        classmap_path: Path to classification_map.tif
        output_dir: Directory to save orthophoto
        tile_id: Tile identifier

    Returns:
        Path to orthophoto.tif
    """
    ortho_path = output_dir / f"orthophoto_{tile_id}.tif"

    print(f"\nFetching WMS orthophoto for tile {tile_id}...")
    fetch_wms_for_raster(
        raster_path=str(classmap_path),
        output_path=str(ortho_path),
        wms_url=WMS_URL,
        layer_name=WMS_LAYER,
        image_format="image/jpeg",
    )
    print(f"✓ Saved orthophoto: {ortho_path}")

    return ortho_path


def process_tile(resolution, entry, output_dir):
    url = entry["url"]
    tile_id = entry["tile_id"]

    try:
        classmap_path = download_and_process_lidar(url, output_dir, resolution)
        fetch_orthophoto(classmap_path, output_dir, tile_id)

        return {"tile_id": tile_id, "status": "success"}

    except Exception as e:
        print(f"\n✗ Error processing {tile_id}: {e}")
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare data from LiDAR tiles using manifest"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest.json",
        help="Path to dataset manifest JSON file",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.2,
        help="Raster resolution in meters (default: 0.2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    start_time = time.time()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Error: Manifest not found: {manifest_path}")
        print("Run update_manifest.py first to create it")
        return

    print("=" * 70)
    print("PREPARING TRAINING DATA FROM LIDAR TILES (PARALLEL)")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")
    print(f"Resolution: {args.resolution}m")
    print(f"Workers: {args.workers}")
    print("=" * 70)

    manifest = load_manifest(manifest_path)
    output_dir = manifest_path.parent

    split_dirs = setup_split_directories(output_dir, ["train", "test"])
    all_tiles = build_tile_list(manifest, split_dirs)

    print(f"  Training: {len(manifest['train'])} tiles")
    print(f"  Testing: {len(manifest['test'])} tiles")

    process_func = partial(process_tile, args.resolution)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
