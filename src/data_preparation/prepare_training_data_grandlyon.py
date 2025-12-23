#!/usr/bin/env python3
"""Prepare training and testing data from GrandLyon LiDAR tiles.

This script reads data/dataset_manifest_grandlyon.json and processes tiles:
- Downloads LiDAR data (if not exists)
- Generates classification_map.tif
- Downloads corresponding orthophoto directly using tile name

Supports parallel processing using threads for faster execution.

Usage:
    # First generate manifest
    python update_manifest_grandlyon.py

    # Then prepare data
    python prepare_training_data_grandlyon.py --manifest data/dataset_manifest_grandlyon.json

    # Use more threads for faster processing
    python prepare_training_data_grandlyon.py --workers 8
"""

import argparse
import time
from pathlib import Path

import laspy
import rasterio
import os

from src.core import (
    build_tile_list,
    create_classification_map,
    download_file,
    filter_ground_vegetation,
    load_manifest,
    print_processing_summary,
    process_tiles_parallel,
    resize_and_save,
    setup_split_directories,
)


def extract_tile_name(url):
    """Extract tile name from GrandLyon URL

    Example:
        https://data.grandlyon.com/files/grandlyon/imagerie/mnt2023/lidar/laz/18435_51770.laz
        -> 18435_51770
    """
    filename = url.split("/")[-1]
    tile_name = filename.split(".")[0]
    return tile_name


def get_orthophoto_url(tile_name):
    """Generate orthophoto URL from tile name

    Args:
        tile_name: Tile identifier like "18435_51770"

    Returns:
        URL to orthophoto TIF file
    """
    base_url = "https://data.grandlyon.com/files/grandlyon/imagerie/ortho2023/ortho/tiff/500m_5cm_cc46"
    return f"{base_url}/{tile_name}_5cm_CC46.tif"


def download_and_process_lidar(url, output_dir, resolution=0.8):
    """
    Download LiDAR data and create classification map

    Args:
        url: LiDAR download URL
        output_dir: Directory to save outputs
        resolution: Raster cell size in meters (default: 0.8)

    Returns:
        Path to classification_map.tif
    """
    filename = url.split("/")[-1]
    tile_name = extract_tile_name(url)
    laz_path = output_dir / filename

    print(f"\n{'=' * 70}")
    print(f"Processing tile: {tile_name}")
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

    filtered_las = filter_ground_vegetation(las, lyon=True)

    classmap_path = output_dir / f"classification_map_{tile_name}.tif"

    create_classification_map(filtered_las, las, classmap_path, resolution=resolution)
    os.remove(laz_path)
    return classmap_path


def download_orthophoto(tile_name, output_dir, resolution=0.8):
    """
    Download orthophoto directly from GrandLyon and resize to match classification map

    Args:
        tile_name: Tile identifier (e.g., "18435_51770")
        output_dir: Directory to save orthophoto
        resolution: Target resolution in meters (default: 0.8)

    Returns:
        Path to orthophoto.tif
    """
    ortho_path = output_dir / f"orthophoto_{tile_name}.tif"

    if ortho_path.exists():
        print(f"✓ Orthophoto already exists: {ortho_path}")
        return ortho_path

    ortho_url = get_orthophoto_url(tile_name)
    print(f"\nDownloading orthophoto from {ortho_url}...")
    temp_path = output_dir / f"temp_{tile_name}_5cm_CC46.tif"
    download_file(ortho_url, str(temp_path))

    with rasterio.open(temp_path) as src:
        bounds = src.bounds
        crs = src.crs

    resize_and_save(
        raster_path=str(temp_path),
        resolution=resolution,
        bounds=bounds,
        crs=crs,
        output_path=str(ortho_path),
    )

    temp_path.unlink()

    print(f"✓ Saved orthophoto: {ortho_path}")

    return ortho_path


def process_tile_wrapper(resolution):
    """Create a process_tile function with fixed resolution.

    Args:
        resolution: Raster resolution in meters

    Returns:
        Function that processes a single tile
    """

    def process_tile(entry, output_dir):
        """Process a single tile (download LiDAR, create classification map, download orthophoto).

        Args:
            entry: Manifest entry with 'url' and 'tile_id'
            output_dir: Directory to save outputs

        Returns:
            Dictionary with tile_id and status
        """
        url = entry["url"]
        tile_id = entry["tile_id"]

        try:
            tile_name = extract_tile_name(url)

            download_and_process_lidar(url, output_dir, resolution)
            download_orthophoto(tile_name, output_dir, resolution)

            return {"tile_id": tile_id, "status": "success"}

        except Exception as e:
            print(f"\n✗ Error processing {tile_id}: {e}")
            return {"tile_id": tile_id, "status": "failed", "error": str(e)}

    return process_tile


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare data from GrandLyon LiDAR tiles using manifest"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_grandlyon.json",
        help="Path to dataset manifest JSON file",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.8,
        help="Raster resolution in meters (default: 0.8)",
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
        print("Run update_manifest_grandlyon.py first to create it")
        return

    print("=" * 70)
    print("PREPARING TRAINING DATA FROM GRANDLYON LIDAR TILES (PARALLEL)")
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

    process_func = process_tile_wrapper(args.resolution)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
