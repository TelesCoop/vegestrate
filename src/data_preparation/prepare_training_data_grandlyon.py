import argparse
import os
import time
from functools import partial
from pathlib import Path

import laspy
import numpy as np
import rasterio
import rasterio.enums
import rasterio.transform

from src.core import (
    build_tile_list,
    create_classification_map,
    create_ndsm,
    download_file,
    filter_ground_vegetation,
    load_manifest,
    print_processing_summary,
    process_tiles_parallel,
    resize_and_save,
    setup_split_directories,
)

IR_MOSAIC_URL = "https://data.grandlyon.com/files/grandlyon/imagerie/ortho2023/infrarouge/tiff/Vue_ensemble_5cm_CC46/IR2023_Dalle_unique_5cm_CC46.tif"


def extract_tile_name(url):
    filename = url.split("/")[-1]
    tile_name = filename.split(".")[0]
    return tile_name


def get_orthophoto_url(tile_name):
    base_url = "https://data.grandlyon.com/files/grandlyon/imagerie/ortho2023/ortho/tiff/500m_5cm_cc46"
    return f"{base_url}/{tile_name}_5cm_CC46.tif"


def download_ir_mosaic(output_path):
    """Download the full-coverage IR mosaic (~35 GB). Skips if already present.

    Args:
        output_path: Destination path for the downloaded GeoTIFF.

    Returns:
        Path to the IR mosaic.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"✓ IR mosaic already exists: {output_path}")
        return output_path

    print(f"Downloading IR mosaic (~35 GB) from {IR_MOSAIC_URL}")
    print("WARNING: this download will take a very long time.")
    download_file(IR_MOSAIC_URL, str(output_path))
    print(f"✓ IR mosaic downloaded to {output_path}")
    return output_path


def extract_ir_tile(tile_name, ir_mosaic_path, rgb_tile_path, output_dir):
    """Extract and resample the IR channel for one tile from the IR mosaic.

    Uses a windowed read so the 35 GB mosaic is never fully loaded.
    The output pixel grid matches the corresponding RGB orthophoto exactly.

    Args:
        tile_name: Tile identifier (e.g. "18435_51770").
        ir_mosaic_path: Path to the full IR mosaic GeoTIFF.
        rgb_tile_path: Matching RGB orthophoto (provides target bounds and size).
        output_dir: Directory to write the extracted IR tile.

    Returns:
        Path to the IR tile.
    """
    ir_path = output_dir / f"{tile_name}_ir.tif"
    if ir_path.exists():
        print(f"✓ IR tile already exists: {ir_path}")
        return ir_path

    with rasterio.open(rgb_tile_path) as rgb_src:
        bounds = rgb_src.bounds
        crs = rgb_src.crs
        target_height = rgb_src.height
        target_width = rgb_src.width

    with rasterio.open(ir_mosaic_path) as ir_src:
        window = ir_src.window(*bounds)
        data = ir_src.read(
            1,
            window=window,
            out_shape=(target_height, target_width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        is_uint16 = ir_src.dtypes[0] == "uint16"

    if is_uint16:
        data = (data / 257).astype(np.uint8)
    else:
        data = data.astype(np.uint8)

    tile_transform = rasterio.transform.from_bounds(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        target_width,
        target_height,
    )
    with rasterio.open(
        ir_path,
        "w",
        driver="GTiff",
        height=target_height,
        width=target_width,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=tile_transform,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    print(f"✓ IR tile saved: {ir_path}")
    return ir_path


def download_and_process_lidar(url, output_dir, resolution=0.2):
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
    try:
        las = laspy.read(str(laz_path))
    except Exception:
        laz_path.unlink(missing_ok=True)
        raise
    print(f"✓ Loaded {len(las.points):,} points")

    filtered_las = filter_ground_vegetation(las, lyon=True)

    classmap_path = output_dir / f"classification_map_{tile_name}.tif"
    create_classification_map(filtered_las, las, classmap_path, resolution=resolution)

    ndsm_path = output_dir / f"ndsm_{tile_name}.tif"
    create_ndsm(las, ndsm_path, resolution=resolution)

    os.remove(laz_path)
    return classmap_path


def download_orthophoto(tile_name, output_dir, resolution=0.2):
    ortho_path = output_dir / f"{tile_name}_orthophoto.tif"

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


def process_tile(resolution, entry, output_dir, ir_mosaic_path=None):
    url = entry["url"]
    tile_id = entry["tile_id"]

    try:
        tile_name = extract_tile_name(url)

        download_and_process_lidar(url, output_dir, resolution)
        ortho_path = download_orthophoto(tile_name, output_dir, resolution)

        if ir_mosaic_path is not None:
            extract_ir_tile(tile_name, ir_mosaic_path, ortho_path, output_dir)

        return {"tile_id": tile_id, "status": "success"}

    except Exception as e:
        print(f"\n✗ Error processing {tile_id}: {e}")
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}


def main():
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
        default=0.2,
        help="Raster resolution in meters (default: 0.2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Number of parallel workers (default: 14)",
    )
    parser.add_argument(
        "--ir_mosaic",
        type=str,
        default=None,
        help="Path to the IR mosaic GeoTIFF. If provided, IR tiles are extracted "
        "for each processed tile. Use --download_ir to fetch it first.",
    )
    parser.add_argument(
        "--download_ir",
        action="store_true",
        help="Download the IR mosaic before processing tiles.",
    )

    args = parser.parse_args()

    start_time = time.time()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Error: Manifest not found: {manifest_path}")
        print("Run update_manifest_grandlyon.py first to create it")
        return

    ir_mosaic_path = Path(args.ir_mosaic) if args.ir_mosaic else None

    if args.download_ir:
        if ir_mosaic_path is None:
            ir_mosaic_path = manifest_path.parent / "IR2023_Dalle_unique_5cm_CC46.tif"
        ir_mosaic_path = download_ir_mosaic(ir_mosaic_path)

    print("=" * 70)
    print("PREPARING TRAINING DATA FROM GRANDLYON LIDAR TILES (PARALLEL)")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")
    print(f"Resolution: {args.resolution}m")
    print(f"Workers: {args.workers}")
    if ir_mosaic_path:
        print(f"IR mosaic: {ir_mosaic_path}")
    print("=" * 70)

    manifest = load_manifest(str(manifest_path))
    output_dir = manifest_path.parent

    split_dirs = setup_split_directories(output_dir, ["train", "test"])
    all_tiles = build_tile_list(manifest, split_dirs)

    print(f"  Training: {len(manifest['train'])} tiles")
    print(f"  Testing: {len(manifest['test'])} tiles")

    process_func = partial(process_tile, args.resolution, ir_mosaic_path=ir_mosaic_path)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
