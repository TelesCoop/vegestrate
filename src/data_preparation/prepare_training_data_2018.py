import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
import rasterio.enums
import rasterio.warp

from src.core import (
    build_tile_list,
    download_file,
    load_manifest,
    print_processing_summary,
    process_tiles_parallel,
    setup_split_directories,
)
from src.data_preparation.prepare_training_data_grandlyon import (
    download_and_process_lidar,
)


def extract_ir_from_tif(tile_id, tif_url, classmap_path, output_dir):
    ortho_path = output_dir / f"{tile_id}_orthophoto.tif"
    ir_path = output_dir / f"{tile_id}_ir.tif"

    if ortho_path.exists() and ir_path.exists():
        return ortho_path, ir_path

    tmp = output_dir / f"{tile_id}_ir_src.tmp.tif"
    download_file(tif_url, str(tmp))

    with rasterio.open(classmap_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(tmp) as src:
        src_crs = src.crs
        src_transform = src.transform
        band_count = src.count
        is_uint16 = src.dtypes[0] == "uint16"
        src_data = src.read()

    tmp.unlink()

    if is_uint16:
        src_data = (src_data / 257).astype(np.uint8)
    else:
        src_data = src_data.astype(np.uint8)

    dst_data = np.zeros((band_count, dst_height, dst_width), dtype=np.uint8)
    rasterio.warp.reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=rasterio.enums.Resampling.bilinear,
    )

    if not ortho_path.exists():
        with rasterio.open(
            ortho_path,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=band_count,
            dtype=np.uint8,
            crs=dst_crs,
            transform=dst_transform,
            compress="lzw",
        ) as dst:
            dst.write(dst_data)
        print(f"✓ Orthophoto saved: {ortho_path}")

    if not ir_path.exists():
        with rasterio.open(
            ir_path,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype=np.uint8,
            crs=dst_crs,
            transform=dst_transform,
            compress="lzw",
        ) as dst:
            dst.write(dst_data[0:1])
        print(f"✓ IR tile saved: {ir_path}")

    return ortho_path, ir_path


def process_tile(resolution, entry, output_dir):
    tile_id = entry["tile_id"]
    url = entry["url"]
    ortho_tif_url = entry.get("ortho_tif_url")

    try:
        classmap_path = download_and_process_lidar(url, output_dir, resolution)

        if ortho_tif_url:
            extract_ir_from_tif(tile_id, ortho_tif_url, classmap_path, output_dir)
        else:
            print(f"⚠ No ortho TIF URL for {tile_id}, skipping orthophoto")

        return {"tile_id": tile_id, "status": "success"}

    except Exception as e:
        print(f"\n✗ Error processing {tile_id}: {e}")
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 2018 data from GrandLyon LiDAR tiles + IR TIFF orthophotos"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_2018.json",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
    )

    args = parser.parse_args()
    start_time = time.time()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        print("Run update_manifest_2018.py first")
        return

    print("=" * 70)
    print("PREPARING 2018 DATA FROM GRANDLYON LIDAR + IR TIFF ORTHOPHOTOS")
    print("=" * 70)
    print(f"Manifest:   {manifest_path}")
    print(f"Resolution: {args.resolution}m")
    print(f"Workers:    {args.workers}")
    print("=" * 70)

    manifest = load_manifest(str(manifest_path))
    output_dir = manifest_path.parent

    split_dirs = setup_split_directories(output_dir, ["train", "test"])
    all_tiles = build_tile_list(manifest, split_dirs)

    print(f"  Training: {len(manifest['train'])} tiles")
    print(f"  Testing:  {len(manifest['test'])} tiles")

    process_func = partial(process_tile, args.resolution)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
