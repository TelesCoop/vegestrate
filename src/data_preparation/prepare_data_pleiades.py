import argparse
import time
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_bounds

from src.core import (
    build_tile_list,
    load_manifest,
    print_processing_summary,
    process_tiles_parallel,
    setup_split_directories,
)

WMS_URL = "https://data.geopf.fr/wms-r"
CRS = "EPSG:2154"

PLEIADES_RGB_LAYER = "ORTHOIMAGERY.ORTHO-SAT.PLEIADES.{year}"
PLEIADES_IRC_LAYER = "ORTHOIMAGERY.ORTHO-SAT.PLEIADES.{year}.IRC"


def tile_id_to_bounds(tile_id, tile_step):
    x, y = map(int, tile_id.split("_"))
    step_m = tile_step * 100.0
    xmin = x * 100.0
    ymin = y * 100.0
    return xmin, ymin, xmin + step_m, ymin + step_m


def fetch_wms_tile(bounds, width_px, height_px, layer_name):
    xmin, ymin, xmax, ymax = bounds
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer_name,
        "CRS": CRS,
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": width_px,
        "HEIGHT": height_px,
        "FORMAT": "image/jpeg",
        "STYLES": "",
    }
    response = requests.get(WMS_URL, params=params, timeout=120)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise ValueError(
            f"WMS returned non-image: {content_type}: {response.text[:200]}"
        )
    return response.content


def save_as_geotiff(arr, bounds, output_path):
    xmin, ymin, xmax, ymax = bounds
    if arr.ndim == 2:
        height, width = arr.shape
        count = 1
    else:
        height, width, count = arr.shape

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=arr.dtype,
        crs=CRS,
        transform=transform,
        compress="lzw",
    ) as dst:
        if count == 1:
            dst.write(arr if arr.ndim == 2 else arr[:, :, 0], 1)
        else:
            for i in range(count):
                dst.write(arr[:, :, i], i + 1)


def process_tile(tile_step, resolution, year, use_ir, entry, output_dir):
    tile_id = entry["tile_id"]
    ortho_path = Path(output_dir) / f"{tile_id}_orthophoto.tif"

    if not ortho_path.exists():
        bounds = tile_id_to_bounds(tile_id, tile_step)
        step_m = tile_step * 100.0
        px = int(step_m / resolution)

        layer = PLEIADES_RGB_LAYER.format(year=year)
        img_bytes = fetch_wms_tile(bounds, px, px, layer)
        arr = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        save_as_geotiff(arr, bounds, ortho_path)
        print(f"  ✓ {ortho_path.name}")
    else:
        print(f"  ⏭ {ortho_path.name}")

    if use_ir:
        ir_path = Path(output_dir) / f"{tile_id}_ir.tif"
        if not ir_path.exists():
            try:
                bounds = tile_id_to_bounds(tile_id, tile_step)
                step_m = tile_step * 100.0
                px = int(step_m / resolution)

                irc_layer = PLEIADES_IRC_LAYER.format(year=year)
                irc_bytes = fetch_wms_tile(bounds, px, px, irc_layer)
                irc_arr = np.array(Image.open(BytesIO(irc_bytes)))
                ir_band = irc_arr[:, :, 0] if irc_arr.ndim == 3 else irc_arr
                save_as_geotiff(ir_band.astype(np.uint8), bounds, ir_path)
                print(f"  ✓ {ir_path.name}")
            except Exception as e:
                print(f"  WARNING: IR fetch failed for {tile_id}: {e}")
        else:
            print(f"  ⏭ {ir_path.name}")

    return {"tile_id": tile_id, "status": "success"}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Pléiades orthoimagery tiles from IGN WMS"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_pleiades.json",
        help="Path to Pléiades manifest JSON",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Pléiades year to fetch (e.g. 2023)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Pixel resolution in meters (default: 0.5)",
    )
    parser.add_argument(
        "--tile_step",
        type=int,
        default=5,
        help="Tile size in 100m units, must match manifest (default: 5 = 500m)",
    )
    parser.add_argument(
        "--ir",
        action="store_true",
        help="Also fetch IRC channel and save as <tile_id>_ir.tif",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        print("Run update_manifest_pleiades.py first")
        return

    print("=" * 70)
    print("FETCH PLÉIADES TILES FROM IGN WMS")
    print("=" * 70)
    print(f"Manifest:   {manifest_path}")
    print(f"Year:       {args.year}")
    print(f"Resolution: {args.resolution}m/px")
    print(
        f"Tile size:  {args.tile_step * 100}m ({int(args.tile_step * 100 / args.resolution)}px)"
    )
    print(f"IR channel: {'yes' if args.ir else 'no'}")
    print(f"Workers:    {args.workers}")
    print("=" * 70)

    manifest = load_manifest(str(manifest_path))
    data_dir = manifest_path.parent

    split_dirs = setup_split_directories(data_dir, args.splits)

    n_train = len(manifest.get("train", []))
    n_test = len(manifest.get("test", []))
    print(f"  Training: {n_train} tiles")
    print(f"  Testing:  {n_test} tiles")

    all_tiles = build_tile_list(manifest, split_dirs)

    start_time = time.time()
    process_func = partial(
        process_tile, args.tile_step, args.resolution, args.year, args.ir
    )
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
