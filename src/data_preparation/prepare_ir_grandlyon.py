import argparse
import time
from functools import partial
from pathlib import Path

from src.core import (
    build_tile_list,
    load_manifest,
    print_processing_summary,
    process_tiles_parallel,
    setup_split_directories,
)
from src.data_preparation.prepare_training_data_grandlyon import (
    IR_MOSAIC_URL,
    download_ir_mosaic,
    extract_ir_tile,
)


def process_tile(ir_mosaic_path, manifest_dir, entry, output_dir):
    tile_id = entry["tile_id"]
    orthophoto_rel = entry.get("orthophoto")

    if orthophoto_rel is None:
        return {
            "tile_id": tile_id,
            "status": "failed",
            "error": "no orthophoto in manifest",
        }

    orthophoto_path = manifest_dir / orthophoto_rel
    if not orthophoto_path.exists():
        return {
            "tile_id": tile_id,
            "status": "failed",
            "error": f"orthophoto not found: {orthophoto_path}",
        }

    try:
        extract_ir_tile(tile_id, ir_mosaic_path, orthophoto_path, Path(output_dir))
        return {"tile_id": tile_id, "status": "success"}
    except Exception as e:
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Download IR mosaic and extract IR tiles alongside existing orthophotos"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_grandlyon.json",
    )
    parser.add_argument(
        "--ir_mosaic",
        type=str,
        default=None,
        help=f"Path to the IR mosaic GeoTIFF. Default location when using --download_ir: data/{IR_MOSAIC_URL.split('/')[-1]}",
    )
    parser.add_argument(
        "--download_ir",
        action="store_true",
        help="Download the IR mosaic before processing tiles.",
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
        print(f"Manifest not found: {manifest_path}")
        print("Run update_manifest_grandlyon.py first to create it")
        return

    ir_mosaic_path = Path(args.ir_mosaic) if args.ir_mosaic else None

    if args.download_ir:
        if ir_mosaic_path is None:
            ir_mosaic_path = manifest_path.parent / IR_MOSAIC_URL.split("/")[-1]
        ir_mosaic_path = download_ir_mosaic(ir_mosaic_path)

    if ir_mosaic_path is None or not ir_mosaic_path.exists():
        print(
            "IR mosaic not available. Use --ir_mosaic to point to an existing file or --download_ir to fetch it."
        )
        return

    print("=" * 70)
    print("EXTRACTING IR TILES FROM GRANDLYON IR MOSAIC")
    print("=" * 70)
    print(f"Manifest:  {manifest_path}")
    print(f"IR mosaic: {ir_mosaic_path}")
    print(f"Workers:   {args.workers}")
    print("=" * 70)

    manifest = load_manifest(str(manifest_path))
    manifest_dir = manifest_path.parent

    split_dirs = setup_split_directories(manifest_dir, ["train", "test"])
    all_tiles = build_tile_list(manifest, split_dirs)

    print(f"  Training: {len(manifest['train'])} tiles")
    print(f"  Testing:  {len(manifest['test'])} tiles")

    process_func = partial(process_tile, ir_mosaic_path, manifest_dir)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
