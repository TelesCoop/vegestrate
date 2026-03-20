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
    download_and_process_lidar,
    extract_ir_from_ecw,
    extract_orthophoto_from_ecw,
)

ECW_CACHE_DIR = Path("data/ecw_cache")


def process_tile(resolution, ecw_cache_dir, entry, output_dir):
    tile_id = entry["tile_id"]
    url = entry["url"]
    ortho_ecw_url = entry.get("ortho_ecw_url")

    try:
        classmap_path = download_and_process_lidar(url, output_dir, resolution)

        if ortho_ecw_url:
            extract_orthophoto_from_ecw(
                tile_id, ortho_ecw_url, classmap_path, output_dir, ecw_cache_dir
            )
            extract_ir_from_ecw(
                tile_id, ortho_ecw_url, classmap_path, output_dir, ecw_cache_dir
            )
        else:
            print(f"⚠ No ortho ECW URL for {tile_id}, skipping orthophoto")

        return {"tile_id": tile_id, "status": "success"}

    except Exception as e:
        print(f"\n✗ Error processing {tile_id}: {e}")
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 2018 data from GrandLyon LiDAR tiles + IR ECW orthophotos"
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
    parser.add_argument(
        "--ecw_cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded ECW ortho tiles (default: data/ecw_cache)",
    )

    args = parser.parse_args()
    start_time = time.time()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        print("Run update_manifest_2018.py first")
        return

    ecw_cache_dir = (
        Path(args.ecw_cache_dir)
        if args.ecw_cache_dir
        else manifest_path.parent / "ecw_cache"
    )

    print("=" * 70)
    print("PREPARING 2018 DATA FROM GRANDLYON LIDAR + IR ECW ORTHOPHOTOS")
    print("=" * 70)
    print(f"Manifest:      {manifest_path}")
    print(f"Resolution:    {args.resolution}m")
    print(f"Workers:       {args.workers}")
    print(f"ECW cache dir: {ecw_cache_dir}")
    print("=" * 70)

    manifest = load_manifest(str(manifest_path))
    output_dir = manifest_path.parent

    split_dirs = setup_split_directories(output_dir, ["train", "test"])
    all_tiles = build_tile_list(manifest, split_dirs)

    print(f"  Training: {len(manifest['train'])} tiles")
    print(f"  Testing:  {len(manifest['test'])} tiles")

    process_func = partial(process_tile, args.resolution, ecw_cache_dir)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


if __name__ == "__main__":
    main()
