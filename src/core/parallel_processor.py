from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import humanize


def process_tiles_parallel(
    all_tiles: list[tuple],
    process_func: Callable,
    max_workers: int = 4,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Process tiles in parallel using ThreadPoolExecutor.

    Args:
        all_tiles: List of tuples (entry, output_dir, split_name)
        process_func: Function to process single tile (entry, output_dir) -> result dict
        max_workers: Number of parallel workers
        verbose: Print progress messages

    Returns:
        Tuple of (successful_results, failed_results)
    """
    total_tiles = len(all_tiles)

    if verbose:
        print(f"\nProcessing {total_tiles} tiles with {max_workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {
            executor.submit(process_func, entry, output_dir): (entry["tile_id"], split)
            for entry, output_dir, split in all_tiles
        }

        completed = 0
        for future in as_completed(future_to_tile):
            tile_id, split = future_to_tile[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)
                status = "✓" if result["status"] == "success" else "✗"
                if verbose:
                    print(f"[{completed}/{total_tiles}] {status} {split:5s} {tile_id}")
            except Exception as e:
                if verbose:
                    print(f"[{completed}/{total_tiles}] ✗ {split:5s} {tile_id}")
                    print(f"  Error: {e}")
                results.append(
                    {"tile_id": tile_id, "status": "failed", "error": str(e)}
                )

    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]

    return successes, failures


def print_processing_summary(
    successes: list[dict],
    failures: list[dict],
    elapsed_time: float,
    verbose: bool = True,
):
    """Print summary of tile processing results.

    Args:
        successes: List of successful result dictionaries
        failures: List of failed result dictionaries
        elapsed_time: Total elapsed time in seconds
        verbose: Print detailed information
    """
    total = len(successes) + len(failures)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Successful: {len(successes)}/{total}")
    print(f"Failed: {len(failures)}/{total}")

    if failures and verbose:
        print("\nFailed tiles:")
        for result in failures:
            print(f"  ✗ {result['tile_id']}: {result.get('error', 'Unknown')}")

    print(f"\nTotal time: {humanize.naturaldelta(elapsed_time)}")
    if total > 0:
        avg_time = elapsed_time / total
        print(f"Average time per tile: {avg_time:.1f}s")


def build_tile_list(manifest: dict, split_dirs: dict[str, any]) -> list[tuple]:
    """Build list of tiles to process from manifest.

    Args:
        manifest: Manifest dictionary with 'train', 'test', etc.
        split_dirs: Dictionary mapping split name to output directory

    Returns:
        List of tuples (entry, output_dir, split_name)
    """
    all_tiles = []
    for split, split_dir in split_dirs.items():
        if split in manifest:
            for entry in manifest[split]:
                all_tiles.append((entry, split_dir, split))
    return all_tiles
