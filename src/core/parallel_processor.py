from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Callable

import humanize


def _future_outcome(future, tile_id, attempts, max_retries):
    """Resolve a finished future to (result_dict, requeue).

    requeue is True only when a worker death is still within the retry budget,
    in which case result_dict is None and the tile should be run again.
    """
    try:
        return future.result(), False
    except BrokenProcessPool:
        attempts[tile_id] = attempts.get(tile_id, 0) + 1
        if attempts[tile_id] <= max_retries:
            return None, True
        return {
            "tile_id": tile_id,
            "status": "failed",
            "error": (
                f"worker process killed {attempts[tile_id]} times "
                "(likely out of memory); giving up"
            ),
        }, False
    except Exception as e:
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}, False


def process_tiles_parallel(
    all_tiles: list[tuple],
    process_func: Callable,
    max_workers: int = 4,
    verbose: bool = True,
    max_retries: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Process tiles in parallel using ProcessPoolExecutor.

    A worker that is killed abruptly (e.g. by the OOM killer) breaks the whole
    pool: every in-flight future then raises BrokenProcessPool. To keep one death
    from losing the entire run, tiles already finished are kept and the in-flight
    tiles are requeued into a fresh pool. Each retry round halves the worker count
    so a memory-heavy or poison tile eventually runs isolated (1 worker) and fails
    on its own instead of taking the batch down.

    Args:
        all_tiles: List of tuples (entry, output_dir, split_name)
        process_func: Function to process single tile (entry, output_dir) -> result dict
        max_workers: Number of parallel workers
        verbose: Print progress messages
        max_retries: How many times a tile may be requeued after a worker death

    Returns:
        Tuple of (successful_results, failed_results)
    """
    total_tiles = len(all_tiles)

    if verbose:
        print(f"\nProcessing {total_tiles} tiles with {max_workers} workers...")

    results = []
    completed = 0
    attempts: dict = {}
    pending = list(all_tiles)
    round_idx = 0

    while pending:
        batch = pending
        pending = []
        round_workers = max(1, max_workers >> round_idx)
        if round_idx and verbose:
            print(
                f"\nRetry round {round_idx}: re-running {len(batch)} tile(s) "
                f"with {round_workers} worker(s)..."
            )

        executor = ProcessPoolExecutor(max_workers=round_workers)
        try:
            future_to_tile = {
                executor.submit(process_func, entry, output_dir): (
                    entry,
                    output_dir,
                    split,
                )
                for entry, output_dir, split in batch
            }

            for future in as_completed(future_to_tile):
                entry, output_dir, split = future_to_tile[future]
                tile_id = entry["tile_id"]

                result, requeue = _future_outcome(
                    future, tile_id, attempts, max_retries
                )
                if requeue:
                    pending.append((entry, output_dir, split))
                    continue

                completed += 1
                results.append(result)
                if verbose:
                    status = "✓" if result["status"] == "success" else "✗"
                    print(f"[{completed}/{total_tiles}] {status} {split:5s} {tile_id}")
                    if result["status"] == "failed":
                        print(f"  Error: {result.get('error', 'Unknown')}")
        finally:
            executor.shutdown(wait=False)

        round_idx += 1

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
