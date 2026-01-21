import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

from src.core import load_manifest
from src.inference.flair_segmentation import FlairSegmentation


def build_tile_map_from_manifest(
    manifest: dict, data_dir: Path
) -> dict[tuple[int, int], Path]:
    """Build coordinate -> path mapping from all manifest entries.

    Args:
        manifest: Loaded manifest dictionary
        data_dir: Base directory for resolving relative paths

    Returns:
        Dictionary mapping (x, y) coordinates to file paths
    """
    tile_map = {}
    all_entries = []
    for split in manifest.keys():
        all_entries.extend(manifest[split])

    for entry in all_entries:
        tile_id = entry["tile_id"]
        ortho_rel_path = entry["orthophoto"]
        ortho_path = data_dir / ortho_rel_path

        coords = parse_tile_id(tile_id)
        if coords and ortho_path.exists():
            tile_map[coords] = ortho_path

    return tile_map


def initialize_model(checkpoint_path: Path) -> FlairSegmentation:
    """Initialize FLAIR segmentation model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Initialized FlairSegmentation model
    """
    checkpoint_num_classes = 19 if checkpoint_path.suffix == ".safetensors" else 4

    print("\nInitializing FLAIR model...")
    model = FlairSegmentation(
        checkpoint_path=str(checkpoint_path),
        use_simplified_classes=True,
        checkpoint_num_classes=checkpoint_num_classes,
    )

    print("\nAvailable classes (4 simplified classes):")
    for class_id, class_name in model.list_classes().items():
        print(f"  {class_id:2d}: {class_name}")

    return model


def process_split(
    split: str,
    entries: list,
    data_dir: Path,
    output_dir: Path,
    tile_map: dict[tuple[int, int], Path],
    model: FlairSegmentation,
    tile_size: int,
    overlap: int,
    grid_step: int,
    use_tta: bool = True,
    tta_modes: Optional[list[str]] = None,
    class_logit_bias: Optional[dict[int, float]] = None,
    herbaceous_recovery_margin: Optional[float] = None,
) -> int:
    """Process all tiles in a split.

    Args:
        split: Split name (e.g., 'train', 'test')
        entries: List of manifest entries for this split
        data_dir: Base data directory
        output_dir: Base output directory
        tile_map: Coordinate to path mapping
        model: FlairSegmentation model
        tile_size: Tile size for processing
        overlap: Overlap for tiled prediction
        grid_step: Grid step between tiles
        use_tta: Enable TTA (default: True)
        tta_modes: TTA modes (default: None uses model defaults)
        class_logit_bias: Dict mapping class_id to logit bias (e.g., {1: 2.0})
        herbaceous_recovery_margin: Margin to recover herbaceous from else (e.g., 3.0)

    Returns:
        Number of successfully processed tiles
    """
    print(f"\n{'=' * 70}")
    print(f"Processing {split.upper()} split ({len(entries)} images)")
    print(f"{'=' * 70}")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for i, entry in enumerate(entries, 1):
        tile_id = entry["tile_id"]
        ortho_rel_path = entry["orthophoto"]
        ortho_path = data_dir / ortho_rel_path

        if not ortho_path.exists():
            print(f"\n[{i}/{len(entries)}] ✗ {tile_id}")
            print(f"  Image not found: {ortho_path}")
            continue

        print(f"\n[{i}/{len(entries)}] Processing {tile_id}...")
        print(f"  Input: {ortho_path}")

        if process_tile_with_context(
            tile_id=tile_id,
            tile_map=tile_map,
            model=model,
            output_dir=split_dir,
            grid_step=grid_step,
            tile_size=tile_size,
            overlap=overlap,
            use_tta=use_tta,
            tta_modes=tta_modes,
            class_logit_bias=class_logit_bias,
            herbaceous_recovery_margin=herbaceous_recovery_margin,
        ):
            processed_count += 1

    return processed_count


def parse_tile_id(tile_id: str) -> Optional[tuple[int, int]]:
    """Extract X,Y coordinates from tile ID.

    Args:
        tile_id: e.g., '18430_51735' or 'orthophoto_18430_51735.tif'

    Returns:
        Tuple of (x, y) or None if parsing fails
    """
    if tile_id.endswith(".tif"):
        tile_id = Path(tile_id).stem

    tile_id = tile_id.replace("orthophoto_", "")
    parts = tile_id.split("_")

    if len(parts) != 2:
        return None

    try:
        x, y = int(parts[0]), int(parts[1])
        return (x, y)
    except ValueError:
        return None


def find_tile_neighbors(
    tile_coords: tuple[int, int], tile_map: dict[tuple[int, int], Path], grid_step: int
) -> dict[str, Optional[Path]]:
    """Find the 8 neighboring tiles around a center tile.

    Args:
        tile_coords: (x, y) coordinates of center tile
        tile_map: Dictionary mapping (x,y) -> file path
        grid_step: Grid spacing between tiles (typically 5)

    Returns:
        Dictionary with keys: NW, N, NE, W, C, E, SW, S, SE
        Values are Path objects or None if neighbor doesn't exist
    """
    x, y = tile_coords

    neighbors = {
        "NW": (x - grid_step, y + grid_step),
        "N": (x, y + grid_step),
        "NE": (x + grid_step, y + grid_step),
        "W": (x - grid_step, y),
        "C": (x, y),
        "E": (x + grid_step, y),
        "SW": (x - grid_step, y - grid_step),
        "S": (x, y - grid_step),
        "SE": (x + grid_step, y - grid_step),
    }

    return {pos: tile_map.get(coords) for pos, coords in neighbors.items()}


def create_mosaic_from_tiles(
    neighbors: dict[str, Optional[Path]], tile_size: int = 625
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """Create 3x3 mosaic from neighboring tiles.

    Args:
        neighbors: Dictionary from find_tile_neighbors()
        tile_size: Size of each tile (default: 625)

    Returns:
        Tuple of (mosaic_array, metadata) or (None, None) if center missing
    """
    if neighbors["C"] is None:
        return None, None

    mosaic_size = tile_size * 3
    mosaic = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)

    # Grid layout positions
    layout = [
        ["NW", "N", "NE"],
        ["W", "C", "E"],
        ["SW", "S", "SE"],
    ]

    center_meta = None

    for row_idx, row in enumerate(layout):
        for col_idx, pos in enumerate(row):
            tile_path = neighbors[pos]

            if tile_path is None or not tile_path.exists():
                # Leave as zeros (black) for missing tiles
                continue

            with rasterio.open(tile_path) as src:
                if pos == "C":
                    center_meta = {
                        "crs": src.crs,
                        "transform": src.transform,
                        "bounds": src.bounds,
                    }

                img = src.read([1, 2, 3])
                img = np.transpose(img, (1, 2, 0))

                y_start = row_idx * tile_size
                x_start = col_idx * tile_size
                mosaic[y_start : y_start + tile_size, x_start : x_start + tile_size] = (
                    img
                )

    return mosaic, center_meta


def process_tile_with_context(
    tile_id: str,
    tile_map: dict[tuple[int, int], Path],
    model: FlairSegmentation,
    output_dir: Path,
    grid_step: int,
    tile_size: int,
    overlap: int,
    use_tta: bool = True,
    tta_modes: Optional[list[str]] = None,
    class_logit_bias: Optional[dict[int, float]] = None,
    herbaceous_recovery_margin: Optional[float] = None,
) -> bool:
    """Process a single tile with neighboring context.

    Args:
        tile_id: Tile ID (e.g., '18430_51735')
        tile_map: Map of all available tiles
        model: FlairSegmentation model
        output_dir: Output directory for predictions
        grid_step: Grid spacing
        tile_size: Tile size for prediction
        overlap: Overlap for tiled prediction
        use_tta: Enable TTA (default: True)
        tta_modes: TTA modes (default: None uses model defaults)
        class_logit_bias: Dict mapping class_id to logit bias (e.g., {1: 2.0})
        herbaceous_recovery_margin: Margin to recover herbaceous from else

    Returns:
        True if successful, False otherwise
    """
    coords = parse_tile_id(tile_id)
    if coords is None:
        print(f"  ✗ Could not parse coordinates from {tile_id}")
        return False

    neighbors = find_tile_neighbors(coords, tile_map, grid_step)

    mosaic, center_meta = create_mosaic_from_tiles(neighbors, tile_size=625)
    if mosaic is None or center_meta is None:
        print("  ✗ Failed to create mosaic")
        return False

    temp_mosaic_path = output_dir / f"temp_mosaic_{coords[0]}_{coords[1]}.tif"

    with rasterio.open(
        temp_mosaic_path,
        "w",
        driver="GTiff",
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=3,
        dtype=mosaic.dtype,
        crs=center_meta["crs"],
        transform=center_meta["transform"],
    ) as dst:
        for i in range(3):
            dst.write(mosaic[:, :, i], i + 1)

    temp_pred_path = output_dir / f"temp_pred_{coords[0]}_{coords[1]}.tif"

    try:
        model.segment_image(
            image_path=str(temp_mosaic_path),
            output_path=str(temp_pred_path),
            class_id=None,
            tile_size=tile_size,
            overlap=overlap,
            use_tta=use_tta,
            tta_modes=tta_modes,
            class_logit_bias=class_logit_bias,
            herbaceous_recovery_margin=herbaceous_recovery_margin,
        )
    finally:
        temp_mosaic_path.unlink(missing_ok=True)

    with rasterio.open(temp_pred_path) as src:
        # Center tile starts at (625, 625) and goes to (1250, 1250)
        center_data = src.read(1, window=Window(625, 625, 625, 625))

    output_path = output_dir / f"prediction_{coords[0]}_{coords[1]}.tif"

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=625,
        width=625,
        count=1,
        dtype=center_data.dtype,
        crs=center_meta["crs"],
        transform=center_meta["transform"],
        nodata=255,
        compress="lzw",
    ) as dst:
        dst.write(center_data, 1)

    # Clean up temp prediction
    temp_pred_path.unlink(missing_ok=True)

    print(f"  ✓ Saved to {output_path}")
    return True


def parse_class_bias(
    class_bias_args: Optional[list[str]],
) -> Optional[dict[int, float]]:
    if not class_bias_args:
        return None
    result = {}
    for pair in class_bias_args:
        class_id, bias = pair.split(":")
        result[int(class_id)] = float(bias)
    return result


def print_config(
    args, manifest_path, checkpoint_path, output_dir, use_tta, class_logit_bias
):
    print("=" * 70)
    print("FLAIR INFERENCE WITH CONTEXT (High Precision Mode)")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {args.tile_size}")
    print(f"Overlap: {args.overlap} ({100 * args.overlap / args.tile_size:.0f}%)")
    print(f"Grid step: {args.grid_step}")
    tta_str = ", ".join(args.tta_modes) if args.tta_modes else "hflip, vflip, hvflip"
    print(f"TTA: {'ENABLED (' + tta_str + ')' if use_tta else 'disabled'}")
    if class_logit_bias:
        bias_str = ", ".join(
            f"class {k}: {v:+.1f}" for k, v in class_logit_bias.items()
        )
        print(f"Class logit bias: {bias_str}")
    if args.herb_margin > 0:
        print(f"Herbaceous recovery margin: {args.herb_margin}")
    print(f"Processing splits: {', '.join(args.splits)}")
    print("=" * 70)


def print_tile_map_info(tile_map: dict):
    print(f"Found {len(tile_map)} tiles in manifest")
    if tile_map:
        print(
            f"X range: {min(c[0] for c in tile_map)} to {max(c[0] for c in tile_map)}"
        )
        print(
            f"Y range: {min(c[1] for c in tile_map)} to {max(c[1] for c in tile_map)}"
        )


def print_summary(total_processed: int, output_dir: Path, splits: list[str]):
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total images processed: {total_processed}")
    print(f"Predictions saved to: {output_dir}")
    print("\nOutput structure:")
    for split in splits:
        split_dir = output_dir / split
        if split_dir.exists():
            predictions = list(split_dir.glob("prediction_*.tif"))
            print(f"  {split}/: {len(predictions)} predictions")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLAIR inference with neighboring tile context"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_grandlyon.json",
        help="Path to dataset manifest JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="FLAIR-HUB_LC-A_RGB_swinlarge-upernet.safetensors",
        help="Path to FLAIR checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions_context",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=384,
        help="Tile size for processing mosaic (default: 384, checkpoint native size)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Tile overlap in pixels (default: 192, which is 50% for smooth blending)",
    )
    parser.add_argument(
        "--grid_step",
        type=int,
        default=5,
        help="Grid step between tile coordinates (default: 5)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Splits to process (default: train test)",
    )
    parser.add_argument(
        "--no_tta",
        action="store_true",
        help="Disable Test-Time Augmentation (enabled by default for better precision)",
    )
    parser.add_argument(
        "--tta_modes",
        type=str,
        nargs="+",
        default=None,
        help="TTA augmentation modes (default: hflip vflip hvflip for 4x)",
    )
    parser.add_argument(
        "--class_bias",
        type=str,
        nargs="+",
        default=None,
        help="Class logit bias as CLASS:BIAS pairs (ex: 1:2.0 to boost herbaceous)",
    )
    parser.add_argument(
        "--herb_margin",
        type=float,
        default=3.0,
        help="Herbaceous recovery margin: pixels classified as 'else' where herbaceous "
        "logit was within this margin get flipped to herbaceous (default: 3.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(str(manifest_path))
    data_dir = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    use_tta = not args.no_tta
    class_logit_bias = parse_class_bias(args.class_bias)

    print_config(
        args, manifest_path, checkpoint_path, output_dir, use_tta, class_logit_bias
    )

    print("\nBuilding tile map from manifest...")
    tile_map = build_tile_map_from_manifest(manifest, data_dir)
    print_tile_map_info(tile_map)

    model = initialize_model(checkpoint_path)

    total_processed = 0
    for split in args.splits:
        if split not in manifest:
            print(f"\n✗ Warning: Split '{split}' not in manifest")
            continue

        total_processed += process_split(
            split=split,
            entries=manifest[split],
            data_dir=data_dir,
            output_dir=output_dir,
            tile_map=tile_map,
            model=model,
            tile_size=args.tile_size,
            overlap=args.overlap,
            grid_step=args.grid_step,
            use_tta=use_tta,
            tta_modes=args.tta_modes,
            class_logit_bias=class_logit_bias,
            herbaceous_recovery_margin=args.herb_margin,
        )

    print_summary(total_processed, output_dir, args.splits)


if __name__ == "__main__":
    main()
