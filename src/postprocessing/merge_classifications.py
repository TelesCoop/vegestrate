from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def merge_classifications(las_path, flair_path, output_path):
    """
    Merge LIDAR and FLAIR classification rasters.

    Args:
        las_path: Path to LIDAR classification raster
        flair_path: Path to FLAIR prediction raster
        output_path: Path to save merged result
    """
    with rasterio.open(las_path) as src:
        las_data = src.read(1)
        profile = src.profile.copy()
        las_transform = src.transform
        las_crs = src.crs

    with rasterio.open(flair_path) as src:
        flair_data = src.read(1)
        flair_transform = src.transform
        flair_crs = src.crs
        flair_shape = flair_data.shape

    if las_data.shape != flair_data.shape:
        print(
            f"  Resampling FLAIR from {flair_shape} to {las_data.shape} using bicubic interpolation"
        )
        flair_resampled = np.zeros(las_data.shape, dtype=np.float32)
        reproject(
            source=flair_data,
            destination=flair_resampled,
            src_transform=flair_transform,
            src_crs=flair_crs,
            dst_transform=las_transform,
            dst_crs=las_crs,
            resampling=Resampling.cubic,
        )

        flair_data = np.round(flair_resampled).astype(np.uint8)
        flair_data = np.clip(flair_data, 0, 3)
        print("  Resampling complete")

    print(f"\nProcessing: {Path(las_path).name}")
    print(f"  Shape: {las_data.shape}")
    print(f"  LAS unique values: {np.unique(las_data)}")
    print(f"  FLAIR unique values: {np.unique(flair_data)}")

    result = las_data.copy()

    # Step 1: Remove unreliable LIDAR class 1 (herbaceous)
    n_las_class1 = np.sum(result == 1)
    result[result == 1] = 0
    print(f"  Removed {n_las_class1} LIDAR class 1 pixels")

    # Step 2: Add FLAIR class 1 (herbaceous) - FLAIR is reliable for this class
    n_flair_class1 = np.sum(flair_data == 1)
    result[flair_data == 1] = 1
    print(f"  Added {n_flair_class1} FLAIR class 1 pixels")

    # Step 3: Use FLAIR to fill in classes 2 and 3 ONLY where LIDAR is 0
    mask_class2 = (flair_data == 2) & (result == 0)
    mask_class3 = (flair_data == 3) & (result == 0)
    n_flair_class2 = np.sum(mask_class2)
    n_flair_class3 = np.sum(mask_class3)
    result[mask_class2] = 2
    result[mask_class3] = 3
    print(f"  Applied {n_flair_class2} FLAIR class 2 pixels (only where LIDAR==0)")
    print(f"  Applied {n_flair_class3} FLAIR class 3 pixels (only where LIDAR==0)")
    print("  Result distribution:")
    for class_val in [0, 1, 2, 3]:
        count = np.sum(result == class_val)
        pct = 100 * count / result.size
        print(f"    Class {class_val}: {count:8d} pixels ({pct:5.2f}%)")

    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result, 1)

    print(f"  Saved to: {output_path}")


def main():
    """Process all matching LIDAR and FLAIR rasters."""
    las_dir = Path("data/test")
    flair_dir = Path("predictions_lyon_08/test")
    output_dir = Path("merged_classifications_lyon_08/test")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Merging LIDAR and FLAIR Classifications")
    print("=" * 70)

    las_files = sorted(las_dir.glob("classification_map_*.tif"))

    print(f"\nFound {len(las_files)} LIDAR classification maps")

    processed = 0
    skipped = 0

    for las_path in las_files:
        las_name = las_path.stem
        coord_id = las_name.replace("classification_map_", "")

        flair_path = flair_dir / f"prediction_{coord_id}.tif"

        if not flair_path.exists():
            print(f"\nSkipping {las_name}: No matching FLAIR prediction")
            skipped += 1
            continue

        output_path = output_dir / f"merged_{coord_id}.tif"

        try:
            merge_classifications(las_path, flair_path, output_path)
            processed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped += 1
            continue

    print("\n" + "=" * 70)
    print(f"Completed: {processed} merged, {skipped} skipped")
    print("=" * 70)


if __name__ == "__main__":
    main()
