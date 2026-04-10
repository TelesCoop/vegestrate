import argparse
import sys

import numpy as np
import rasterio


def apply_vegetation_elevation(veg_path, ndsm_path, output_path, nodata=-9999.0):
    with rasterio.open(veg_path) as veg_src:
        veg = veg_src.read(1)
        meta = veg_src.meta.copy()

    with rasterio.open(ndsm_path) as ndsm_src:
        ndsm = ndsm_src.read(1)

    if veg.shape != ndsm.shape:
        raise ValueError(f"Shape mismatch: vegetation {veg.shape} vs nDSM {ndsm.shape}")

    result = np.where(
        (veg > 0) & (ndsm != nodata),
        ndsm,
        np.float32(nodata),
    ).astype(np.float32)

    meta.update(dtype="float32", nodata=nodata, count=1)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(result, 1)

    veg_pixels = int(np.sum((veg > 0) & (ndsm != nodata)))
    print(f"✓ Vegetation elevation saved: {output_path}")
    print(f"  Vegetation pixels with elevation: {veg_pixels:,}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Mask nDSM by vegetation classification — outputs elevation only where vegetation class > 0"
    )
    parser.add_argument("--veg", required=True, help="Vegetation classification raster")
    parser.add_argument("--ndsm", required=True, help="nDSM raster")
    parser.add_argument("--output", required=True, help="Output float32 raster")
    parser.add_argument(
        "--nodata", type=float, default=-9999.0, help="NoData value (default: -9999)"
    )
    args = parser.parse_args()

    try:
        apply_vegetation_elevation(args.veg, args.ndsm, args.output, args.nodata)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
