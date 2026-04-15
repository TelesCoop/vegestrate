import argparse
import gc
import sys

import numpy as np
from osgeo import gdal


def apply_vegetation_elevation(
    veg_path, ndsm_path, output_path, nodata=-9999.0, block_size=4096
):
    veg_ds = gdal.Open(veg_path)
    ndsm_ds = gdal.Open(ndsm_path)

    width = veg_ds.RasterXSize
    height = veg_ds.RasterYSize

    veg_gt = veg_ds.GetGeoTransform()
    ndsm_gt = ndsm_ds.GetGeoTransform()

    if not np.isclose(veg_gt[1], ndsm_gt[1], rtol=1e-6) or not np.isclose(
        veg_gt[5], ndsm_gt[5], rtol=1e-6
    ):
        raise ValueError(
            f"Pixel size mismatch: veg {veg_gt[1]},{veg_gt[5]} vs nDSM {ndsm_gt[1]},{ndsm_gt[5]}"
        )

    ndsm_off_x = round((veg_gt[0] - ndsm_gt[0]) / ndsm_gt[1])
    ndsm_off_y = round((veg_gt[3] - ndsm_gt[3]) / ndsm_gt[5])
    ndsm_width = ndsm_ds.RasterXSize
    ndsm_height = ndsm_ds.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(veg_gt)
    out_ds.SetProjection(veg_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)

    veg_band = veg_ds.GetRasterBand(1)
    ndsm_band = ndsm_ds.GetRasterBand(1)

    ndsm_nodata = ndsm_band.GetNoDataValue()
    if ndsm_nodata is None:
        ndsm_nodata = nodata

    n_blocks_x = (width + block_size - 1) // block_size
    n_blocks_y = (height + block_size - 1) // block_size
    total = n_blocks_x * n_blocks_y
    done = 0
    veg_pixels = 0

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            bw = min(block_size, width - bx)
            bh = min(block_size, height - by)

            veg_buf = veg_band.ReadRaster(bx, by, bw, bh, buf_type=gdal.GDT_Byte)
            veg = np.frombuffer(veg_buf, dtype=np.uint8).reshape(bh, bw)

            nx = bx + ndsm_off_x
            ny = by + ndsm_off_y
            if nx < 0 or ny < 0 or nx + bw > ndsm_width or ny + bh > ndsm_height:
                ndsm = np.full((bh, bw), ndsm_nodata, dtype=np.float32)
            else:
                ndsm_buf = ndsm_band.ReadRaster(
                    nx, ny, bw, bh, buf_type=gdal.GDT_Float32
                )
                ndsm = np.frombuffer(ndsm_buf, dtype=np.float32).reshape(bh, bw).copy()

            mask = (veg > 0) & (ndsm != ndsm_nodata)
            result = np.where(mask, ndsm, np.float32(nodata))
            veg_pixels += int(np.sum(mask))

            out_band.WriteRaster(
                bx,
                by,
                bw,
                bh,
                result.astype(np.float32).tobytes(),
                buf_type=gdal.GDT_Float32,
            )

            del veg, ndsm, result
            done += 1
            print(f"\r  Block {done}/{total}", end="", flush=True)

    out_ds.FlushCache()
    out_ds = None
    veg_ds = None
    ndsm_ds = None
    gc.collect()
    print()

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
