import argparse
import gc
import sys

import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi


def apply_vegetation_elevation(
    veg_path, ndsm_path, output_path, nodata=-9999, block_size=4096
):
    veg_ds = gdal.Open(veg_path)
    ndsm_ds = gdal.Open(ndsm_path)

    width = veg_ds.RasterXSize
    height = veg_ds.RasterYSize

    veg_gt = veg_ds.GetGeoTransform()
    ndsm_gt = ndsm_ds.GetGeoTransform()

    if not np.isclose(veg_gt[1], ndsm_gt[1], rtol=1e-3) or not np.isclose(
        veg_gt[5], ndsm_gt[5], rtol=1e-3
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
        gdal.GDT_Int16,
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
                ndsm = np.full((bh, bw), ndsm_nodata, dtype=np.int16)
            else:
                ndsm_buf = ndsm_band.ReadRaster(nx, ny, bw, bh, buf_type=gdal.GDT_Int16)
                ndsm = np.frombuffer(ndsm_buf, dtype=np.int16).reshape(bh, bw).copy()

            mask = (veg > 0) & (ndsm != ndsm_nodata)
            result = np.where(mask, ndsm, np.int16(nodata)).astype(np.int16)
            veg_pixels += int(np.sum(mask))

            out_band.WriteRaster(
                bx,
                by,
                bw,
                bh,
                result.tobytes(),
                buf_type=gdal.GDT_Int16,
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


def compute_zone_elevation(veg_path, ndsm_path, output_path, nodata=-9999):
    """
    For each connected zone (same-class pixels touching each other), assign the
    median nDSM height of all valid pixels in that zone. Intended for per-tile
    use where the full tile fits comfortably in memory.
    """
    veg_ds = gdal.Open(veg_path)
    ndsm_ds = gdal.Open(ndsm_path)

    width = veg_ds.RasterXSize
    height = veg_ds.RasterYSize
    veg_gt = veg_ds.GetGeoTransform()
    ndsm_gt = ndsm_ds.GetGeoTransform()

    ndsm_off_x = round((veg_gt[0] - ndsm_gt[0]) / ndsm_gt[1])
    ndsm_off_y = round((veg_gt[3] - ndsm_gt[3]) / ndsm_gt[5])
    ndsm_w = ndsm_ds.RasterXSize
    ndsm_h = ndsm_ds.RasterYSize

    ndsm_band = ndsm_ds.GetRasterBand(1)
    ndsm_nodata = ndsm_band.GetNoDataValue()
    if ndsm_nodata is None:
        ndsm_nodata = nodata

    veg = veg_ds.GetRasterBand(1).ReadAsArray()

    dst_x = max(0, ndsm_off_x)
    dst_y = max(0, ndsm_off_y)
    dst_x2 = min(width, ndsm_off_x + ndsm_w)
    dst_y2 = min(height, ndsm_off_y + ndsm_h)
    src_x = max(0, -ndsm_off_x)
    src_y = max(0, -ndsm_off_y)

    ndsm = np.full((height, width), ndsm_nodata, dtype=np.int16)
    if dst_x2 > dst_x and dst_y2 > dst_y:
        chunk = ndsm_band.ReadAsArray(src_x, src_y, dst_x2 - dst_x, dst_y2 - dst_y)
        ndsm[dst_y:dst_y2, dst_x:dst_x2] = chunk
        del chunk

    valid_ndsm = ndsm != ndsm_nodata
    zone_out = np.full((height, width), nodata, dtype=np.int16)

    for cls in range(1, 4):
        cls_mask = veg == cls
        if not np.any(cls_mask):
            continue

        labeled, n = ndi.label(cls_mask)
        print(f"    class {cls}: {n:,} zones")

        labels_valid = labeled[cls_mask & valid_ndsm]
        heights_valid = ndsm[cls_mask & valid_ndsm]

        if len(labels_valid) > 0:
            sort_idx = np.argsort(labels_valid, kind="stable")
            labels_sorted = labels_valid[sort_idx]
            heights_sorted = heights_valid[sort_idx]

            unique_labels, first_idx = np.unique(labels_sorted, return_index=True)
            last_idx = np.append(first_idx[1:], len(labels_sorted))

            median_lookup = np.full(n + 1, nodata, dtype=np.int16)
            for i in range(len(unique_labels)):
                h = heights_sorted[first_idx[i] : last_idx[i]]
                median_lookup[unique_labels[i]] = int(np.median(h))

            zone_out[labeled > 0] = median_lookup[labeled[labeled > 0]]

        del labeled
        gc.collect()

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        width,
        height,
        1,
        gdal.GDT_Int16,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(veg_gt)
    out_ds.SetProjection(veg_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    out_band.WriteArray(zone_out)
    out_ds.FlushCache()
    out_ds = None
    veg_ds = None
    ndsm_ds = None
    del veg, ndsm, zone_out
    gc.collect()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Mask nDSM by vegetation classification — outputs elevation only where vegetation class > 0"
    )
    parser.add_argument("--veg", required=True, help="Vegetation classification raster")
    parser.add_argument("--ndsm", required=True, help="nDSM raster")
    parser.add_argument(
        "--output", required=True, help="Output per-pixel elevation raster"
    )
    parser.add_argument(
        "--nodata", type=int, default=-9999, help="NoData value (default: -9999)"
    )
    args = parser.parse_args()

    try:
        apply_vegetation_elevation(args.veg, args.ndsm, args.output, args.nodata)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
