from osgeo import gdal
import numpy as np
import cv2
import argparse
import tempfile
import os
import gc

CLASS_PRIORITY = [1, 2, 3]


def make_disk(radius):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return ((X**2 + Y**2) <= radius**2).astype(np.uint8)


def sieve_raster(input_path, output_path, threshold, connectedness=4):
    """Sieve filtering from GDQL : https://gdal.org/en/stable/programs/gdal_sieve.html"""
    print(f"Sieve filter (threshold={threshold} pixels, {connectedness}-connected)...")

    src_ds = gdal.Open(input_path)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.CreateCopy(
        output_path,
        src_ds,
        strict=0,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.FlushCache()
    src_ds = None

    band = out_ds.GetRasterBand(1)
    gdal.SieveFilter(
        band, None, band, threshold, connectedness, callback=gdal.TermProgress
    )
    out_ds.FlushCache()
    out_ds = None
    print()


def _morph_close_block(data, kernel):
    result = np.zeros_like(data)

    for cls in CLASS_PRIORITY:
        mask = (data == cls).view(np.uint8)
        dilated = cv2.dilate(mask, kernel)
        result[dilated > 0] = cls

    for cls in CLASS_PRIORITY:
        mask = (result == cls).view(np.uint8)
        eroded = cv2.erode(mask, kernel)
        result[(mask > 0) & (eroded == 0)] = 0

    for cls in CLASS_PRIORITY:
        unclaimed = result == 0
        neighbors = cv2.dilate((result == cls).view(np.uint8), kernel)
        result[unclaimed & (neighbors > 0)] = cls

    return result


def morphological_clean(input_path, output_path, radius=5, block_size=4096):
    print(f"Morphological close (disk radius={radius} pixels)...")

    src_ds = gdal.Open(input_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    print(f"  Raster size: {width}x{height} pixels")

    kernel = make_disk(radius)
    pad = radius + 1

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)

    n_blocks_x = (width + block_size - 1) // block_size
    n_blocks_y = (height + block_size - 1) // block_size
    total = n_blocks_x * n_blocks_y
    done = 0

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            bw = min(block_size, width - bx)
            bh = min(block_size, height - by)

            read_x = max(bx - pad, 0)
            read_y = max(by - pad, 0)
            read_x2 = min(bx + bw + pad, width)
            read_y2 = min(by + bh + pad, height)

            data = band.ReadAsArray(read_x, read_y, read_x2 - read_x, read_y2 - read_y)
            result = _morph_close_block(data, kernel)

            crop_x = bx - read_x
            crop_y = by - read_y
            out_band.WriteArray(
                result[crop_y : crop_y + bh, crop_x : crop_x + bw], bx, by
            )

            del data, result
            done += 1
            print(f"\r  Block {done}/{total}", end="", flush=True)

    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    gc.collect()
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clean raster: sieve then morphological close",
    )
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "--sieve",
        type=int,
        default=13,
        metavar="N",
        help="Remove regions smaller than N pixels (default: 13). Set to 0 to skip.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        metavar="N",
        help="Disk radius in pixels for morphological closing (default: 5). Set to 0 to skip.",
    )

    args = parser.parse_args()
    print(f"{'=' * 70}")
    print(f"  Sieve threshold: {args.sieve} pixels")
    print(f"  Morph close radius: {args.radius} pixels")
    print(f"{'=' * 70}\n")

    tmp_path = None

    if args.sieve > 0 and args.radius > 0:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)
        morphological_clean(args.input, tmp_path, args.radius)
        gc.collect()
        sieve_raster(tmp_path, args.output, args.sieve)

    elif args.radius > 0:
        morphological_clean(args.input, args.output, args.radius)

    elif args.sieve > 0:
        sieve_raster(args.input, args.output, args.sieve)

    else:
        src_ds = gdal.Open(args.input)
        driver = gdal.GetDriverByName("GTiff")
        driver.CreateCopy(
            args.output,
            src_ds,
            options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
        )
        src_ds = None

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
