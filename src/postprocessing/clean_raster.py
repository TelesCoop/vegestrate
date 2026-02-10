from osgeo import gdal
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import argparse
import tempfile
import os
import gc

CLASS_PRIORITY = [1, 2, 3]


def make_disk(radius):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return (X**2 + Y**2) <= radius**2


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


def morphological_clean(input_path, output_path, radius=5):
    """Closing: dilation, fusion, erosion. Remove small area and smooth geometries."""
    print(f"Morphological close (disk radius={radius} pixels)...")

    src_ds = gdal.Open(input_path)
    data = src_ds.GetRasterBand(1).ReadAsArray()
    print(f"  Raster size: {data.shape[1]}x{data.shape[0]} pixels")

    struct = make_disk(radius)
    result = np.zeros_like(data)

    for cls in CLASS_PRIORITY:
        mask = data == cls
        dilated = binary_dilation(mask, structure=struct)
        result[dilated] = cls

    del data
    gc.collect()

    for cls in CLASS_PRIORITY:
        mask = result == cls
        eroded = binary_erosion(mask, structure=struct)
        result[mask & ~eroded] = 0

    for cls in CLASS_PRIORITY:
        unclaimed = result == 0
        neighbors = binary_dilation(result == cls, structure=struct)
        result[unclaimed & neighbors] = cls

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(result)
    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    del result
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

        sieve_raster(args.input, tmp_path, args.sieve)
        gc.collect()
        morphological_clean(tmp_path, args.output, args.radius)

    elif args.sieve > 0:
        sieve_raster(args.input, args.output, args.sieve)

    elif args.radius > 0:
        morphological_clean(args.input, args.output, args.radius)

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
