from osgeo import gdal
import argparse
import tempfile
import os


def sieve_raster(src_ds, threshold, connectedness=4):
    print(
        f"Applying sieve filter (threshold={threshold} pixels, {connectedness}-connected)..."
    )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)

    driver = gdal.GetDriverByName("GTiff")
    sieved_ds = driver.CreateCopy(
        tmp_path,
        src_ds,
        strict=0,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
        callback=gdal.TermProgress,
    )
    sieved_ds.FlushCache()
    print()
    sieved_band = sieved_ds.GetRasterBand(1)

    gdal.SieveFilter(
        sieved_band,
        None,
        sieved_band,
        threshold,
        connectedness,
        callback=gdal.TermProgress,
    )
    sieved_ds.FlushCache()
    print()
    return sieved_ds, tmp_path


def main():
    parser = argparse.ArgumentParser(
        description="Clean the TIF using Sieve algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  https://gdal.org/en/stable/api/gdal_alg.html#_CPPv415GDALSieveFilter15GDALRasterBandH15GDALRasterBandH15GDALRasterBandHiiPPc16GDALProgressFuncPv
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input raster file (TIF)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output raster file (TIF)",
    )

    parser.add_argument(
        "--sieve",
        type=int,
        default=13,
        metavar="N",
        help="Remove regions smaller than N pixels (default: 25). 2 would remove only single pixels). Set to 0 to disable.",
    )

    args = parser.parse_args()

    if args.sieve > 0:
        print(f"Sieve threshold: {args.sieve} pixels")
    print(f"{'=' * 70}\n")

    src_ds = gdal.Open(str(args.input))
    if src_ds is None:
        print(f"Error: Could not open raster: {args.input}")
        return False

    tmp_path = None
    if args.sieve > 0:
        connectedness = 4
        result_ds, tmp_path = sieve_raster(src_ds, args.sieve, connectedness)
    else:
        result_ds = src_ds

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.CreateCopy(
        str(args.output),
        result_ds,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.FlushCache()
    out_ds = None
    result_ds = None
    src_ds = None

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
