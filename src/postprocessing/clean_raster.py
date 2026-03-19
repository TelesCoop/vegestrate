from osgeo import gdal
import numpy as np
import cv2
import argparse
import tempfile
import os
import gc
from scipy.ndimage import uniform_filter


def make_disk(radius):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return ((X**2 + Y**2) <= radius**2).astype(np.uint8)


def sieve_raster(input_path, output_path, threshold, connectedness=4):
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


def _mode_filter_block(data, k, iterations):
    n_classes = 4
    result = data.copy()
    for _ in range(iterations):
        scores = np.zeros((n_classes,) + result.shape, dtype=np.float32)
        for c in range(n_classes):
            scores[c] = uniform_filter((result == c).astype(np.float32), size=k)
        result = np.argmax(scores, axis=0).astype(np.uint8)
    return result


def mode_filter_clean(input_path, output_path, kernel=3, iterations=1, block_size=4096):
    print(f"Mode filter (kernel={kernel}x{kernel}, iterations={iterations})...")

    src_ds = gdal.Open(input_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    print(f"  Raster size: {width}x{height} pixels")

    pad = (kernel // 2) * iterations + 1

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

            rw, rh = read_x2 - read_x, read_y2 - read_y
            buf = band.ReadRaster(read_x, read_y, rw, rh, buf_type=gdal.GDT_Byte)
            data = np.frombuffer(buf, dtype=np.uint8).reshape(rh, rw).copy()
            result = _mode_filter_block(data, kernel, iterations)

            crop_x = bx - read_x
            crop_y = by - read_y
            tile = np.ascontiguousarray(
                result[crop_y : crop_y + bh, crop_x : crop_x + bw]
            )
            out_band.WriteRaster(bx, by, bw, bh, tile.tobytes(), buf_type=gdal.GDT_Byte)

            del data, result
            done += 1
            print(f"\r  Block {done}/{total}", end="", flush=True)

    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    gc.collect()
    print()


def _morph_close_block(data, k_dil, k_er):
    classes = range(1, 4)

    dil = {cls: cv2.dilate((data == cls).astype(np.uint8), k_dil) for cls in classes}
    claim_count = sum(dil[cls] for cls in classes)
    result = np.zeros_like(data)
    for cls in classes:
        result[(dil[cls] > 0) & (claim_count == 1)] = cls
    result[claim_count > 1] = data[claim_count > 1]

    for cls in classes:
        mask = (result == cls).astype(np.uint8)
        eroded = cv2.erode(mask, k_er)
        result[(mask > 0) & (eroded == 0)] = 0

    unclaimed = result == 0
    dil2 = {cls: cv2.dilate((result == cls).astype(np.uint8), k_dil) for cls in classes}
    claim_count2 = sum(dil2[cls] for cls in classes)
    for cls in classes:
        result[unclaimed & (dil2[cls] > 0) & (claim_count2 == 1)] = cls

    return result


def morphological_clean(input_path, output_path, r_dil=3, r_er=6, block_size=4096):
    print(f"Morphological close (dilate={r_dil}, erode={r_er}, dilate={r_dil})...")

    src_ds = gdal.Open(input_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    print(f"  Raster size: {width}x{height} pixels")

    k_dil = make_disk(r_dil)
    k_er = make_disk(r_er)
    pad = r_dil + r_er + 1

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

            rw, rh = read_x2 - read_x, read_y2 - read_y
            buf = band.ReadRaster(read_x, read_y, rw, rh, buf_type=gdal.GDT_Byte)
            data = np.frombuffer(buf, dtype=np.uint8).reshape(rh, rw).copy()
            result = _morph_close_block(data, k_dil, k_er)

            crop_x = bx - read_x
            crop_y = by - read_y
            tile = np.ascontiguousarray(
                result[crop_y : crop_y + bh, crop_x : crop_x + bw]
            )
            out_band.WriteRaster(bx, by, bw, bh, tile.tobytes(), buf_type=gdal.GDT_Byte)

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
        description="Clean raster: sieve then mode filter (default) or morphological close (legacy)",
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
        "--mode-kernel",
        type=int,
        default=3,
        metavar="N",
        help="Mode filter kernel size (default: 3). Set to 0 to skip.",
    )
    parser.add_argument(
        "--mode-iterations",
        type=int,
        default=1,
        metavar="N",
        help="Number of mode filter passes (default: 1).",
    )
    parser.add_argument(
        "--r-dil",
        type=int,
        default=0,
        metavar="N",
        help="[Legacy] Dilation disk radius for morphological close. Set > 0 to use instead of mode filter.",
    )
    parser.add_argument(
        "--r-er",
        type=int,
        default=0,
        metavar="N",
        help="[Legacy] Erosion disk radius for morphological close. Set > 0 to use instead of mode filter.",
    )

    args = parser.parse_args()
    use_morph = args.r_dil > 0

    print(f"{'=' * 70}")
    print(f"  Sieve threshold: {args.sieve} pixels")
    if use_morph:
        print(
            f"  [Legacy] Morph close: dilate={args.r_dil}, erode={args.r_er}, dilate={args.r_dil}"
        )
    else:
        print(
            f"  Mode filter: kernel={args.mode_kernel}x{args.mode_kernel}, iterations={args.mode_iterations}"
        )
    print(f"{'=' * 70}\n")

    tmp_path = None

    if use_morph:
        if args.sieve > 0:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
            os.close(tmp_fd)
            morphological_clean(args.input, tmp_path, args.r_dil, args.r_er)
            gc.collect()
            sieve_raster(tmp_path, args.output, args.sieve)
        else:
            morphological_clean(args.input, args.output, args.r_dil, args.r_er)
    else:
        if args.sieve > 0 and args.mode_kernel > 0:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
            os.close(tmp_fd)
            sieve_raster(args.input, tmp_path, args.sieve)
            gc.collect()
            mode_filter_clean(
                tmp_path, args.output, args.mode_kernel, args.mode_iterations
            )
        elif args.sieve > 0:
            sieve_raster(args.input, args.output, args.sieve)
        elif args.mode_kernel > 0:
            mode_filter_clean(
                args.input, args.output, args.mode_kernel, args.mode_iterations
            )
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
