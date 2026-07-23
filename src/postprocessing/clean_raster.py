from osgeo import gdal, gdal_array, osr
import numpy as np
import cv2
import argparse
import importlib.util
import shutil
import tempfile
import os
import gc
from pathlib import Path
from scipy.ndimage import uniform_filter, median_filter, distance_transform_edt


def _load_class_mappings():
    """Load class_mappings by path: importing the package would pull in torch."""
    path = Path(__file__).resolve().parents[1] / "flairhub_utils" / "class_mappings.py"
    spec = importlib.util.spec_from_file_location("class_mappings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CLASSES = _load_class_mappings()
SIMPLIFIED_COLORS = _CLASSES.SIMPLIFIED_COLORS
TRANSPARENT_CLASS = _CLASSES.TRANSPARENT_CLASS

# gdal.TermProgress is a raw C function pointer in recent bindings and is rejected
# by callback= ("Object given is not a Python function"); _nocb is the callable one.
TERM_PROGRESS = getattr(gdal, "TermProgress_nocb", None)


def make_disk(radius):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return ((X**2 + Y**2) <= radius**2).astype(np.uint8)


def class_color_table():
    ct = gdal.ColorTable()
    for value, rgba in SIMPLIFIED_COLORS.items():
        ct.SetColorEntry(int(value), tuple(int(c) for c in rgba))
    return ct


def set_class_palette(band):
    """Attach the simplified-class palette to a freshly created Byte band."""
    if band.DataType != gdal.GDT_Byte:
        return
    band.SetRasterColorTable(class_color_table())
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)


def apply_class_transparency(path):
    """
    Make the "else" class transparent in the final classification raster.

    TIFF colormaps cannot store an alpha channel, so transparency is carried by
    the nodata value — that is what QGIS and web viewers honour. Only ever call
    this on the final output: SieveFilter and Polygonize both fall back to the
    band's nodata mask, so tagging intermediates would exclude class 0 pixels
    from those steps.
    """
    ds = gdal.Open(path, gdal.GA_Update)
    if ds is None:
        print(f"  ⚠ Could not open {path} to set transparency")
        return
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(TRANSPARENT_CLASS))
    if band.GetRasterColorTable() is None:
        # Our writers set it at creation time; a plain CreateCopy of a palette-less
        # input has none, and GTiff cannot always add one after the fact.
        try:
            failed = band.SetRasterColorTable(class_color_table()) != gdal.CE_None
        except RuntimeError:
            failed = True
        if failed:
            print("  ⚠ Class palette not attached (colors only, transparency is set)")
        else:
            band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    ds.FlushCache()
    ds = None
    print(f"  Class {TRANSPARENT_CLASS} (else) set transparent via nodata")


def sieve_raster(input_path, output_path, threshold, connectedness=8):
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
        band, None, band, threshold, connectedness, callback=TERM_PROGRESS
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


def median_filter_clean(input_path, output_path, kernel=3, block_size=4096):
    print(f"Median filter (kernel={kernel}x{kernel})...")

    src_ds = gdal.Open(input_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    dtype = band.DataType
    print(f"  Raster size: {width}x{height} pixels")

    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = -9999.0
    nodata_out = nodata

    pad = kernel // 2

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        width,
        height,
        1,
        dtype,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata_out)

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

            buf = band.ReadRaster(read_x, read_y, rw, rh, buf_type=gdal.GDT_Int16)
            data = np.frombuffer(buf, dtype=np.int16).reshape(rh, rw).copy()

            valid = data != int(nodata)

            if not valid.any():
                crop_x = bx - read_x
                crop_y = by - read_y
                tile_i16 = np.full((bh, bw), int(nodata_out), dtype=np.int16)
                out_band.WriteRaster(
                    bx, by, bw, bh, tile_i16.tobytes(), buf_type=gdal.GDT_Int16
                )
                del data, valid, tile_i16
                done += 1
                print(f"\r  Block {done}/{total}", end="", flush=True)
                continue

            data_f = data.astype(np.float32)
            if not valid.all():
                _, indices = distance_transform_edt(~valid, return_indices=True)
                data_f[~valid] = data_f[indices[0][~valid], indices[1][~valid]]

            filtered = median_filter(data_f, size=kernel)

            crop_x = bx - read_x
            crop_y = by - read_y
            tile = filtered[crop_y : crop_y + bh, crop_x : crop_x + bw]
            tile_i16 = np.ascontiguousarray(np.round(tile).astype(np.int16))
            out_band.WriteRaster(
                bx, by, bw, bh, tile_i16.tobytes(), buf_type=gdal.GDT_Int16
            )

            del data, data_f, valid, filtered, tile, tile_i16
            done += 1
            print(f"\r  Block {done}/{total}", end="", flush=True)

    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    gc.collect()
    print()


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
    set_class_palette(out_band)

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
    set_class_palette(out_band)

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


def _is_vector(path):
    ds = gdal.OpenEx(path, gdal.OF_VECTOR)
    if ds is None:
        return False
    n_layers = ds.GetLayerCount()
    ds = None
    return n_layers > 0


def _usable_srs(wkt):
    """An SRS we can actually transform with (a LOCAL_CS is neither projected nor geographic)."""
    if not wkt:
        return None
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(wkt) != 0:
        return None
    return srs if (srs.IsProjected() or srs.IsGeographic()) else None


def _reproject_vector(buildings_path, target_wkt, work_dir):
    """
    Return a vector path in the target CRS.

    gdal.Rasterize burns raw layer coordinates without reprojecting, so a footprint
    file in another CRS silently produces an empty mask. Convert up front instead.
    """
    ds = gdal.OpenEx(buildings_path, gdal.OF_VECTOR)
    layer = ds.GetLayer(0)
    layer_srs = layer.GetSpatialRef()
    layer_wkt = layer_srs.ExportToWkt() if layer_srs else None
    ds = None

    source_srs = _usable_srs(layer_wkt)
    target_srs = _usable_srs(target_wkt)

    if source_srs is None or target_srs is None:
        if source_srs is not None and target_srs is None:
            print(
                "  ⚠ Raster has no usable CRS; burning footprints as raw coordinates. "
                "Tag the raster (e.g. EPSG:3946) if the mask lands in the wrong place."
            )
        return buildings_path

    if source_srs.IsSame(target_srs):
        return buildings_path

    print(
        f"  Reprojecting footprints {source_srs.GetName()} -> {target_srs.GetName()}"
    )
    reprojected = os.path.join(work_dir, "buildings_reprojected.gpkg")
    gdal.VectorTranslate(
        reprojected,
        buildings_path,
        options=gdal.VectorTranslateOptions(
            format="GPKG", dstSRS=target_wkt, reproject=True
        ),
    )
    return reprojected


def _grid_matches(ds, ref_gt, ref_proj, ref_width, ref_height):
    return (
        ds.RasterXSize == ref_width
        and ds.RasterYSize == ref_height
        and all(np.isclose(a, b) for a, b in zip(ds.GetGeoTransform(), ref_gt))
        and ds.GetProjection() == ref_proj
    )


def build_aligned_mask(buildings_path, ref_ds, output_path):
    """Burn a building source onto the reference raster grid (1 = building)."""
    width, height = ref_ds.RasterXSize, ref_ds.RasterYSize
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    creation = ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]

    if _is_vector(buildings_path):
        print(f"  Rasterizing building footprints: {buildings_path}")
        mask_ds = gdal.GetDriverByName("GTiff").Create(
            output_path, width, height, 1, gdal.GDT_Byte, options=creation
        )
        mask_ds.SetGeoTransform(gt)
        mask_ds.SetProjection(proj)
        work_dir = tempfile.mkdtemp()
        try:
            source = _reproject_vector(buildings_path, proj, work_dir)
            gdal.Rasterize(
                mask_ds,
                source,
                options=gdal.RasterizeOptions(
                    burnValues=[1], allTouched=True, callback=TERM_PROGRESS
                ),
            )
        finally:
            mask_ds.FlushCache()
            mask_ds = None
            shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"  Aligning building raster: {buildings_path}")
        minx, maxy = gt[0], gt[3]
        maxx, miny = minx + width * gt[1], maxy + height * gt[5]
        gdal.Warp(
            output_path,
            buildings_path,
            options=gdal.WarpOptions(
                format="GTiff",
                outputBounds=(minx, miny, maxx, maxy),
                width=width,
                height=height,
                dstSRS=proj if proj else None,
                outputType=gdal.GDT_Byte,
                resampleAlg="near",
                creationOptions=creation,
                callback=TERM_PROGRESS,
            ),
        )
    print()


def mask_buildings(input_path, output_path, buildings_path, fill=0, block_size=4096):
    """Set every pixel covered by a building to `fill` (None means the nodata value)."""
    print("Masking buildings...")

    src_ds = gdal.Open(input_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    dtype = band.DataType
    np_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(dtype)
    print(f"  Raster size: {width}x{height} pixels")

    nodata = band.GetNoDataValue()
    if fill is None:
        fill = nodata if nodata is not None else -9999.0
    fill_value = np.array(fill).astype(np_dtype)
    print(f"  Building pixels set to {fill_value}")

    tmp_mask = None
    mask_ds = gdal.Open(buildings_path) if not _is_vector(buildings_path) else None
    if mask_ds is not None and _grid_matches(
        mask_ds, src_ds.GetGeoTransform(), src_ds.GetProjection(), width, height
    ):
        print(f"  Building raster already on the target grid: {buildings_path}")
    else:
        mask_ds = None
        tmp_fd, tmp_mask = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)
        build_aligned_mask(buildings_path, src_ds, tmp_mask)
        mask_ds = gdal.Open(tmp_mask)
    mask_band = mask_ds.GetRasterBand(1)
    mask_nodata = mask_band.GetNoDataValue()

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        width,
        height,
        1,
        dtype,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    set_class_palette(out_band)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    elif fill_value != 0:
        out_band.SetNoDataValue(float(fill_value))

    n_blocks_x = (width + block_size - 1) // block_size
    n_blocks_y = (height + block_size - 1) // block_size
    total = n_blocks_x * n_blocks_y
    done = 0
    masked = 0

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            bw = min(block_size, width - bx)
            bh = min(block_size, height - by)

            buf = band.ReadRaster(bx, by, bw, bh, buf_type=dtype)
            data = np.frombuffer(buf, dtype=np_dtype).reshape(bh, bw).copy()

            mask_buf = mask_band.ReadRaster(bx, by, bw, bh, buf_type=gdal.GDT_Byte)
            mask = np.frombuffer(mask_buf, dtype=np.uint8).reshape(bh, bw)

            hit = mask > 0
            if mask_nodata is not None:
                # A mask carrying its own nodata must not mask those pixels.
                hit &= mask != np.uint8(mask_nodata)
            data[hit] = fill_value
            masked += int(np.count_nonzero(hit))

            out_band.WriteRaster(
                bx, by, bw, bh, np.ascontiguousarray(data).tobytes(), buf_type=dtype
            )

            del data, mask, hit
            done += 1
            print(f"\r  Block {done}/{total}", end="", flush=True)

    out_ds.FlushCache()
    out_ds = None
    mask_ds = None
    src_ds = None
    gc.collect()
    print()
    print(f"  Masked {masked:,} pixels ({100 * masked / (width * height):.2f}%)")

    if tmp_mask and os.path.exists(tmp_mask):
        os.remove(tmp_mask)


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
        "--median-kernel",
        type=int,
        default=0,
        metavar="N",
        help="Median filter kernel size for continuous rasters (e.g. nDSM). Set > 0 to use. Nodata-aware.",
    )
    parser.add_argument(
        "--no-transparent-else",
        dest="transparent_else",
        action="store_false",
        help="Keep class 0 opaque. By default the output tags class 0 (else) as nodata "
        "so it renders transparent, and carries the simplified-class palette.",
    )
    parser.add_argument(
        "--buildings",
        type=str,
        default=None,
        metavar="PATH",
        help="Building footprints to mask out, applied last. Raster (nonzero = building) "
        "or vector (SHP/GPKG/GeoJSON); reprojected/resampled onto the input grid.",
    )
    parser.add_argument(
        "--buildings-value",
        type=float,
        default=None,
        metavar="V",
        help="Value written under buildings (default: 0 for class rasters, nodata for --median-kernel).",
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
    use_median = args.median_kernel > 0

    print(f"{'=' * 70}")
    if use_median:
        print(
            f"  Median filter: kernel={args.median_kernel}x{args.median_kernel} (nodata-aware)"
        )
    elif use_morph:
        print(
            f"  [Legacy] Morph close: dilate={args.r_dil}, erode={args.r_er}, dilate={args.r_dil}"
        )
        print(f"  Sieve threshold: {args.sieve} pixels")
    else:
        print(
            f"  Mode filter: kernel={args.mode_kernel}x{args.mode_kernel}, iterations={args.mode_iterations}"
        )
        print(f"  Sieve threshold: {args.sieve} pixels")
    if args.buildings:
        print(f"  Building mask: {args.buildings}")
    if args.transparent_else and not use_median:
        print(f"  Class {TRANSPARENT_CLASS} (else): transparent + class palette")
    print(f"{'=' * 70}\n")

    tmp_paths = []

    def new_temp():
        tmp_fd, tmp = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)
        tmp_paths.append(tmp)
        return tmp

    # Buildings are masked out last, so the filters run on the untouched raster.
    clean_output = new_temp() if args.buildings else args.output

    try:
        if use_median:
            median_filter_clean(args.input, clean_output, args.median_kernel)
        elif use_morph:
            if args.sieve > 0:
                tmp_path = new_temp()
                morphological_clean(args.input, tmp_path, args.r_dil, args.r_er)
                gc.collect()
                sieve_raster(tmp_path, clean_output, args.sieve)
            else:
                morphological_clean(args.input, clean_output, args.r_dil, args.r_er)
        else:
            if args.sieve > 0 and args.mode_kernel > 0:
                tmp_path = new_temp()
                mode_filter_clean(
                    args.input, tmp_path, args.mode_kernel, args.mode_iterations
                )
                gc.collect()
                sieve_raster(tmp_path, clean_output, args.sieve)
            elif args.sieve > 0:
                sieve_raster(args.input, clean_output, args.sieve)
            elif args.mode_kernel > 0:
                mode_filter_clean(
                    args.input, clean_output, args.mode_kernel, args.mode_iterations
                )
            else:
                src_ds = gdal.Open(args.input)
                driver = gdal.GetDriverByName("GTiff")
                driver.CreateCopy(
                    clean_output,
                    src_ds,
                    options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
                )
                src_ds = None

        if args.buildings:
            gc.collect()
            fill = args.buildings_value
            if fill is None:
                fill = None if use_median else 0
            mask_buildings(clean_output, args.output, args.buildings, fill)

        # Continuous rasters (--median-kernel) keep their own nodata, no classes.
        if args.transparent_else and not use_median:
            apply_class_transparency(args.output)
    finally:
        # These are full-size copies of the raster; never leave them behind on a crash.
        for tmp in tmp_paths:
            if os.path.exists(tmp):
                os.remove(tmp)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
