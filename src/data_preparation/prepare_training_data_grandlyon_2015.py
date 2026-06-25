import argparse
import shutil
import time
import zipfile
from functools import partial
from io import BytesIO
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from src.core import (
    build_tile_list,
    create_classification_map,
    create_ndsm,
    download_file,
    filter_ground_vegetation,
    load_manifest,
    print_processing_summary,
    process_tiles_parallel,
    retry_session,
    setup_split_directories,
)

WCS_URL = "https://data.grandlyon.com/geoserver/grandlyon/ows"
RGB_COVERAGE = "grandlyon__ortho_2015"
IR_COVERAGE = "grandlyon__Grand_Lyon_IRC_8cm_CC46"
MAX_BLOCK_PX = 4000


def extract_tile_name(url):
    return url.split("/")[-1].split(".")[0]


def extract_pointcloud(zip_path, output_dir):
    targets = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".laz", ".las"))]
        if not members:
            raise RuntimeError(f"no point-cloud file in {zip_path.name}")
        for member in members:
            target = output_dir / Path(member).name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            targets.append(target)
    return targets


def merge_pointclouds(paths):
    clouds = [laspy.read(str(p)) for p in paths]
    if len(clouds) == 1:
        return clouds[0]
    xs = np.concatenate([c.x for c in clouds])
    ys = np.concatenate([c.y for c in clouds])
    zs = np.concatenate([c.z for c in clouds])
    cls = np.concatenate([np.asarray(c.classification) for c in clouds])
    header = clouds[0].header
    merged = laspy.LasData(
        header, laspy.PackedPointRecord.zeros(len(xs), header.point_format)
    )
    merged.x = xs
    merged.y = ys
    merged.z = zs
    merged.classification = cls
    return merged


def download_and_process_lidar(url, output_dir, resolution=0.8):
    tile_name = extract_tile_name(url)
    zip_path = output_dir / f"{tile_name}.zip"

    print(f"\n{'=' * 70}")
    print(f"Processing tile: {tile_name}")
    print(f"{'=' * 70}")

    if not zip_path.exists():
        print(f"Downloading LiDAR data from {url}...")
        download_file(url, str(zip_path))
        print(f"✓ Downloaded to {zip_path}")
    else:
        print(f"✓ LiDAR data already exists: {zip_path}")

    laz_paths = extract_pointcloud(zip_path, output_dir)

    print(f"Loading LAS data ({len(laz_paths)} sub-tiles)...")
    las = merge_pointclouds(laz_paths)
    print(f"✓ Loaded {len(las.points):,} points")

    filtered_las = filter_ground_vegetation(las, lyon=True)

    classmap_path = output_dir / f"classification_map_{tile_name}.tif"
    create_classification_map(filtered_las, las, classmap_path, resolution=resolution)

    ndsm_path = output_dir / f"ndsm_{tile_name}.tif"
    create_ndsm(las, ndsm_path, resolution=resolution)

    zip_path.unlink()
    for laz_path in laz_paths:
        laz_path.unlink()
    return classmap_path


def iter_blocks(width, height, block_px=MAX_BLOCK_PX):
    """Yield (row, col, block_height, block_width) tiles covering a width x height grid."""
    for row in range(0, height, block_px):
        for col in range(0, width, block_px):
            yield row, col, min(block_px, height - row), min(block_px, width - col)


def fetch_wcs_block(coverage_id, left, bottom, right, top, px_w, px_h):
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage_id,
        "format": "image/tiff",
        "subset": [f"X({left},{right})", f"Y({bottom},{top})"],
        "scaleSize": f"i({px_w}),j({px_h})",
    }
    resp = retry_session().get(WCS_URL, params=params, timeout=(30, 300))
    resp.raise_for_status()
    if "tif" not in resp.headers.get("Content-Type", "").lower():
        raise RuntimeError(f"WCS error for {coverage_id}: {resp.text[:300]}")
    with rasterio.open(BytesIO(resp.content)) as src:
        data = src.read()
    if not data.any():
        raise RuntimeError(
            f"WCS returned an empty block for {coverage_id} at {left},{bottom}"
        )
    return data[:3]


def fetch_wcs_for_raster(reference_raster, output_path, coverage_id, resolution=0.8):
    """Mosaic a WCS coverage onto the exact grid of a reference raster.

    The 5 km source dalles are decoded server-side (no local ECW driver), so each
    LiDAR tile is rebuilt block by block under the GeoServer pixel cap and written
    pixel-aligned to the classification map.
    """
    with rasterio.open(reference_raster) as ref:
        bounds = ref.bounds
        crs = ref.crs

    width = int(round((bounds.right - bounds.left) / resolution))
    height = int(round((bounds.top - bounds.bottom) / resolution))
    out = np.zeros((3, height, width), dtype=np.uint8)

    for row, col, bh, bw in iter_blocks(width, height):
        left = bounds.left + col * resolution
        right = left + bw * resolution
        top = bounds.top - row * resolution
        bottom = top - bh * resolution
        data = fetch_wcs_block(coverage_id, left, bottom, right, top, bw, bh)
        dh = min(bh, data.shape[1])
        dw = min(bw, data.shape[2])
        out[:, row : row + dh, col : col + dw] = data[:, :dh, :dw]

    transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
    )
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(out)

    print(f"✓ Saved {coverage_id}: {output_path}")
    return output_path


def process_tile(resolution, entry, output_dir):
    url = entry["url"]
    tile_id = entry["tile_id"]

    try:
        tile_name = extract_tile_name(url)
        classmap_path = download_and_process_lidar(url, output_dir, resolution)

        fetch_wcs_for_raster(
            classmap_path,
            output_dir / f"{tile_name}_orthophoto.tif",
            RGB_COVERAGE,
            resolution,
        )
        fetch_wcs_for_raster(
            classmap_path,
            output_dir / f"{tile_name}_ir.tif",
            IR_COVERAGE,
            resolution,
        )
        return {"tile_id": tile_id, "status": "success"}
    except Exception as e:
        print(f"\n✗ Error processing {tile_id}: {e}")
        return {"tile_id": tile_id, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 2015 GrandLyon training data (LiDAR + WCS RGB/IR)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_grandlyon_2015.json",
    )
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return

    start_time = time.time()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Error: Manifest not found: {manifest_path}")
        print("Run update_manifest_grandlyon_2015.py first to create it")
        return

    print("=" * 70)
    print("PREPARING 2015 TRAINING DATA FROM GRANDLYON LIDAR TILES (PARALLEL)")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")
    print(f"Resolution: {args.resolution}m")
    print(f"Workers: {args.workers}")
    print("=" * 70)

    manifest = load_manifest(str(manifest_path))
    output_dir = manifest_path.parent

    split_dirs = setup_split_directories(output_dir, ["train", "test"])
    all_tiles = build_tile_list(manifest, split_dirs)

    print(f"  Training: {len(manifest['train'])} tiles")
    print(f"  Testing: {len(manifest['test'])} tiles")

    process_func = partial(process_tile, args.resolution)
    successes, failures = process_tiles_parallel(
        all_tiles, process_func, max_workers=args.workers
    )

    elapsed = time.time() - start_time
    print_processing_summary(successes, failures, elapsed)


def _self_check():
    for width, height in [(15000, 15000), (4000, 4000), (4001, 999), (10001, 12345)]:
        blocks = list(iter_blocks(width, height))
        covered = sum(bh * bw for _, _, bh, bw in blocks)
        assert covered == width * height, (width, height, covered)
        for row, col, bh, bw in blocks:
            assert 0 < bh <= MAX_BLOCK_PX and 0 < bw <= MAX_BLOCK_PX
            assert row + bh <= height and col + bw <= width
    print("✓ iter_blocks self-check passed")

    def _make_cloud(x0, y0):
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.offsets = [x0, y0, 0.0]
        header.scales = [0.01, 0.01, 0.01]
        las = laspy.LasData(
            header, laspy.PackedPointRecord.zeros(3, header.point_format)
        )
        las.x = [x0, x0 + 1, x0 + 2]
        las.y = [y0, y0 + 1, y0 + 2]
        las.z = [10.0, 11.0, 12.0]
        las.classification = [2, 3, 5]
        return las

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (x0, y0) in enumerate([(1000.0, 5000.0), (2000.0, 6000.0)]):
            p = Path(tmp) / f"c{i}.laz"
            _make_cloud(x0, y0).write(str(p))
            paths.append(p)
        merged = merge_pointclouds(paths)
        assert len(merged.points) == 6, len(merged.points)
        assert np.isclose(merged.x.max(), 2002.0), merged.x.max()
        assert sorted(np.unique(merged.classification)) == [2, 3, 5]
    print("✓ merge_pointclouds self-check passed")


if __name__ == "__main__":
    main()
