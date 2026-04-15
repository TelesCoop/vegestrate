import argparse
import os
import subprocess
import sys

import rasterio


def snap_to_grid(input_path, cell_size=0.2):
    with rasterio.open(input_path) as src:
        t = src.transform
        origin_x = t.c
        origin_y = t.f
        old_px = t.a
        old_py = abs(t.e)
        width = src.width
        height = src.height
        compression = src.compression.name.lower() if src.compression else "deflate"

    if abs(old_px - cell_size) < 1e-9 and abs(old_py - cell_size) < 1e-9:
        print(f"Already at {cell_size}m grid, skipping: {input_path}")
        return

    new_width = round(old_px * width / cell_size)
    new_height = round(old_py * height / cell_size)
    xmin = origin_x
    ymax = origin_y
    xmax = xmin + cell_size * new_width
    ymin = ymax - cell_size * new_height

    print(f"Snapping {os.path.basename(input_path)}")
    print(f"  pixel: ({old_px:.10f}, {-old_py:.10f}) → ({cell_size}, {-cell_size})")
    print(f"  size:  {width}x{height} → {new_width}x{new_height}")

    tmp_path = input_path + ".tmp.tif"
    try:
        subprocess.run(
            [
                "gdalwarp",
                "-r",
                "near",
                "-tr",
                str(cell_size),
                str(cell_size),
                "-te",
                str(xmin),
                str(ymin),
                str(xmax),
                str(ymax),
                "-co",
                f"COMPRESS={compression.upper()}",
                "-co",
                "TILED=YES",
                "-co",
                "BIGTIFF=YES",
                input_path,
                tmp_path,
            ],
            check=True,
        )
        os.replace(tmp_path, input_path)
        print("  Done.")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Snap TIF pixel size to an exact grid by resampling with gdalwarp"
    )
    parser.add_argument("files", nargs="+", help="Input TIF file(s) to fix in-place")
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.2,
        help="Target cell size in CRS units (default: 0.2)",
    )
    args = parser.parse_args()

    for f in args.files:
        try:
            snap_to_grid(f, args.cell_size)
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
