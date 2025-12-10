import argparse
import glob
import os
import sys
from typing import List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import shape
from smoothify import smoothify


class TileMerger:
    """Sequential tile merger to regroupe all the tiles in a single one."""

    def __init__(
        self,
        merge_strategy: str = "mode",
        nodata: Optional[float] = None,
        clip_values: Optional[tuple] = None,
    ):
        """
        Initialize the tile merger.

        Args:
            merge_strategy: Strategy for overlapping pixels ('mode', 'last')
            nodata: NoData value to use (None = auto-detect from first file)
            clip_values: Tuple (min, max) to clip output values (e.g., (0, 3))
        """
        valid_strategies = ["mode", "last"]
        if merge_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid merge strategy: {merge_strategy}. Must be one of {valid_strategies}"
            )
        self.merge_strategy = merge_strategy
        self.nodata = nodata
        self.clip_values = clip_values
        self.tiles_info = []

    def scan_tiles(self, file_paths: List[str]) -> dict:
        """
        Scan all tiles to get metadata and calculate mosaic bounds.

        Args:
            file_paths: List of paths to TIF files

        Returns:
            Dictionary with mosaic metadata
        """
        if not file_paths:
            raise ValueError("No input files found")

        print(f"Scanning {len(file_paths)} tiles...")

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        ref_crs = None
        ref_res = None
        ref_dtype = None
        ref_count = None

        for i, fpath in enumerate(file_paths, 1):
            with rasterio.open(fpath) as src:
                bounds = src.bounds
                self.tiles_info.append(
                    {
                        "path": fpath,
                        "bounds": bounds,
                        "transform": src.transform,
                        "shape": src.shape,
                        "window": None,  # Will be calculated later
                    }
                )

                min_x = min(min_x, bounds.left)
                min_y = min(min_y, bounds.bottom)
                max_x = max(max_x, bounds.right)
                max_y = max(max_y, bounds.top)

                if ref_crs is None:
                    ref_crs = src.crs
                    ref_res = (src.transform.a, src.transform.e)
                    ref_dtype = src.dtypes[0]
                    ref_count = src.count
                    if self.nodata is None:
                        self.nodata = src.nodata
                    print(
                        f"  Reference CRS: {ref_crs} (EPSG:{ref_crs.to_epsg() if ref_crs.to_epsg() else 'custom'})"
                    )
                else:
                    if src.crs != ref_crs:
                        raise ValueError(
                            f"CRS mismatch: {fpath} has {src.crs} (EPSG:{src.crs.to_epsg()}), "
                            f"expected {ref_crs} (EPSG:{ref_crs.to_epsg()})"
                        )
                    tile_res = (src.transform.a, src.transform.e)
                    if (
                        abs(tile_res[0] - ref_res[0]) > 1e-4
                        or abs(tile_res[1] - ref_res[1]) > 1e-4
                    ):
                        raise ValueError(
                            f"Resolution mismatch: {fpath} has {tile_res}, expected {ref_res}"
                        )
                    if src.dtypes[0] != ref_dtype:
                        print(
                            f"Warning: dtype mismatch in {fpath}: {src.dtypes[0]} vs {ref_dtype}"
                        )
                    if src.count != ref_count:
                        raise ValueError(
                            f"Band count mismatch: {fpath} has {src.count}, expected {ref_count}"
                        )

            if i % 10 == 0:
                print(f"  Scanned {i}/{len(file_paths)} tiles...")

        width = int(np.round((max_x - min_x) / ref_res[0]))
        height = int(np.round((max_y - min_y) / abs(ref_res[1])))

        transform = Affine(ref_res[0], 0.0, min_x, 0.0, ref_res[1], max_y)

        for tile in self.tiles_info:
            col_off = int(np.round((tile["bounds"].left - min_x) / ref_res[0]))
            row_off = int(np.round((max_y - tile["bounds"].top) / abs(ref_res[1])))

            tile["window"] = Window(
                col_off, row_off, tile["shape"][1], tile["shape"][0]
            )

        mosaic_meta = {
            "driver": "GTiff",
            "dtype": ref_dtype,
            "count": ref_count,
            "crs": ref_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": self.nodata,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        print(f"\nMosaic dimensions: {width} x {height} pixels")
        print(f"Mosaic bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print(f"CRS: {ref_crs}")
        print(f"Resolution: {ref_res}")
        print(f"Data type: {ref_dtype}")

        return mosaic_meta

    def _merge_last_strategy(self, output_path: str, mosaic_meta: dict):
        """Merge using 'last' strategy: last tile wins."""
        with rasterio.open(output_path, "w", **mosaic_meta) as dst:
            if self.nodata is not None:
                for band_idx in range(1, mosaic_meta["count"] + 1):
                    chunk_size = 1000
                    nodata_array = np.full(
                        (chunk_size, mosaic_meta["width"]),
                        self.nodata,
                        dtype=mosaic_meta["dtype"],
                    )
                    for row in range(0, mosaic_meta["height"], chunk_size):
                        height = min(chunk_size, mosaic_meta["height"] - row)
                        if height < chunk_size:
                            nodata_array = nodata_array[:height, :]
                        window = Window(0, row, mosaic_meta["width"], height)
                        dst.write(nodata_array, band_idx, window=window)

            for i, tile in enumerate(self.tiles_info, 1):
                with rasterio.open(tile["path"]) as src:
                    data = src.read()
                    if self.clip_values is not None:
                        data = np.clip(data, self.clip_values[0], self.clip_values[1])
                    dst.write(data, window=tile["window"])
                if i % 5 == 0:
                    print(f"  Merged {i}/{len(self.tiles_info)} tiles...")

    def _merge_mode_strategy(self, output_path: str, mosaic_meta: dict):
        """Merge using 'mode' strategy: most frequent class wins."""
        print("  Using mode strategy - collecting class counts from all tiles...")

        unique_classes = set()
        for tile in self.tiles_info:
            with rasterio.open(tile["path"]) as src:
                tile_classes = np.unique(src.read(1))
                unique_classes.update(tile_classes.tolist())

        if self.nodata is not None and self.nodata in unique_classes:
            unique_classes.remove(self.nodata)

        unique_classes = sorted(unique_classes)
        print(f"  Detected classes: {unique_classes}")

        counts = {
            class_val: np.zeros(
                (mosaic_meta["height"], mosaic_meta["width"]), dtype=np.uint8
            )
            for class_val in unique_classes
        }

        print("  Pass 1: Accumulating class counts...")
        for i, tile in enumerate(self.tiles_info, 1):
            with rasterio.open(tile["path"]) as src:
                data = src.read(1)
                if self.clip_values is not None:
                    data = np.clip(data, self.clip_values[0], self.clip_values[1])

                win = tile["window"]
                row_start = win.row_off
                row_end = row_start + win.height
                col_start = win.col_off
                col_end = col_start + win.width

                for class_val in unique_classes:
                    mask = data == class_val
                    if self.nodata is not None:
                        mask = mask & (data != self.nodata)
                    counts[class_val][row_start:row_end, col_start:col_end] += mask

            if i % 5 == 0:
                print(f"    Processed {i}/{len(self.tiles_info)} tiles...")

        print("  Pass 2: Computing mode for each pixel...")

        count_stack = np.stack([counts[c] for c in unique_classes], axis=2)
        max_indices = np.argmax(count_stack, axis=2)

        class_mapping = np.array(unique_classes, dtype=mosaic_meta["dtype"])
        result = class_mapping[max_indices]

        if self.nodata is not None:
            total_counts = count_stack.sum(axis=2)
            nodata_mask = total_counts == 0
            result[nodata_mask] = self.nodata

        print("  Writing output...")
        with rasterio.open(output_path, "w", **mosaic_meta) as dst:
            dst.write(result, 1)

    def merge_tiles(self, output_path: str, mosaic_meta: dict):
        """
        Merge tiles into output file.

        Args:
            output_path: Path for output merged TIF
            mosaic_meta: Metadata dictionary from scan_tiles()
        """
        print(f"\nMerging {len(self.tiles_info)} tiles into {output_path}...")

        if self.merge_strategy == "last":
            self._merge_last_strategy(output_path, mosaic_meta)
        elif self.merge_strategy == "mode":
            self._merge_mode_strategy(output_path, mosaic_meta)

        print(f"\nMerge complete! Output saved to: {output_path}")

        file_size = os.path.getsize(output_path)
        if file_size > 1024**3:
            print(f"Output file size: {file_size / 1024**3:.2f} GB")
        elif file_size > 1024**2:
            print(f"Output file size: {file_size / 1024**2:.2f} MB")
        else:
            print(f"Output file size: {file_size / 1024:.2f} KB")

    def vectorize_and_smooth(
        self,
        raster_path: str,
        output_vector_path: str,
        pixel_size: float,
        smooth_iterations: int = 3,
        num_cores: int = 0,
    ):
        """
        Vectorize merged raster and apply smoothing to reduce tile boundary artifacts.

        Args:
            raster_path: Path to merged raster TIF
            output_vector_path: Path for smoothed vector output (GeoPackage)
            pixel_size: Raster resolution in map units (for segment_length)
            smooth_iterations: Chaikin smoothing iterations (default: 3)
            num_cores: CPU cores for parallel processing (0 = all available)
        """
        print(f"\nVectorizing raster: {raster_path}")

        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

            print(f"  Raster shape: {raster_data.shape}")
            print(f"  CRS: {crs}")
            print(f"  NoData: {nodata}")

            mask = None
            if nodata is not None:
                mask = raster_data != nodata

            print("  Converting raster to polygons...")
            geoms = []
            values = []

            for geom, value in shapes(raster_data, mask=mask, transform=transform):
                geoms.append(shape(geom))
                values.append(value)

            print(f"  Generated {len(geoms)} polygons")

        gdf = gpd.GeoDataFrame({"class": values}, geometry=geoms, crs=crs)

        print(f"\nApplying Smoothify (iterations={smooth_iterations})...")
        print(f"  Segment length (pixel size): {pixel_size} map units")

        smoothed_gdf = smoothify(
            geom=gdf,
            segment_length=pixel_size,
            smooth_iterations=smooth_iterations,
            num_cores=num_cores,
            merge_collection=True,
            preserve_area=True,
            area_tolerance=0.01,
        )

        print(f"\nSaving smoothed polygons to: {output_vector_path}")
        smoothed_gdf.to_file(output_vector_path, driver="GPKG")

        file_size = os.path.getsize(output_vector_path)
        if file_size > 1024**2:
            print(f"Output file size: {file_size / 1024**2:.2f} MB")
        else:
            print(f"Output file size: {file_size / 1024:.2f} KB")

        print("\n✓ Smoothing complete!")
        print(f"  Input polygons: {len(gdf)}")
        print(f"  Output polygons: {len(smoothed_gdf)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple GeoTIFF files sequentially (memory-efficient)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        default="merged_classifications/test",
        help="Path input TIF files (default: merged_classifications/test)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="merged_output.tif",
        help="Output file path (default: merged_output.tif)",
    )

    parser.add_argument(
        "--strategy",
        "-s",
        choices=["mode", "last"],
        default="mode",
        help="Merge strategy for overlapping pixels (default: mode)",
    )

    parser.add_argument(
        "--nodata",
        "-n",
        type=float,
        default=None,
        help="NoData value (default: auto-detect from first file)",
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply smoothing to reduce tile boundary artifacts (vectorize + smooth)",
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.8,
        help="Raster pixel size in map units (default: 0.8 for Grand Lyon data)",
    )

    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=3,
        help="Smoothing iterations (3-5 typical, default: 3)",
    )

    parser.add_argument(
        "--smooth-cores",
        type=int,
        default=3,
        help="CPU cores for smoothing (0 = all available, default: 0)",
    )

    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Minimum value to clip output (e.g., 0 for classes {0,1,2,3})",
    )

    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Maximum value to clip output (e.g., 3 for classes {0,1,2,3})",
    )

    args = parser.parse_args()

    file_paths = sorted(glob.glob(args.input + "/*.tif"))

    if not file_paths:
        print("Error: No files found matching pattern")
        sys.exit(1)

    print(f"Found {len(file_paths)} files to merge")
    print(f"Input path: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Merge strategy: {args.strategy}")

    clip_values = None
    if args.clip_min is not None and args.clip_max is not None:
        clip_values = (args.clip_min, args.clip_max)
        print(f"Clipping output values to [{args.clip_min}, {args.clip_max}]")

    if os.path.exists(args.output):
        response = input(f"\nWarning: {args.output} already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    merger = TileMerger(
        merge_strategy=args.strategy, nodata=args.nodata, clip_values=clip_values
    )

    try:
        mosaic_meta = merger.scan_tiles(file_paths)
        merger.merge_tiles(args.output, mosaic_meta)

        if args.smooth:
            output_base = os.path.splitext(args.output)[0]
            vector_output = f"{output_base}_smoothed.gpkg"

            print("\n" + "=" * 70)
            print("SMOOTHING MERGED RASTER")
            print("=" * 70)

            merger.vectorize_and_smooth(
                raster_path=args.output,
                output_vector_path=vector_output,
                pixel_size=args.pixel_size,
                smooth_iterations=args.smooth_iterations,
                num_cores=args.smooth_cores,
            )

    except Exception as e:
        print(f"\nError during merge: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
