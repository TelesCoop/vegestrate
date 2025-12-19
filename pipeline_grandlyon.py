import argparse
import subprocess
import sys
import time
from pathlib import Path
from osgeo import gdal, ogr


def run_command(cmd, description):
    """Run a subprocess command and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with code {result.returncode}")
        return False

    print(f"\n✓ {description} complete")
    return True


def vectorize_raster(
    raster_path, vector_path, vector_format, use_8connected=False, field_name="DN"
):
    """Vectorize a raster to polygons using GDAL Python API."""
    print(f"\n{'=' * 70}")
    print(f"Vectorizing: {raster_path} -> {vector_path}")
    print(f"Format: {vector_format}, 8-connected: {use_8connected}")
    print(f"{'=' * 70}\n")

    src_ds = gdal.Open(str(raster_path))
    if src_ds is None:
        print(f"✗ Error: Could not open raster: {raster_path}")
        return False

    src_band = src_ds.GetRasterBand(1)

    driver_name = (
        vector_format if vector_format != "ESRI Shapefile" else "ESRI Shapefile"
    )
    drv = ogr.GetDriverByName(driver_name)
    if drv is None:
        print(f"✗ Error: Could not get driver: {driver_name}")
        return False

    if Path(vector_path).exists():
        drv.DeleteDataSource(str(vector_path))

    dst_ds = drv.CreateDataSource(str(vector_path))
    if dst_ds is None:
        print(f"✗ Error: Could not create vector file: {vector_path}")
        return False

    srs = None
    if src_ds.GetProjection():
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())

    dst_layer = dst_ds.CreateLayer("vegetation", srs=srs)
    field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
    dst_layer.CreateField(field_defn)

    options = []
    if use_8connected:
        options.append("8CONNECTED=8")

    gdal.Polygonize(src_band, None, dst_layer, 0, options, callback=gdal.TermProgress)

    dst_ds = None
    src_ds = None

    print(f"\n✓ Vectorization complete: {vector_path}")
    return True


def parse_arguments():
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete GrandLyon vegetation stratification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline phases:
  1. Update manifest (optional with --update-manifest)
  2. Data preparation: LiDAR + orthophotos (parallel)
  3. FLAIR context-aware inference
  4. Merge LiDAR + FLAIR classifications
  5. Final tile merge into single raster
  6. Vectorization to shapefile/geopackage (optional)

Examples:
  python pipeline_grandlyon.py --checkpoint model.safetensors
  python pipeline_grandlyon.py --skip-data-prep --checkpoint model.safetensors
  python pipeline_grandlyon.py --only-merge
  python pipeline_grandlyon.py --checkpoint model.safetensors --vectorize
        """,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_manifest_grandlyon.json",
        help="Path to manifest JSON (default: data/dataset_manifest_grandlyon.json)",
    )

    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update manifest from CSV before processing",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="FLAIR-HUB_LC-A_RGB_swinlarge-upernet.safetensors",
        help="Path to FLAIR checkpoint",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.8,
        help="Raster resolution in meters (default: 0.8)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=384,
        help="Tile size for FLAIR inference (default: 384)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Tile overlap for inference (default: 192, 50%%)",
    )

    parser.add_argument(
        "--grid-step",
        type=int,
        default=5,
        help="Grid step between tiles (default: 5)",
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test"],
        help="Splits to process (default: test)",
    )

    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Test-Time Augmentation",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="lyon",
        help="Output name prefix (default: lyon)",
    )

    parser.add_argument(
        "--merge-strategy",
        choices=["mode", "last"],
        default="mode",
        help="Final merge strategy (default: mode)",
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply smoothing to final raster",
    )

    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation phase",
    )

    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference phase",
    )

    parser.add_argument(
        "--skip-lidar-flair-merge",
        action="store_true",
        help="Skip LiDAR+FLAIR merge phase",
    )

    parser.add_argument(
        "--only-merge",
        action="store_true",
        help="Only run merge phases (skip data prep and inference)",
    )

    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="Vectorize final raster to vector format",
    )

    parser.add_argument(
        "--vector-format",
        choices=["ESRI Shapefile", "GPKG", "GeoJSON"],
        default="ESRI Shapefile",
        help="Vector output format (default: ESRI Shapefile)",
    )

    parser.add_argument(
        "--vector-8connected",
        action="store_true",
        help="Use 8-connectedness for polygonization (treats diagonal pixels as connected)",
    )

    return parser.parse_args()


def phase0_update_manifest(args):
    """Update manifest from CSV if requested."""
    if not args.update_manifest:
        return True
    return run_command(
        ["python", "src/data_preparation/update_manifest_grandlyon.py"],
        "PHASE 0: Update manifest from CSV",
    )


def phase1_data_preparation(args):
    """Prepare training data from LiDAR and orthophotos."""
    if args.only_merge or args.skip_data_prep:
        return True
    return run_command(
        [
            "python",
            "src/data_preparation/prepare_training_data_grandlyon.py",
            "--manifest",
            args.manifest,
            "--resolution",
            str(args.resolution),
            "--workers",
            str(args.workers),
        ],
        "PHASE 1: Data preparation (LiDAR + orthophotos)",
    )


def phase2_flair_inference(args):
    """Run FLAIR context-aware inference on orthophotos."""
    if args.only_merge or args.skip_inference:
        return True

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n✗ Error: Checkpoint not found: {checkpoint_path}")
        return False

    predictions_dir = f"predictions_{args.output_name}"

    cmd = [
        "python",
        "src/inference/inference_flair_context.py",
        "--manifest",
        args.manifest,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        predictions_dir,
        "--tile_size",
        str(args.tile_size),
        "--overlap",
        str(args.overlap),
        "--grid_step",
        str(args.grid_step),
        "--splits",
        *args.splits,
    ]

    if args.no_tta:
        cmd.append("--no_tta")

    return run_command(cmd, "PHASE 2: FLAIR context-aware inference")


def phase3_merge_lidar_flair(args):
    """Merge LiDAR and FLAIR classification maps."""
    if args.only_merge or args.skip_lidar_flair_merge:
        return True

    for split in args.splits:
        las_dir = f"data/{split}"
        flair_dir = f"predictions_{args.output_name}/{split}"
        output_dir = f"merged_classifications_{args.output_name}/{split}"

        if not run_command(
            [
                "python",
                "src/postprocessing/merge_classifications.py",
                "--las-dir",
                las_dir,
                "--flair-dir",
                flair_dir,
                "--output-dir",
                output_dir,
            ],
            f"PHASE 3: Merge LiDAR + FLAIR for {split} split",
        ):
            print(f"\n⚠ Warning: Merge failed for {split} split")

    return True


def phase4_final_merge(args):
    """Merge all tiles into final rasters."""
    for split in args.splits:
        merged_dir = f"merged_classifications_{args.output_name}/{split}"
        output_file = f"final_{args.output_name}_{split}.tif"

        if not Path(merged_dir).exists():
            print(f"\n✗ Warning: Merged directory not found: {merged_dir}")
            continue

        cmd = [
            "python",
            "src/postprocessing/merge_tifs.py",
            "--input",
            merged_dir,
            "--output",
            output_file,
            "--strategy",
            args.merge_strategy,
            "--clip-min",
            "0",
            "--clip-max",
            "3",
        ]

        if args.smooth:
            cmd.extend(
                [
                    "--smooth",
                    "--pixel-size",
                    str(args.resolution),
                ]
            )

        if not run_command(cmd, f"PHASE 4: Final merge for {split} split"):
            return False

    return True


def phase5_vectorization(args):
    """Vectorize final rasters to vector format."""
    if not args.vectorize:
        return True

    for split in args.splits:
        output_file = f"final_{args.output_name}_{split}.tif"

        if not Path(output_file).exists():
            print(f"\n✗ Warning: Final raster not found: {output_file}")
            continue

        ext_map = {"ESRI Shapefile": "shp", "GPKG": "gpkg", "GeoJSON": "geojson"}
        ext = ext_map[args.vector_format]
        vector_output = f"final_{args.output_name}_{split}.{ext}"

        if not vectorize_raster(
            output_file,
            vector_output,
            args.vector_format,
            use_8connected=args.vector_8connected,
            field_name="vegetation_class",
        ):
            print(f"\n⚠ Warning: Vectorization failed for {split} split")

    return True


def print_summary(args, elapsed):
    """Print pipeline completion summary and output locations."""
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
    print("\nOutputs:")
    print(f"  Predictions: predictions_{args.output_name}/")
    print(f"  Merged classifications: merged_classifications_{args.output_name}/")

    for split in args.splits:
        output_file = f"final_{args.output_name}_{split}.tif"
        if Path(output_file).exists():
            print(f"  Final {split} raster: {output_file}")

        if args.vectorize:
            ext_map = {"ESRI Shapefile": "shp", "GPKG": "gpkg", "GeoJSON": "geojson"}
            ext = ext_map[args.vector_format]
            vector_output = f"final_{args.output_name}_{split}.{ext}"
            if Path(vector_output).exists():
                print(f"  Final {split} vector: {vector_output}")


def main():
    """Execute the complete vegetation stratification pipeline."""
    args = parse_arguments()
    start_time = time.time()

    print("=" * 70)
    print("GRANDLYON VEGETATION STRATIFICATION PIPELINE")
    print("=" * 70)
    print(f"Manifest: {args.manifest}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution: {args.resolution}m")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Output prefix: {args.output_name}")

    manifest_path = Path(args.manifest)

    if not phase0_update_manifest(args):
        return 1

    if not manifest_path.exists():
        print(f"\n✗ Error: Manifest not found: {manifest_path}")
        print("Run with --update-manifest first")
        return 1

    if not phase1_data_preparation(args):
        return 1

    if not phase2_flair_inference(args):
        return 1

    phase3_merge_lidar_flair(args)

    if not phase4_final_merge(args):
        return 1

    phase5_vectorization(args)

    elapsed = time.time() - start_time
    print_summary(args, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
