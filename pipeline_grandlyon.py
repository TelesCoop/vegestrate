import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


class StateManager:
    def __init__(self, state_path: Path, cfg_hash: str):
        self.state_path = state_path
        self.cfg_hash = cfg_hash
        if state_path.exists():
            with open(state_path) as f:
                self.state = json.load(f)
            if self.state.get("config_hash") != cfg_hash:
                print(
                    "⚠ Config changed since last run — some phases may be stale. Use --force all to rerun everything."
                )
            for info in self.state["phases"].values():
                if info["status"] == "running":
                    info["status"] = "failed"
                    info["error"] = "interrupted"
            self._save()
        else:
            self.state = {
                "config_hash": cfg_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phases": {},
            }
            self._save()

    def _save(self):
        self.state["config_hash"] = self.cfg_hash
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def should_run(self, phase: str, forced: set) -> bool:
        if "all" in forced or phase in forced:
            return True
        status = self.state["phases"].get(phase, {}).get("status")
        return status != "success"

    def begin(self, phase: str):
        self.state["phases"][phase] = {
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def end(self, phase: str, success: bool, error: str | None = None):
        info = self.state["phases"].setdefault(phase, {})
        start = info.get("start_time")
        end_time = datetime.now(timezone.utc).isoformat()
        info["status"] = "success" if success else "failed"
        info["end_time"] = end_time
        if start:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end_time)
            info["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 1)
        if error:
            info["error"] = error
        self._save()

    def skip(self, phase: str, reason: str):
        self.state["phases"][phase] = {"status": "skipped", "reason": reason}
        self._save()


def run_module_main(module_path: str, args: list[str], description: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Module: {module_path}")
    print(f"Args: {' '.join(args)}\n")

    try:
        import importlib

        module = importlib.import_module(module_path)
        old_argv = sys.argv.copy()
        sys.argv = [module_path] + args
        try:
            result = module.main()
            success = result == 0 if result is not None else True
        finally:
            sys.argv = old_argv

        if success:
            print(f"\n✓ {description} complete")
        else:
            print(f"\n✗ Error: {description} failed")
        return success

    except Exception as e:
        print(f"\n✗ Error: {description} failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def phase_data_preparation(config: dict) -> bool:
    cmd_args = [
        "--manifest",
        config["data"]["manifest"],
        "--resolution",
        str(config["data"]["resolution"]),
        "--workers",
        str(config["data"]["workers"]),
    ]
    if config["data"].get("ir_mosaic"):
        cmd_args.extend(["--ir_mosaic", config["data"]["ir_mosaic"]])
    if config["data"].get("download_ir", False):
        cmd_args.append("--download_ir")
    return run_module_main(
        "src.data_preparation.prepare_training_data_grandlyon",
        cmd_args,
        "PHASE 1: Data preparation (LiDAR + orthophotos)",
    )


def phase_flair_inference(config: dict) -> bool:
    checkpoint = config["inference"]["checkpoint"]
    if not Path(checkpoint).exists():
        print(f"\n✗ Error: Checkpoint not found: {checkpoint}")
        return False

    output_name = config["pipeline"]["output_name"]
    predictions_dir = f"predictions_{output_name}"

    cmd_args = [
        "--manifest",
        config["data"]["manifest"],
        "--checkpoint",
        checkpoint,
        "--output_dir",
        predictions_dir,
        "--tile_size",
        str(config["inference"]["tile_size"]),
        "--overlap",
        str(config["inference"]["overlap"]),
        "--grid_step",
        str(config["inference"]["grid_step"]),
        "--splits",
        *config["pipeline"]["splits"],
        "--batch_size",
        str(config["inference"].get("batch_size", 8)),
        "--herb_margin",
        str(config["inference"].get("herb_margin", 3.0)),
    ]

    if config["inference"].get("download_checkpoint", False):
        cmd_args.append("--download_checkpoint")
    if not config["inference"].get("tta", True):
        cmd_args.append("--no-tta")
    if config["inference"].get("tta_modes"):
        cmd_args.extend(["--tta-modes", *config["inference"]["tta_modes"]])
    if config["inference"].get("class_bias"):
        cmd_args.extend(["--class_bias", *config["inference"]["class_bias"]])
    if not config["inference"].get("fp16", True):
        cmd_args.append("--no-fp16")
    if not config["inference"].get("compile", True):
        cmd_args.append("--no-compile")
    if config["inference"].get("use_ir", False):
        cmd_args.append("--use_ir")

    return run_module_main(
        "src.inference.inference_flair_context",
        cmd_args,
        "PHASE 2: FLAIR context-aware inference",
    )


def phase_lidar_flair_merge(config: dict) -> bool:
    output_name = config["pipeline"]["output_name"]
    for split in config["pipeline"]["splits"]:
        las_dir = f"data/{split}"
        flair_dir = f"predictions_{output_name}/{split}"
        output_dir = f"merged_classifications_{output_name}/{split}"

        if not run_module_main(
            "src.postprocessing.merge_classifications",
            [
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


def phase_final_merge(config: dict) -> bool:
    output_name = config["pipeline"]["output_name"]
    resolution = config["data"]["resolution"]
    for split in config["pipeline"]["splits"]:
        merged_dir = f"merged_classifications_{output_name}/{split}"
        output_file = f"final_{output_name}_{split}.tif"

        if not Path(merged_dir).exists():
            print(f"\n✗ Warning: Merged directory not found: {merged_dir}")
            continue

        cmd_args = [
            "--input",
            merged_dir,
            "--output",
            output_file,
            "--strategy",
            config["merge"]["strategy"],
            "--clip-min",
            "0",
            "--clip-max",
            "3",
        ]

        if config["merge"].get("smooth", False):
            cmd_args.extend(
                [
                    "--smooth",
                    "--pixel-size",
                    str(resolution),
                    "--smooth-iterations",
                    str(config["merge"].get("smooth_iterations", 3)),
                    "--smooth-cores",
                    str(config["merge"].get("smooth_cores", 3)),
                ]
            )
        if config["merge"].get("resample_mismatch", False):
            cmd_args.append("--resample-mismatch")

        if not run_module_main(
            "src.postprocessing.merge_tifs",
            cmd_args,
            f"PHASE 4: Final merge for {split} split",
        ):
            return False

    return True


def phase_vectorization(config: dict) -> bool:
    try:
        from src.postprocessing.vectorize_raster import vectorize_raster
    except ImportError as e:
        print(f"\n✗ Error: Cannot import vectorize_raster: {e}")
        print("Make sure GDAL Python bindings are installed: pip install gdal")
        return False

    output_name = config["pipeline"]["output_name"]
    vec_fmt = config["vectorization"]["format"]
    eight_connected = config["vectorization"].get("eight_connected", False)
    ext_map = {"ESRI Shapefile": "shp", "GPKG": "gpkg", "GeoJSON": "geojson"}
    ext = ext_map[vec_fmt]

    for split in config["pipeline"]["splits"]:
        output_file = f"final_{output_name}_{split}.tif"
        if not Path(output_file).exists():
            print(f"\n✗ Warning: Final raster not found: {output_file}")
            continue

        vector_output = f"final_{output_name}_{split}.{ext}"

        print(f"\n{'=' * 70}")
        print(f"PHASE 5: Vectorization for {split} split")
        print(f"{'=' * 70}\n")

        if not vectorize_raster(
            output_file,
            vector_output,
            vec_fmt,
            use_8connected=eight_connected,
            field_name=config["vectorization"].get("field_name", "class"),
        ):
            print(f"\n⚠ Warning: Vectorization failed for {split} split")

    return True


PHASE_FUNCS = {
    "data_preparation": phase_data_preparation,
    "flair_inference": phase_flair_inference,
    "lidar_flair_merge": phase_lidar_flair_merge,
    "final_merge": phase_final_merge,
    "vectorization": phase_vectorization,
}

PHASE_ORDER = [
    "data_preparation",
    "flair_inference",
    "lidar_flair_merge",
    "final_merge",
    "vectorization",
]

BLOCKING_PHASES = {
    "data_preparation",
    "flair_inference",
    "lidar_flair_merge",
    "final_merge",
}


def print_summary(config: dict, state: StateManager, elapsed: float) -> None:
    output_name = config["pipeline"]["output_name"]
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")

    print("\nPhase results:")
    for phase in PHASE_ORDER:
        info = state.state["phases"].get(phase, {})
        status = info.get("status", "pending")
        dur = info.get("duration_seconds")
        dur_str = f"  ({dur:.1f}s)" if dur is not None else ""
        symbol = {"success": "✓", "failed": "✗", "skipped": "→", "pending": " "}.get(
            status, " "
        )
        print(f"  {symbol} {phase}: {status}{dur_str}")

    print("\nOutputs:")
    print(f"  Predictions: predictions_{output_name}/")
    print(f"  Merged classifications: merged_classifications_{output_name}/")

    vec_fmt = config["vectorization"]["format"]
    ext_map = {"ESRI Shapefile": "shp", "GPKG": "gpkg", "GeoJSON": "geojson"}
    ext = ext_map.get(vec_fmt, "shp")

    for split in config["pipeline"]["splits"]:
        output_file = f"final_{output_name}_{split}.tif"
        if Path(output_file).exists():
            print(f"  Final {split} raster: {output_file}")
        if config["phases"].get("vectorization", False):
            vector_output = f"final_{output_name}_{split}.{ext}"
            if Path(vector_output).exists():
                print(f"  Final {split} vector: {vector_output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Complete GrandLyon vegetation stratification pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to YAML config file (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--force",
        nargs="+",
        default=[],
        metavar="PHASE",
        help="Force specific phases to rerun (use 'all' to force everything)",
    )
    args = parser.parse_args()

    config_path = args.config
    forced = set(args.force)

    config = load_config(config_path)
    cfg_hash = config_hash(config_path)

    state_path = Path(config_path).with_name(Path(config_path).stem + "_state.json")
    state = StateManager(state_path, cfg_hash)

    start_time = time.time()

    print("=" * 70)
    print("GRANDLYON VEGETATION STRATIFICATION PIPELINE")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"State:  {state_path}")
    print(f"Manifest: {config['data']['manifest']}")
    print(f"Checkpoint: {config['inference']['checkpoint']}")
    print(f"Resolution: {config['data']['resolution']}m")
    print(f"Splits: {', '.join(config['pipeline']['splits'])}")
    print(f"Output prefix: {config['pipeline']['output_name']}")

    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.exists():
        print(f"\n✗ Error: Manifest not found: {manifest_path}")
        return 1

    for phase_name in PHASE_ORDER:
        if not config["phases"].get(phase_name, True):
            state.skip(phase_name, "disabled in config")
            continue

        if not state.should_run(phase_name, forced):
            dur = state.state["phases"][phase_name].get("duration_seconds")
            dur_str = f" in {dur:.1f}s" if dur is not None else ""
            print(f"→ {phase_name}: skipping (succeeded{dur_str})")
            continue

        state.begin(phase_name)
        try:
            success = PHASE_FUNCS[phase_name](config)
            state.end(phase_name, success)
        except Exception as e:
            import traceback

            traceback.print_exc()
            state.end(phase_name, False, error=str(e))
            success = False

        if not success and phase_name in BLOCKING_PHASES:
            print(f"✗ Pipeline stopped at {phase_name}")
            return 1

    elapsed = time.time() - start_time
    print_summary(config, state, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
