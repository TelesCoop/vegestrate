import argparse
import json
from pathlib import Path


def generate_tile_grid(bbox, tile_step):
    xmin, ymin, xmax, ymax = bbox
    step_m = tile_step * 100.0

    x_start = int(xmin // step_m) * tile_step
    y_start = int(ymin // step_m) * tile_step

    tiles = []
    x = x_start
    while x * 100.0 < xmax:
        y = y_start
        while y * 100.0 < ymax:
            tiles.append((x, y))
            y += tile_step
        x += tile_step

    return tiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset manifest for Pléiades satellite imagery"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[820000, 6490000, 880000, 6560000],
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Bounding box in Lambert 93 (EPSG:2154) meters (default: Métropole de Lyon)",
    )
    parser.add_argument(
        "--tile_step",
        type=int,
        default=5,
        help="Tile size in 100m units (default: 5 = 500m, 1000x1000px at 0.5m/px)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset_manifest_pleiades.json",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENERATE PLÉIADES MANIFEST FROM BOUNDING BOX")
    print("=" * 70)
    print(f"BBOX (Lambert 93): {args.bbox}")
    print(f"Tile step: {args.tile_step} (= {args.tile_step * 100}m per tile)")
    print(f"Split: {args.split}")
    print(f"Output: {args.output}")
    print("=" * 70)

    tiles = generate_tile_grid(args.bbox, args.tile_step)
    print(f"\nGenerated {len(tiles)} tiles")

    entries = []
    for x, y in tiles:
        tile_id = f"{x}_{y}"
        entries.append(
            {
                "tile_id": tile_id,
                "orthophoto": f"{args.split}/{tile_id}_orthophoto.tif",
            }
        )

    manifest = {
        "train": entries if args.split == "train" else [],
        "test": entries if args.split == "test" else [],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest saved to {output_path}")
    print(f"  {args.split}: {len(entries)} tiles")


if __name__ == "__main__":
    main()
