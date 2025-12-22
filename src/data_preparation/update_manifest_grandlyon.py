import csv
import json
from pathlib import Path

RECT_X_MIN = 1836000
RECT_Y_MIN = 5165000
RECT_X_MAX = 1858000
RECT_Y_MAX = 5190000


def tile_intersects_rectangle(x_min, y_min, x_max, y_max):
    """Check if a tile intersects with the target rectangle.

    A tile intersects if it overlaps with the rectangle in both X and Y dimensions.
    """

    no_overlap = (
        x_max <= RECT_X_MIN
        or x_min >= RECT_X_MAX
        or y_max <= RECT_Y_MIN
        or y_min >= RECT_Y_MAX
    )
    return not no_overlap


def main():
    csv_path = Path("data/nuage-de-points-lidar-2023-de-la-metropole-de-lyon.csv")
    manifest_path = Path("data/dataset_manifest_grandlyon.json")

    matching_tiles = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])

            if tile_intersects_rectangle(x_min, y_min, x_max, y_max):
                tile_id = row["nom"]
                url = row["url"].strip()
                matching_tiles.append(
                    {
                        "tile_id": tile_id,
                        "orthophoto": f"test/{tile_id}_orthophoto.tif",
                        "classification_map": f"test/{tile_id}_classification_map.tif",
                        "url": url,
                    }
                )

    print(f"Found {len(matching_tiles)} tiles in rectangle:")
    print(f"  X range: {RECT_X_MIN} to {RECT_X_MAX}")
    print(f"  Y range: {RECT_Y_MIN} to {RECT_Y_MAX}")
    print("\nTiles:")
    for tile in matching_tiles:
        print(f"  - {tile['tile_id']}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest["test"] = matching_tiles

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Updated {manifest_path}")
    print(f"  Test tiles: {len(manifest['test'])}")
    print(f"  Train tiles: {len(manifest['train'])}")


if __name__ == "__main__":
    main()
