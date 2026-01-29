import csv
import json
from pathlib import Path


def main():
    csv_path = Path("data/nuage-de-points-lidar-2023-de-la-metropole-de-lyon.csv")
    manifest_path = Path("data/dataset_manifest_grandlyon.json")

    all_tiles = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            tile_id = row["nom"]
            url = row["url"].strip()
            all_tiles.append(
                {
                    "tile_id": tile_id,
                    "orthophoto": f"test/{tile_id}_orthophoto.tif",
                    "classification_map": f"test/{tile_id}_classification_map.tif",
                    "url": url,
                }
            )

    print(f"Found {len(all_tiles)} tiles from CSV")

    manifest = {"test": all_tiles, "train": []}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Updated {manifest_path}")
    print(f"  Test tiles: {len(manifest['test'])}")
    print(f"  Train tiles: {len(manifest['train'])}")


if __name__ == "__main__":
    main()
