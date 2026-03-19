import csv
import json
from pathlib import Path


def main():
    lidar_csv = Path("data/LIDAR2018.csv")
    ortho_csv = Path("data/ortho_IR_2018.csv")
    manifest_path = Path("data/dataset_manifest_2018.json")

    ortho_lookup = {}
    with open(ortho_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nom = row["nom"]
            x_str, y_str = nom.split("_")[:2]
            ortho_lookup[(int(x_str), int(y_str))] = row["url"].strip()

    all_tiles = []
    missing_ortho = 0
    with open(lidar_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_id = row["nom"]
            url = row["url"].strip()

            x, y = map(int, tile_id.split("_"))
            ortho_x = (x // 5) * 5
            ortho_y = (y // 5) * 5
            ortho_ecw_url = ortho_lookup.get((ortho_x, ortho_y))
            if ortho_ecw_url is None:
                missing_ortho += 1

            entry = {
                "tile_id": tile_id,
                "orthophoto": f"test/{tile_id}_orthophoto.tif",
                "classification_map": f"test/{tile_id}_classification_map.tif",
                "url": url,
            }
            if ortho_ecw_url:
                entry["ortho_ecw_url"] = ortho_ecw_url

            all_tiles.append(entry)

    manifest = {"test": all_tiles, "train": []}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Found {len(all_tiles)} LiDAR tiles")
    if missing_ortho:
        print(f"  Warning: {missing_ortho} tiles have no matching ortho")
    print(f"✓ Updated {manifest_path}")
    print(f"  Test tiles: {len(manifest['test'])}")
    print(f"  Train tiles: {len(manifest['train'])}")


if __name__ == "__main__":
    main()
