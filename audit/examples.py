import argparse

import numpy as np
import pandas as pd
import rasterio

from audit.common import Inputs, load_config, output_path, stratum

NODATA = 255

EVA_COLORS = {
    "arbore": "#1a7f37",
    "arbustif": "#8fc31f",
    "herbace": "#f2e05c",
    "boise_mixte": "#0b4d23",
    "non_vegetal": "#bdbdbd",
    "eau": "#3c8dde",
    "non_comparable": "#c77dff",
}

STRATA_CATEGORIES = [
    (0, "Aucune végétation", "#e0e0e0"),
    (1, "Herbe", "#f2e05c"),
    (2, "Arbuste", "#8fc31f"),
    (3, "Arbre", "#1a7f37"),
]

VEGETATION_CATEGORIES = [
    (1, "Végétal pour EVA et Vegestrate", "#1a7f37"),
    (2, "Végétal pour EVA seul", "#e66101"),
    (3, "Végétal pour Vegestrate seul", "#5e3c99"),
    (4, "Non végétal pour les deux", "#e0e0e0"),
    (5, "EVA non comparable", "#f4c2f4"),
]

STRATES_CATEGORIES = [
    (1, "Même strate", "#1a7f37"),
    (2, "Végétal pour les deux, strate différente", "#fdb863"),
    (3, "EVA boisé multistrate, Vegestrate ligneux", "#4dac26"),
    (4, "EVA boisé multistrate, Vegestrate herbe", "#b8e186"),
    (5, "Végétal pour EVA seul", "#e66101"),
    (6, "Végétal pour Vegestrate seul", "#5e3c99"),
    (7, "Non végétal pour les deux", "#e0e0e0"),
    (8, "EVA non comparable", "#f4c2f4"),
]


def hex_rgb(color):
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def write_qml(path, categories):
    entries = "\n".join(
        f'<paletteEntry value="{v}" label="{label}" color="{c}" alpha="255"/>'
        for v, label, c in categories
    )
    path.write_text(
        '<!DOCTYPE qgis><qgis><pipe><rasterrenderer type="paletted" band="1" '
        f'opacity="1"><colorPalette>\n{entries}\n</colorPalette></rasterrenderer>'
        "</pipe></qgis>\n"
    )


def write_categorical(path, array, transform, crs, categories):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.write_colormap(1, {v: hex_rgb(c) for v, _, c in categories})
    write_qml(path.with_suffix(".qml"), categories)


def classify(d, inputs, cfg):
    ec = inputs.eva_classes
    lv = cfg["levels"]
    native = d["native"]
    recut = stratum(native, d["hbin"], cfg["height_cuts"]["principal"])
    valid = (d["iris"] > 0) & (d["excluded"] == 0) & (d["eva"] > 0)
    group = d["group"]
    eva_veg = ec.group_mask(group, lv["vegetation"]["eva"]["vegetal"])
    not_cmp = ec.group_mask(group, ["non_comparable"])
    mixed = ec.group_mask(group, ["boise_mixte"])
    veg = recut > 0
    accord_veg = np.select(
        [not_cmp, eva_veg & veg, eva_veg, veg], [5, 1, 2, 3], 4
    ).astype(np.uint8)
    eva_stratum = np.select(
        [
            ec.group_mask(group, ["arbore"]),
            ec.group_mask(group, ["arbustif"]),
            ec.group_mask(group, ["herbace"]),
        ],
        [3, 2, 1],
        0,
    )
    accord_strates = np.select(
        [
            not_cmp,
            mixed & (recut >= 2),
            mixed & (recut == 1),
            eva_veg & veg & (eva_stratum == recut),
            eva_veg & veg,
            eva_veg,
            veg,
        ],
        [8, 3, 4, 1, 2, 5, 6],
        7,
    ).astype(np.uint8)
    eva_groups = group.astype(np.uint8)
    rasters = {
        "eva_groupes": eva_groups,
        "vegestrate_natif": native.astype(np.uint8),
        "vegestrate_recoupe": recut,
        "accord_vegetation": accord_veg,
        "accord_strates": accord_strates,
    }
    return {k: np.where(valid, v, NODATA).astype(np.uint8) for k, v in rasters.items()}


def main():
    parser = argparse.ArgumentParser(description="Rasters d'exemple sur 5 lieux")
    parser.add_argument("--config", default="audit/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    inputs = Inputs(cfg)
    out_dir = output_path(cfg, "exemples")
    out_dir.mkdir(exist_ok=True)
    eva_categories = [
        (i + 1, spec["label"], EVA_COLORS[g])
        for i, (g, spec) in enumerate(cfg["eva_groups"].items())
    ]
    categories = {
        "eva_groupes": eva_categories,
        "vegestrate_natif": STRATA_CATEGORIES,
        "vegestrate_recoupe": STRATA_CATEGORIES,
        "accord_vegetation": VEGETATION_CATEGORIES,
        "accord_strates": STRATES_CATEGORIES,
    }
    pixel_m2 = inputs.transform.a**2
    rows = []
    for place, spec in cfg["places"].items():
        window = inputs.grid_window(spec["center"], cfg["place_size_m"])
        d = inputs.window(*window)
        rasters = classify(d, inputs, cfg)
        for name, array in rasters.items():
            path = out_dir / f"{place}_{name}.tif"
            write_categorical(path, array, d["transform"], cfg["crs"], categories[name])
            if name.startswith("accord"):
                counts = np.bincount(array.ravel(), minlength=NODATA + 1)
                total = counts[:NODATA].sum()
                for v, label, _ in categories[name]:
                    rows.append(
                        {
                            "lieu": place,
                            "libelle_lieu": spec["label"],
                            "raster": name,
                            "categorie": label,
                            "surface_m2": counts[v] * pixel_m2,
                            "part": counts[v] / total if total else np.nan,
                        }
                    )
        print(f"{place} done")
    pd.DataFrame(rows).to_csv(output_path(cfg, "exemples_stats.csv"), index=False)


if __name__ == "__main__":
    main()
