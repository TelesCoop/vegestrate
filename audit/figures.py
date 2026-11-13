import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from audit.common import load_config, output_path, pick
from audit.examples import NODATA, hex_rgb, raster_categories
from audit.prepare import fetch_wms
from audit.report import pct

GREEN = "#32412d"
LIGHT_GREEN = "#97b090"
GOLD = "#c4ab66"

CLASS_LABELS = {
    "vegetal": "Végétal",
    "non_vegetal": "Non végétal",
    "ligneux": "Ligneux",
    "herbace": "Herbacé",
    "arbre": "Arbre",
    "arbuste": "Arbuste",
    "herbe": "Herbe",
}

PLACE_LAYERS = {
    "eva": ("eva_groupes", "EVA 2023"),
    "vegestrate": ("vegestrate_natif", "Végéstrate 2023"),
    "accord": ("accord_vegetation", "Accord végétation"),
}


def colorize(array, categories, nodata, transparent=()):
    lut = np.zeros((256, 4), dtype=np.uint8)
    for value, _, color in categories:
        if value not in transparent:
            lut[value] = (*hex_rgb(color), 255)
    lut[nodata] = 0
    return lut[array]


def transparent_values(cfg):
    groups = list(cfg["eva_groups"])
    return {
        "eva_groupes": {groups.index("non_vegetal") + 1},
        "vegestrate_natif": {0},
        "accord_vegetation": {4},
    }


def blend(ortho, rgba, alpha):
    a = rgba[..., 3:4] / 255 * alpha
    return (ortho[..., :3] * (1 - a) + rgba[..., :3] * a).astype(np.uint8)


def read_raster(path, bounds, size_px):
    with rasterio.open(path) as src:
        return src.read(
            1,
            window=from_bounds(*bounds, src.transform),
            out_shape=(size_px, size_px),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=NODATA,
        )


def place_bounds(cfg, place):
    x, y = cfg["places"][place]["center"]
    half = cfg["place_size_m"] / 2
    return x - half, y - half, x + half, y + half


def render_layers(cfg, place, bounds, size_px):
    fig_cfg = cfg["figures"]
    categories = raster_categories(cfg)
    transparent = transparent_values(cfg)
    ortho = fetch_wms(cfg, bounds, size_px, size_px)
    images = {"ortho": ortho[..., :3]}
    for key, (raster, _) in PLACE_LAYERS.items():
        path = output_path(cfg, "exemples") / f"{place}_{raster}.tif"
        array = read_raster(path, bounds, size_px)
        if (array == NODATA).all():
            raise ValueError(f"{path} has no data in {bounds}")
        rgba = colorize(array, categories[raster], NODATA, transparent[raster])
        images[key] = blend(ortho, rgba, fig_cfg["overlay_alpha"])
    return images


def save_jpg(fig, path, cfg, dpi=None):
    fig.savefig(
        path,
        dpi=dpi or cfg["figures"]["dpi"],
        pil_kwargs={"quality": cfg["figures"]["jpeg_quality"]},
    )


def save_photo(path, image, cfg):
    plt.imsave(path, image, pil_kwargs={"quality": cfg["figures"]["jpeg_quality"]})


def legend_handles(cfg, raster):
    hidden = transparent_values(cfg)[raster]
    return [
        Patch(color=color, label=label)
        for value, label, color in raster_categories(cfg)[raster]
        if value not in hidden
    ]


def places_figures(cfg, out):
    size_px = cfg["figures"]["web_px"]
    grid = {}
    for place in cfg["places"]:
        images = render_layers(cfg, place, place_bounds(cfg, place), size_px)
        for key, image in images.items():
            save_photo(out / f"{place}_{key}.jpg", image, cfg)
        grid[place] = images
    titles = {"ortho": "Orthophotographie 2023"} | {
        k: title for k, (_, title) in PLACE_LAYERS.items()
    }
    fig, axes = plt.subplots(len(grid), 4, figsize=(12, 3 * len(grid) + 1.5))
    for row, (place, images) in zip(axes, grid.items()):
        for ax, (key, image) in zip(row, images.items()):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if ax is row[0]:
                ax.set_ylabel(cfg["places"][place]["label"], fontsize=8)
    for ax, key in zip(axes[0], titles):
        ax.set_title(titles[key], fontsize=10)
    for ax, (raster, _) in zip(axes[-1][1:], PLACE_LAYERS.values()):
        ax.legend(
            handles=legend_handles(cfg, raster),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout()
    save_jpg(fig, out / "lieux.jpg", cfg)
    plt.close(fig)


def zoom_figures(cfg, out):
    fig_cfg = cfg["figures"]
    half = fig_cfg["zoom_size_m"] / 2
    size_px = fig_cfg["zoom_px"]
    for name, spec in fig_cfg["zooms"].items():
        x, y = spec["center"]
        bounds = (x - half, y - half, x + half, y + half)
        images = render_layers(cfg, spec["place"], bounds, size_px)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        for ax, key, title in zip(
            axes,
            ("ortho", "eva", "vegestrate"),
            ("Orthophotographie 2023", "EVA 2023", "Végéstrate 2023"),
        ):
            ax.imshow(images[key])
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        fig.tight_layout()
        save_jpg(fig, out / f"zoom_{name}.jpg", cfg, fig_cfg["web_dpi"])
        plt.close(fig)


def legend_data(cfg, out):
    legend = {
        key: [
            {"label": h.get_label(), "color": to_hex(h.get_facecolor())}
            for h in legend_handles(cfg, raster)
        ]
        for key, (raster, _) in PLACE_LAYERS.items()
    }
    (out / "legende.json").write_text(json.dumps(legend, ensure_ascii=False, indent=1))


def excluded_zone_figure(cfg, out):
    zone = gpd.read_file(output_path(cfg, "exclusion.gpkg"))
    xmin, ymin, xmax, ymax = zone.total_bounds
    side = max(xmax - xmin, ymax - ymin) + 800
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    bounds = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    size_px = cfg["figures"]["web_px"]
    ortho = fetch_wms(cfg, bounds, size_px, size_px)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ortho, extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
    zone.boundary.plot(ax=ax, color="#d7301f", linewidth=1.2)
    ax.set_axis_off()
    ax.set_title("Zone exclue : camp militaire du Mont Verdun", fontsize=10)
    fig.tight_layout()
    save_jpg(fig, out / "zone_exclue.jpg", cfg)
    plt.close(fig)


def matrix_figures(cfg, out):
    mat = pd.read_csv(output_path(cfg, "matrices.csv"))
    for name, level, version in (
        ("vegetation", "vegetation", "natif"),
        ("ligneux", "ligneux_herbace", "principal"),
        ("strates", "strates", "principal"),
    ):
        sel = pick(mat, niveau=level, version_vegestrate=version)
        rows = list(cfg["levels"][level]["eva"])
        cols = list(cfg["levels"][level]["vegestrate"])
        grid = sel.pivot(
            index="classe_eva",
            columns="classe_vegestrate",
            values="part_de_la_classe_eva",
        ).reindex(index=rows, columns=cols)
        fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2, 1.1 * len(rows) + 1.2))
        ax.imshow(grid.values, cmap="Greens", vmin=0, vmax=1)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = grid.values[i, j]
                ax.text(
                    j,
                    i,
                    pct(v),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > 0.6 else GREEN,
                )
        ax.set_xticks(range(len(cols)), [CLASS_LABELS[c] for c in cols])
        ax.set_yticks(range(len(rows)), [CLASS_LABELS[r] for r in rows])
        ax.set_xlabel("Végéstrate")
        ax.set_ylabel("EVA")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        fig.tight_layout()
        fig.savefig(out / f"matrice_{name}.png", dpi=cfg["figures"]["dpi"])
        plt.close(fig)


def cover_rules_figure(cfg, out):
    df = pick(
        pd.read_csv(output_path(cfg, "regles_couverture.csv")),
        version_vegestrate="principal",
    )
    labels = [f"{c} ({r})" for c, r in zip(df["code_eva"], df["regle"])]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))
    ax.barh(
        y + 0.2,
        df["part_surface_conforme"],
        0.4,
        color=GREEN,
        label="part de la surface",
    )
    ax.barh(
        y - 0.2,
        df["part_polygones_conformes"],
        0.4,
        color=LIGHT_GREEN,
        label="part des polygones",
    )
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(
        "Polygones EVA conformes à leur règle de couverture selon Végéstrate",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "regles_couverture.png", dpi=cfg["figures"]["dpi"])
    plt.close(fig)


def edge_figure(cfg, out):
    df = pd.read_csv(output_path(cfg, "distance_bord.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, level, title in zip(
        axes, ("vegetation", "ligneux"), ("Végétation", "Ligneux")
    ):
        for side, color in (("dans EVA", GREEN), ("hors EVA", GOLD)):
            sel = pick(df, niveau=level, cote=side)
            ax.plot(
                sel["distance_au_bord"],
                sel["part_vegestrate_positif"],
                "o-",
                color=color,
                label=side,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("distance au bord de la zone EVA")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("part vue positive par Végéstrate")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "distance_bord.png", dpi=cfg["figures"]["dpi"])
    plt.close(fig)


def patches_figure(cfg, out):
    df = pick(pd.read_csv(output_path(cfg, "petites_taches.csv")), niveau="ligneux")
    df = df[df["groupe_eva"].isin(cfg["eva_groups"])]
    labels = [cfg["eva_groups"][g]["label"] for g in df["groupe_eva"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(labels, df["part_petites_taches"], color=GREEN)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(
        "Part de la végétation ligneuse Végéstrate en taches de moins de 100 m²",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "petites_taches.png", dpi=cfg["figures"]["dpi"])
    plt.close(fig)


def communes_figure(cfg, out):
    df = pd.read_csv(output_path(cfg, "couverture_communes.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, positive, title in zip(
        axes, ("vegetal", "ligneux"), ("Végétal", "Ligneux")
    ):
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8)
        ax.scatter(
            df[f"part_{positive}_eva"],
            df[f"part_{positive}_vegestrate"],
            s=df[f"surface_comparable_{positive}_km2"] * 6,
            color=GREEN,
            alpha=0.6,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("part selon EVA")
        ax.set_ylabel("part selon Végéstrate")
        ax.set_title(f"{title}, une commune par point", fontsize=10)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(out / "communes.png", dpi=cfg["figures"]["dpi"])
    plt.close(fig)


def figure_links(text):
    return re.findall(r"\]\((figures/[^)]+\.(?:png|jpg))\)", text)


def check_figure_links(cfg):
    text = Path(cfg["paths"]["methodology"]).read_text()
    links = figure_links(text)
    assert links, "no figure link in METHODOLOGY.md"
    missing = [link for link in links if not output_path(cfg, link).exists()]
    assert not missing, f"missing figures: {missing}"
    assert "$" not in text, "unsubstituted placeholder in METHODOLOGY.md"


def main():
    parser = argparse.ArgumentParser(description="Figures du rapport")
    parser.add_argument("--config", default="audit/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = output_path(cfg, "figures")
    out.mkdir(exist_ok=True)
    for build in (
        matrix_figures,
        cover_rules_figure,
        edge_figure,
        patches_figure,
        communes_figure,
        excluded_zone_figure,
        zoom_figures,
        legend_data,
        places_figures,
    ):
        build(cfg, out)
        print(f"{build.__name__} done", flush=True)
    check_figure_links(cfg)


if __name__ == "__main__":
    main()
