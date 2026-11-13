import argparse
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import pyogrio

from audit.common import (
    STRATA,
    EvaClasses,
    load_config,
    output_path,
    pick,
    pontius,
    stratum,
)

PIXEL_M2 = 0.04


class Cubes:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ec = EvaClasses(cfg)
        zones = output_path(cfg, "zones.gpkg")
        self.iris = pyogrio.read_dataframe(zones, layer="iris", read_geometry=False)
        lcz = pyogrio.read_dataframe(zones, layer="lcz", read_geometry=False)
        lcz_desc = lcz.drop_duplicates("lcz_index").set_index("lcz_index")
        self.lcz_codes = sorted(lcz["lcz_index"].unique())
        self.lcz_label = ["Sans LCZ"] + [
            f"{c} - {lcz_desc.loc[c, 'lcz_description']}" for c in self.lcz_codes
        ]
        self.versions = {"natif": None, **cfg["height_cuts"]}
        z = pd.read_csv(output_path(cfg, "cube_zones.csv.gz"))
        groups = np.array(["absent"] + self.ec.groups)
        z["groupe"] = groups[self.ec.group_of_index[z["cod3"]]]
        z["code_eva"] = np.array([0] + self.ec.codes)[z["cod3"]]
        for name, cut in self.versions.items():
            z[f"strate_{name}"] = np.array(STRATA)[
                stratum(z["natif"].values, z["hbin"].values, cut)
            ]
        z["km2"] = z["pixels"] * PIXEL_M2 / 1e6
        self.excluded = z[z["exclu"] == 1]
        self.zones = z[(z["exclu"] == 0) & (z["cod3"] > 0)]

    def labels(self, level, version):
        spec = self.cfg["levels"][level]
        eva = {g: cat for cat, gs in spec["eva"].items() for g in gs}
        veg = {s: cat for cat, ss in spec["vegestrate"].items() for s in ss}
        return (
            self.zones["groupe"].map(eva),
            self.zones[f"strate_{version}"].map(veg),
            list(spec["eva"]),
        )

    def level_versions(self, level):
        if level.startswith("vegetation"):
            return ["natif"]
        return list(self.versions)


def matrices(cubes):
    rows, metrics = [], []
    for level in cubes.cfg["levels"]:
        for version in cubes.level_versions(level):
            eva, veg, cats = cubes.labels(level, version)
            inside = eva.notna()
            m = (
                cubes.zones[inside]
                .groupby([eva[inside], veg[inside]])["km2"]
                .sum()
                .unstack(fill_value=0.0)
                .reindex(index=cats, columns=cats, fill_value=0.0)
            )
            row_share = m.div(m.sum(axis=1), axis=0)
            col_share = m.div(m.sum(axis=0), axis=1)
            for e in cats:
                for v in cats:
                    rows.append(
                        {
                            "niveau": level,
                            "version_vegestrate": version,
                            "classe_eva": e,
                            "classe_vegestrate": v,
                            "surface_km2": m.loc[e, v],
                            "part_de_la_classe_eva": row_share.loc[e, v],
                            "part_de_la_classe_vegestrate": col_share.loc[e, v],
                        }
                    )
            metrics.append(
                {
                    "niveau": level,
                    "version_vegestrate": version,
                    "surface_km2": m.values.sum(),
                    "surface_hors_matrice_km2": cubes.zones.loc[~inside, "km2"].sum(),
                    **pontius(m.values),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(metrics)


def compare(op, share, value):
    return share >= value if op == "ge" else share < value


def polygons(cubes):
    cfg = cubes.cfg
    attrs = pyogrio.read_dataframe(
        cfg["paths"]["eva"],
        layer=cfg["paths"]["eva_layer"],
        columns=["Id", "COD3", "Area_m2"],
        read_geometry=False,
    )
    p = pd.read_csv(output_path(cfg, "cube_polygones.csv.gz"))
    comp, summary = [], []
    for version in ("natif", "principal"):
        s = np.array(STRATA)[
            stratum(p["natif"].values, p["hbin"].values, cubes.versions[version])
        ]
        t = p.assign(strate=s).pivot_table(
            index="eva", columns="strate", values="pixels", aggfunc="sum", fill_value=0
        )
        t = t.reindex(columns=STRATA, fill_value=0)
        n = t.sum(axis=1)
        shares = t.div(n, axis=0)
        shares["ligneux"] = shares["arbre"] + shares["arbuste"]
        shares["vegetal"] = 1 - shares["aucune"]
        a = attrs.iloc[shares.index - 1]
        shares["id_eva"] = a["Id"].values
        shares["code_eva"] = a["COD3"].values
        shares["surface_eva_m2"] = a["Area_m2"].values
        shares["surface_mesuree_m2"] = n.values * PIXEL_M2
        shares["version_vegestrate"] = version
        complete = shares["surface_mesuree_m2"] >= 0.5 * shares["surface_eva_m2"]
        comp.append(shares.reset_index(drop=True))
        for code, rule in cfg["cover_rules"].items():
            sel = shares[complete & (shares["code_eva"] == code)]
            if sel.empty:
                continue
            ok = compare(rule["op"], sel[rule["target"]], rule["value"])
            q = sel[rule["target"]].quantile([0.25, 0.5, 0.75])
            summary.append(
                {
                    "code_eva": code,
                    "version_vegestrate": version,
                    "regle": f"{rule['target']} {'>=' if rule['op'] == 'ge' else '<'} {rule['value']:.0%}",
                    "page_guide": cfg["guide_pages"][code],
                    "polygones": len(sel),
                    "surface_km2": sel["surface_mesuree_m2"].sum() / 1e6,
                    "part_polygones_conformes": ok.mean(),
                    "part_surface_conforme": sel.loc[ok, "surface_mesuree_m2"].sum()
                    / sel["surface_mesuree_m2"].sum(),
                    "q1": q[0.25],
                    "mediane": q[0.5],
                    "q3": q[0.75],
                }
            )
    return pd.concat(comp), pd.DataFrame(summary)


def distance_labels(edges):
    bounds = [0] + list(edges)
    return [f"{a}-{b} m" for a, b in zip(bounds, bounds[1:])] + [f"> {edges[-1]} m"]


def edges(cubes):
    labels = distance_labels(cubes.cfg["edge_distance_bins_m"])
    rows = []
    for level in ("vegetation", "ligneux"):
        b = pd.read_csv(output_path(cubes.cfg, f"cube_bord_{level}.csv"))
        for (inside, dist), g in b.groupby(["interieur", "classe_distance"]):
            total = g["pixels"].sum()
            rows.append(
                {
                    "niveau": level,
                    "cote": "dans EVA" if inside else "hors EVA",
                    "distance_au_bord": labels[dist],
                    "surface_km2": total * PIXEL_M2 / 1e6,
                    "part_vegestrate_positif": g.loc[g["accord"] == 1, "pixels"].sum()
                    / total,
                }
            )
    return pd.DataFrame(rows)


def patches(cubes):
    groups = ["absent"] + cubes.ec.groups
    rows = []
    for level in ("vegetation", "ligneux"):
        t = pd.read_csv(output_path(cubes.cfg, f"cube_taches_{level}.csv"))
        t["groupe_eva"] = [groups[g] for g in t["groupe"]]
        for name, g in [("total", t)] + list(t.groupby("groupe_eva")):
            small = g.loc[g["petite"] == 1, "pixels"].sum()
            rows.append(
                {
                    "niveau": level,
                    "groupe_eva": name,
                    "surface_petites_taches_km2": small * PIXEL_M2 / 1e6,
                    "surface_vegestrate_km2": g["pixels"].sum() * PIXEL_M2 / 1e6,
                    "part_petites_taches": small / g["pixels"].sum(),
                }
            )
    return pd.DataFrame(rows)


def coverage(cubes):
    z = cubes.zones
    out = {}
    for level in ("vegetation", "ligneux_herbace"):
        eva, veg, _ = cubes.labels(level, "principal")
        positive = "vegetal" if level == "vegetation" else "ligneux"
        inside = eva.notna()
        frame = pd.DataFrame(
            {
                "iris": z["iris"],
                "km2": z["km2"],
                "comparable": z["km2"].where(inside, 0.0),
                "eva": z["km2"].where(inside & (eva == positive), 0.0),
                "vegestrate": z["km2"].where(inside & (veg == positive), 0.0),
            }
        )
        out[positive] = frame.groupby("iris").sum()
    iris = cubes.iris.copy()
    iris.index = np.arange(1, len(iris) + 1)
    table = iris[["code", "nom", "code_commune", "commune"]].copy()
    table["surface_km2"] = out["vegetal"]["km2"]
    for positive, f in out.items():
        table[f"surface_comparable_{positive}_km2"] = f["comparable"]
        table[f"{positive}_eva_km2"] = f["eva"]
        table[f"{positive}_vegestrate_km2"] = f["vegestrate"]
    table = table.dropna(subset=["surface_km2"])
    communes = table.groupby(["code_commune", "commune"], as_index=False).sum(
        numeric_only=True
    )
    for t in (table, communes):
        for positive in out:
            base = t[f"surface_comparable_{positive}_km2"]
            t[f"part_{positive}_eva"] = t[f"{positive}_eva_km2"] / base
            t[f"part_{positive}_vegestrate"] = t[f"{positive}_vegestrate_km2"] / base
            t[f"ecart_{positive}_points"] = 100 * (
                t[f"part_{positive}_vegestrate"] - t[f"part_{positive}_eva"]
            )
    return table, communes


def lcz(cubes):
    rows = []
    z = cubes.zones
    for level in ("vegetation", "ligneux_herbace"):
        eva, veg, cats = cubes.labels(level, "principal")
        inside = eva.notna()
        for code, g in z[inside].groupby("lcz"):
            m = (
                g.groupby([eva[g.index], veg[g.index]])["km2"]
                .sum()
                .unstack(fill_value=0.0)
                .reindex(index=cats, columns=cats, fill_value=0.0)
            )
            rows.append(
                {
                    "niveau": level,
                    "lcz": cubes.lcz_label[code],
                    "surface_km2": m.values.sum(),
                    **pontius(m.values),
                    f"part_{cats[0]}_eva": m.loc[cats[0]].sum() / m.values.sum(),
                    f"part_{cats[0]}_vegestrate": m[cats[0]].sum() / m.values.sum(),
                }
            )
    return pd.DataFrame(rows)


def exclusion(cubes):
    x = cubes.excluded
    eva = x.groupby("groupe")["km2"].sum().rename("surface_km2").reset_index()
    eva.insert(0, "source", "EVA")
    eva = eva.rename(columns={"groupe": "classe"})
    veg = x.groupby("strate_natif")["km2"].sum().rename("surface_km2").reset_index()
    veg.insert(0, "source", "Végéstrate")
    veg = veg.rename(columns={"strate_natif": "classe"})
    return pd.concat([eva, veg])


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}".replace(".", ",")
    return str(v)


def pct(v):
    return f"{100 * v:.1f} %".replace(".", ",")


def md_table(df, percent=()):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    lines = [
        "| "
        + " | ".join(pct(r[c]) if c in percent else fmt(r[c]) for c in df.columns)
        + " |"
        for _, r in df.iterrows()
    ]
    return "\n".join([head, sep, *lines])


def value(df, column, **filters):
    return pick(df, **filters)[column].iloc[0]


def summary(cfg, metrics, rules, edge, patch):
    acc = metrics.set_index(["niveau", "version_vegestrate"])["accord"]
    strata_variants = acc["strates"][list(cfg["height_cuts"])]
    labels = distance_labels(cfg["edge_distance_bins_m"])
    near, far = labels[0], labels[-1]
    shares = {
        "accord_vegetation": acc["vegetation", "natif"],
        "accord_vegetation_nc": acc["vegetation_avec_non_comparable", "natif"],
        "accord_ligneux": acc["ligneux_herbace", "principal"],
        "accord_strates": acc["strates", "principal"],
        "strates_min": strata_variants.min(),
        "strates_max": strata_variants.max(),
        "taches_vegetation": value(
            patch, "part_petites_taches", niveau="vegetation", groupe_eva="total"
        ),
        "taches_ligneux_non_vegetal": value(
            patch, "part_petites_taches", niveau="ligneux", groupe_eva="non_vegetal"
        ),
    }
    for code in (111, 112, 311):
        shares[f"conforme_{code}"] = value(
            rules,
            "part_surface_conforme",
            code_eva=code,
            version_vegestrate="principal",
        )
    for key, side, dist in (
        ("dans_pres", "dans EVA", near),
        ("dans_loin", "dans EVA", far),
        ("hors_pres", "hors EVA", near),
        ("hors_loin", "hors EVA", far),
    ):
        shares[f"bord_{key}"] = value(
            edge,
            "part_vegestrate_positif",
            niveau="vegetation",
            cote=side,
            distance_au_bord=dist,
        )
    return {
        **{k: pct(v) for k, v in shares.items()},
        "bord_pres": cfg["edge_distance_bins_m"][0],
        "bord_loin": cfg["edge_distance_bins_m"][-1],
        "taches_ligneux_non_vegetal_km2": fmt(
            value(
                patch,
                "surface_petites_taches_km2",
                niveau="ligneux",
                groupe_eva="non_vegetal",
            )
        ),
    }


def render(cfg, cubes, metrics, rules, edge, patch):
    metric_cols = ["accord", "desaccord_quantite", "desaccord_allocation"]
    shown = metrics[
        metrics["version_vegestrate"].isin(["natif", "principal"])
        & (metrics["niveau"] != "vegetation_avec_non_comparable")
    ]
    principal = cfg["height_cuts"]["principal"]
    values = {
        **summary(cfg, metrics, rules, edge, patch),
        "indicateurs": md_table(
            shown[
                ["niveau", "version_vegestrate", "surface_km2", *metric_cols, "kappa"]
            ],
            percent=metric_cols,
        ),
        "exclusion_km2": fmt(cubes.excluded["km2"].sum()),
        "domaine_km2": fmt(cubes.zones["km2"].sum()),
        "coupe_principale": principal["label"],
        "arbuste_min": principal["shrub_min"],
        "arbre_min": principal["tree_min"],
        "variantes": ", ".join(f"`{name}`" for name in cfg["height_cuts"]),
        "groupes": md_table(
            pd.DataFrame(
                [
                    {
                        "groupe": g,
                        "libellé": spec["label"],
                        "codes EVA (page du guide)": ", ".join(
                            f"{c} (p. {cfg['guide_pages'][c]})" for c in spec["codes"]
                        ),
                    }
                    for g, spec in cfg["eva_groups"].items()
                ]
            )
        ),
    }
    template = Template((Path(__file__).parent / "methodology_template.md").read_text())
    Path(cfg["paths"]["methodology"]).write_text(template.substitute(values))


def main():
    parser = argparse.ArgumentParser(description="Tableaux et METHODOLOGY.md")
    parser.add_argument("--config", default="audit/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cubes = Cubes(cfg)
    mat, metrics = matrices(cubes)
    comp, rules = polygons(cubes)
    edge = edges(cubes)
    patch = patches(cubes)
    iris_t, communes = coverage(cubes)
    lcz_t = lcz(cubes)
    excl = exclusion(cubes)
    tables = {
        "matrices.csv": mat,
        "indicateurs_accord.csv": metrics,
        "polygones_composition.csv.gz": comp,
        "regles_couverture.csv": rules,
        "distance_bord.csv": edge,
        "petites_taches.csv": patch,
        "couverture_iris.csv": iris_t,
        "couverture_communes.csv": communes,
        "accord_lcz.csv": lcz_t,
        "zone_exclue.csv": excl,
    }
    for name, df in tables.items():
        df.to_csv(output_path(cfg, name), index=False)
    render(cfg, cubes, metrics, rules, edge, patch)
    print(f"tables -> {cfg['paths']['output_dir']}, {cfg['paths']['methodology']}")


if __name__ == "__main__":
    main()
