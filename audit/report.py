import argparse
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import pyogrio

from audit.common import STRATA, EvaClasses, load_config, output_path, pontius, stratum

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
        self.eva_absent_km2 = z[(z["exclu"] == 0) & (z["cod3"] == 0)]["km2"].sum()

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
    veg.insert(0, "source", "Vegestrate")
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


def matrix_md(mat, level, version, share):
    m = mat[(mat["niveau"] == level) & (mat["version_vegestrate"] == version)]
    t = m.pivot(index="classe_eva", columns="classe_vegestrate", values=share)
    order = list(dict.fromkeys(m["classe_eva"]))
    t = t.reindex(index=order, columns=order)
    t.index.name = "EVA \\ Vegestrate"
    return md_table(t.reset_index(), percent=order)


def km2_md(mat, level, version):
    m = mat[(mat["niveau"] == level) & (mat["version_vegestrate"] == version)]
    order = list(dict.fromkeys(m["classe_eva"]))
    t = m.pivot(index="classe_eva", columns="classe_vegestrate", values="surface_km2")
    t = t.reindex(index=order, columns=order)
    t.index.name = "EVA \\ Vegestrate (km²)"
    return md_table(t.reset_index())


def render(
    cfg, cubes, mat, metrics, rules, edge, patch, communes, lcz_t, excl, examples
):
    metric_cols = ["accord", "desaccord_quantite", "desaccord_allocation"]
    blocks = {}
    for level in cfg["levels"]:
        parts = []
        for version in cubes.level_versions(level):
            parts.append(f"**Version Vegestrate : {version}**\n")
            parts.append(km2_md(mat, level, version) + "\n")
            parts.append("Part de chaque classe EVA (lignes) :\n")
            parts.append(matrix_md(mat, level, version, "part_de_la_classe_eva") + "\n")
            parts.append("Part de chaque classe Vegestrate (colonnes) :\n")
            parts.append(
                matrix_md(mat, level, version, "part_de_la_classe_vegestrate") + "\n"
            )
        blocks[f"matrices_{level}"] = "\n".join(parts)
    rules_t = rules[
        [
            "code_eva",
            "version_vegestrate",
            "regle",
            "page_guide",
            "polygones",
            "part_polygones_conformes",
            "part_surface_conforme",
            "mediane",
        ]
    ]
    top = communes.sort_values("surface_km2", ascending=False)
    communes_t = top[
        [
            "commune",
            "surface_comparable_vegetal_km2",
            "part_vegetal_eva",
            "part_vegetal_vegestrate",
            "part_ligneux_eva",
            "part_ligneux_vegestrate",
        ]
    ]
    values = {
        **blocks,
        "indicateurs": md_table(
            metrics[
                ["niveau", "version_vegestrate", "surface_km2", *metric_cols, "kappa"]
            ],
            percent=metric_cols,
        ),
        "regles": md_table(
            rules_t,
            percent=["part_polygones_conformes", "part_surface_conforme", "mediane"],
        ),
        "bords": md_table(edge, percent=["part_vegestrate_positif"]),
        "taches": md_table(patch, percent=["part_petites_taches"]),
        "communes": md_table(
            communes_t,
            percent=[
                "part_vegetal_eva",
                "part_vegetal_vegestrate",
                "part_ligneux_eva",
                "part_ligneux_vegestrate",
            ],
        ),
        "lcz": md_table(
            lcz_t[["niveau", "lcz", "surface_km2", *metric_cols]], percent=metric_cols
        ),
        "exclusion": md_table(excl),
        "exclusion_km2": fmt(cubes.excluded["km2"].sum()),
        "eva_absente_km2": fmt(cubes.eva_absent_km2),
        "domaine_km2": fmt(cubes.zones["km2"].sum()),
        "exemples": examples,
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
        "coupes": md_table(
            pd.DataFrame(
                [
                    {
                        "variante": name,
                        "arbuste dès (m arrondi)": cut["shrub_min"],
                        "arbre dès (m arrondi)": cut["tree_min"],
                        "description": cut["label"],
                    }
                    for name, cut in cfg["height_cuts"].items()
                ]
            )
        ),
    }
    template = Template((Path(__file__).parent / "methodology_template.md").read_text())
    Path(cfg["paths"]["methodology"]).write_text(template.substitute(values))


def examples_md(cfg):
    path = output_path(cfg, "exemples_stats.csv")
    if not path.exists():
        return "_Rasters d'exemple non encore produits (`python -m audit.examples`)._"
    e = pd.read_csv(path)
    e = e[e["raster"] == "accord_strates"]
    t = e.pivot(index="categorie", columns="libelle_lieu", values="part").reset_index()
    return md_table(t, percent=list(t.columns[1:]))


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
    render(
        cfg,
        cubes,
        mat,
        metrics,
        rules,
        edge,
        patch,
        communes,
        lcz_t,
        excl,
        examples_md(cfg),
    )
    print(f"tables -> {cfg['paths']['output_dir']}, {cfg['paths']['methodology']}")


if __name__ == "__main__":
    main()
