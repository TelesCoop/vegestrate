from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
import yaml
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import box

STRATA = ["aucune", "herbe", "arbuste", "arbre"]
N_NATIVE = 4
N_HBIN = 7
HBIN_UNKNOWN = 6
CHM_NODATA = -9999


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def output_path(cfg, name):
    return Path(cfg["paths"]["output_dir"]) / name


class EvaClasses:
    def __init__(self, cfg):
        self.groups = list(cfg["eva_groups"])
        self.codes = sorted(c for g in cfg["eva_groups"].values() for c in g["codes"])
        self.code_index = {c: i + 1 for i, c in enumerate(self.codes)}
        self.code_group = {
            c: g for g, spec in cfg["eva_groups"].items() for c in spec["codes"]
        }
        self.group_of_index = np.zeros(len(self.codes) + 1, dtype=np.uint8)
        for c, i in self.code_index.items():
            self.group_of_index[i] = self.groups.index(self.code_group[c]) + 1

    def group_mask(self, group_idx, names):
        wanted = [self.groups.index(n) + 1 for n in names]
        return np.isin(group_idx, wanted)


def pick(df, **filters):
    sel = df
    for column, value in filters.items():
        sel = sel[sel[column] == value]
    if sel.empty:
        raise KeyError(f"no row for {filters}")
    return sel


def zone_median_height(native, chm):
    out = np.full(native.shape, CHM_NODATA, dtype=np.int16)
    measured = chm != CHM_NODATA
    for cls in (2, 3):
        labels, n = ndi.label(native == cls)
        if n == 0:
            continue
        sel = measured & (labels > 0)
        lab, h = labels[sel], chm[sel]
        order = np.lexsort((h, lab))
        lab, h = lab[order], h[order]
        ids, first, counts = np.unique(lab, return_index=True, return_counts=True)
        mid = (h[first + (counts - 1) // 2] + h[first + counts // 2].astype(float)) / 2
        medians = np.full(n + 1, CHM_NODATA, dtype=np.int16)
        medians[ids] = np.trunc(mid)
        zone = labels > 0
        out[zone] = medians[labels[zone]]
    return out


HEIGHT_SOURCES = {"pixel": lambda native, chm: chm, "zone_median": zone_median_height}


def hbin(native, chm):
    woody = native >= 2
    unmeasured = (chm == CHM_NODATA) | (chm <= 0)
    h = np.where(unmeasured, HBIN_UNKNOWN, np.clip(chm, 0, 5))
    return np.where(woody, h, 0).astype(np.uint8)


def stratum(native, hbins, cut=None):
    if cut is None:
        return native.astype(np.uint8)
    recut = np.select([hbins >= cut["tree_min"], hbins >= cut["shrub_min"]], [3, 2], 1)
    keep_native = (native < 2) | (hbins == HBIN_UNKNOWN)
    return np.where(keep_native, native, recut).astype(np.uint8)


def pontius(matrix):
    p = np.asarray(matrix, dtype=float)
    p = p / p.sum()
    agreement = np.trace(p)
    quantity = 0.5 * np.abs(p.sum(axis=1) - p.sum(axis=0)).sum()
    disagreement = 1 - agreement
    expected = (p.sum(axis=1) * p.sum(axis=0)).sum()
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else np.nan
    return {
        "accord": agreement,
        "desaccord_quantite": quantity,
        "desaccord_allocation": disagreement - quantity,
        "kappa": kappa,
    }


def distance_bins(mask, edges_px):
    if mask.all():
        dist = np.full(mask.shape, np.inf)
    else:
        dist = ndi.distance_transform_edt(mask)
    return np.digitize(dist, edges_px, right=True).astype(np.uint8)


def small_patches(mask, min_px):
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    border = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    # shortcut: a patch touching the window border counts as large, so small patches longer than the halo are undercounted
    small = sizes < min_px
    small[border] = False
    small[0] = False
    return small[labels]


def burn(gdf, values, transform, shape, dtype):
    bounds = array_bounds(shape[0], shape[1], transform)
    hits = gdf.sindex.query(box(*bounds), predicate="intersects")
    if len(hits) == 0:
        return np.zeros(shape, dtype=dtype)
    geoms = shapely.clip_by_rect(gdf.geometry.values[hits], *bounds)
    return rasterize(
        zip(geoms, values[hits]),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=dtype,
    )


class Inputs:
    def __init__(self, cfg):
        self.cfg = cfg
        self.eva_classes = EvaClasses(cfg)
        paths = cfg["paths"]
        eva = pyogrio.read_dataframe(
            paths["eva"], layer=paths["eva_layer"], columns=["Id", "COD3"]
        )
        self.eva = eva.to_crs(cfg["crs"])
        self.eva_idx = np.arange(1, len(eva) + 1, dtype=np.uint32)
        self.cod3_index = np.zeros(len(eva) + 1, dtype=np.uint8)
        self.cod3_index[1:] = [self.eva_classes.code_index[c] for c in eva["COD3"]]
        zones = output_path(cfg, "zones.gpkg")
        self.iris = pyogrio.read_dataframe(zones, layer="iris")
        self.iris_idx = np.arange(1, len(self.iris) + 1, dtype=np.uint16)
        self.lcz = pyogrio.read_dataframe(zones, layer="lcz")
        self.lcz_codes = sorted(self.lcz["lcz_index"].unique())
        self.lcz_idx = np.array(
            [self.lcz_codes.index(c) + 1 for c in self.lcz["lcz_index"]],
            dtype=np.uint8,
        )
        self.exclusion = pyogrio.read_dataframe(output_path(cfg, "exclusion.gpkg"))
        self.exclusion_val = np.ones(len(self.exclusion), dtype=np.uint8)
        with rasterio.open(paths["vegestrate"]) as src:
            self.transform = src.transform
            self.height, self.width = src.height, src.width
        self.vegestrate = None
        self.chm = None

    def open(self):
        self.vegestrate = rasterio.open(self.cfg["paths"]["vegestrate"])
        self.chm = rasterio.open(self.cfg["paths"]["chm"])

    def window(self, row, col, height, width):
        if self.vegestrate is None:
            self.open()
        win = Window(col, row, width, height)
        transform = rasterio.windows.transform(win, self.transform)
        shape = (height, width)
        native = self.vegestrate.read(1, window=win)
        # shortcut: zones are cut at the window edge (block + halo), so a zone larger than the window gets the mean of its visible part
        chm = HEIGHT_SOURCES[self.cfg["height_source"]](
            native, self.chm.read(1, window=win)
        )
        eva = burn(self.eva, self.eva_idx, transform, shape, np.uint32)
        cod3 = self.cod3_index[eva]
        return {
            "transform": transform,
            "native": native,
            "hbin": hbin(native, chm),
            "eva": eva,
            "cod3": cod3,
            "group": self.eva_classes.group_of_index[cod3],
            "iris": burn(self.iris, self.iris_idx, transform, shape, np.uint16),
            "lcz": burn(self.lcz, self.lcz_idx, transform, shape, np.uint8),
            "excluded": burn(
                self.exclusion, self.exclusion_val, transform, shape, np.uint8
            ),
        }

    def grid_window(self, center, size_m):
        res = self.transform.a
        half = size_m / 2
        col = int(round((center[0] - half - self.transform.c) / res))
        row = int(round((self.transform.f - (center[1] + half)) / -self.transform.e))
        n = int(round(size_m / res))
        return row, col, n, n
