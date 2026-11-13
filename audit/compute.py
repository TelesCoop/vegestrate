import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from shapely.geometry import box

from audit.common import (
    N_HBIN,
    N_NATIVE,
    Inputs,
    distance_bins,
    load_config,
    output_path,
    small_patches,
    stratum,
)

STATE = {}
MAX_ATTEMPTS = 3


def dims(inputs):
    return {
        "exclu": 2,
        "iris": len(inputs.iris) + 1,
        "lcz": len(inputs.lcz_codes) + 1,
        "cod3": len(inputs.eva_classes.codes) + 1,
        "natif": N_NATIVE,
        "hbin": N_HBIN,
    }


def ravel(arrays, sizes):
    key = np.zeros(arrays[0].shape, dtype=np.int64)
    for a, n in zip(arrays, sizes):
        key = key * n + a
    return key


def tally(key):
    keys, counts = np.unique(key, return_counts=True)
    return keys, counts


def edge_tally(valid, positive, comparable, agree, edges_px):
    out = []
    for side, mask in ((1, positive), (0, comparable & ~positive)):
        bins = distance_bins(positive if side else ~positive, edges_px)
        sel = valid & mask
        out.append(
            tally(ravel([np.full(sel.sum(), side), bins[sel], agree[sel]], [2, 8, 2]))
        )
    return out


def process(block):
    inputs, cfg = STATE["inputs"], STATE["cfg"]
    r0, c0 = block
    bs, halo = cfg["processing"]["block_size"], cfg["processing"]["halo_px"]
    h, w = min(bs, inputs.height - r0), min(bs, inputs.width - c0)
    wr0, wc0 = max(0, r0 - halo), max(0, c0 - halo)
    wr1, wc1 = min(inputs.height, r0 + h + halo), min(inputs.width, c0 + w + halo)
    t = inputs.transform
    block_box = box(
        t.c + c0 * t.a, t.f + (r0 + h) * t.e, t.c + (c0 + w) * t.a, t.f + r0 * t.e
    )
    if len(inputs.iris.sindex.query(block_box, predicate="intersects")) == 0:
        return None
    d = inputs.window(wr0, wc0, wr1 - wr0, wc1 - wc0)
    core = np.zeros(d["native"].shape, dtype=bool)
    core[r0 - wr0 : r0 - wr0 + h, c0 - wc0 : c0 - wc0 + w] = True
    domain = core & (d["iris"] > 0)
    if not domain.any():
        return None
    sizes = STATE["dims"]
    zone_key = ravel(
        [d["excluded"], d["iris"], d["lcz"], d["cod3"], d["native"], d["hbin"]],
        list(sizes.values()),
    )[domain]
    valid = domain & (d["excluded"] == 0)
    poly_sel = valid & (d["eva"] > 0)
    poly_key = ravel(
        [d["eva"][poly_sel], d["native"][poly_sel], d["hbin"][poly_sel]],
        [STATE["n_eva"], N_NATIVE, N_HBIN],
    )
    ec = inputs.eva_classes
    groups = cfg["levels"]
    veg = stratum(d["native"], d["hbin"], cfg["height_cuts"]["principal"])
    edges_px = np.array(cfg["edge_distance_bins_m"]) / inputs.transform.a
    eva_veg = ec.group_mask(d["group"], groups["vegetation"]["eva"]["vegetal"])
    eva_veg_cmp = eva_veg | ec.group_mask(
        d["group"], groups["vegetation"]["eva"]["non_vegetal"]
    )
    eva_woody = ec.group_mask(d["group"], groups["ligneux_herbace"]["eva"]["ligneux"])
    edge = {
        "vegetation": edge_tally(valid, eva_veg, eva_veg_cmp, veg > 0, edges_px),
        "ligneux": edge_tally(valid, eva_woody, eva_veg_cmp, veg >= 2, edges_px),
    }
    min_px = cfg["small_patch_area_m2"] / inputs.transform.a**2
    patches = {}
    for name, mask in (("vegetation", veg > 0), ("ligneux", veg >= 2)):
        small = small_patches(mask, min_px)
        sel = valid & mask
        patches[name] = tally(ravel([small[sel], d["group"][sel]], [2, 16]))
    return tally(zone_key), tally(poly_key), edge, patches


def cube_specs():
    sizes = STATE["dims"]
    specs = {
        "zones": (list(sizes), list(sizes.values()), "cube_zones.csv.gz"),
        "polygones": (
            ["eva", "natif", "hbin"],
            [STATE["n_eva"], N_NATIVE, N_HBIN],
            "cube_polygones.csv.gz",
        ),
    }
    for level in ("vegetation", "ligneux"):
        specs[f"bord_{level}"] = (
            ["interieur", "classe_distance", "accord"],
            [2, 8, 2],
            f"cube_bord_{level}.csv",
        )
        specs[f"taches_{level}"] = (
            ["petite", "groupe"],
            [2, 16],
            f"cube_taches_{level}.csv",
        )
    return specs


def to_frame(acc, names, sizes):
    keys = np.flatnonzero(acc)
    df = pd.DataFrame(dict(zip(names, np.unravel_index(keys, sizes))))
    df["pixels"] = acc[keys]
    return df


def accumulate(store, res):
    if res is None:
        return
    zone, poly, edge, patches = res
    tallies = [("zones", zone), ("polygones", poly)]
    tallies += [(f"bord_{lv}", side) for lv, sides in edge.items() for side in sides]
    tallies += [(f"taches_{lv}", tal) for lv, tal in patches.items()]
    for name, (keys, counts) in tallies:
        store[name][keys] += counts


def main():
    parser = argparse.ArgumentParser(description="Comptages pixel EVA x Végéstrate")
    parser.add_argument("--config", default="audit/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    inputs = Inputs(cfg)
    STATE.update(inputs=inputs, cfg=cfg, dims=dims(inputs), n_eva=len(inputs.eva) + 1)
    bs = cfg["processing"]["block_size"]
    xmin, ymin, xmax, ymax = inputs.iris.total_bounds
    t = inputs.transform
    c_lo, c_hi = int((xmin - t.c) / t.a) // bs * bs, int((xmax - t.c) / t.a)
    r_lo, r_hi = int((t.f - ymax) / -t.e) // bs * bs, int((t.f - ymin) / -t.e)
    blocks = [
        (r, c)
        for r in range(max(0, r_lo), min(inputs.height, r_hi + 1), bs)
        for c in range(max(0, c_lo), min(inputs.width, c_hi + 1), bs)
    ]
    blocks = blocks[: args.limit]
    print(f"{len(blocks)} blocks")
    specs = cube_specs()
    store = {
        name: np.zeros(np.prod(sizes), dtype=np.int64)
        for name, (_, sizes, _) in specs.items()
    }
    start = time.time()
    pending = set(blocks)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with ProcessPoolExecutor(cfg["processing"]["workers"]) as pool:
                futures = {pool.submit(process, b): b for b in pending}
                for fut in as_completed(futures):
                    accumulate(store, fut.result())
                    pending.discard(futures[fut])
                    done = len(blocks) - len(pending)
                    if done % 50 == 0 or not pending:
                        el = time.time() - start
                        print(
                            f"{done}/{len(blocks)} blocks, {el / 60:.1f} min",
                            flush=True,
                        )
        except BrokenProcessPool:
            print(
                f"worker died (attempt {attempt}), {len(pending)} blocks left",
                flush=True,
            )
        if not pending:
            break
    if pending:
        raise RuntimeError(
            f"{len(pending)} blocks failed after {MAX_ATTEMPTS} attempts"
        )
    for name, (names, sizes, filename) in specs.items():
        to_frame(store[name], names, sizes).to_csv(
            output_path(cfg, filename), index=False
        )
    print(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
