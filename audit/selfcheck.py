import numpy as np
import pandas as pd

from audit.figures import colorize
from audit.common import (
    CHM_NODATA,
    HBIN_UNKNOWN,
    distance_bins,
    hbin,
    pick,
    pontius,
    small_patches,
    stratum,
    zone_median_height,
)


def check_pontius():
    m = pontius([[40, 10], [0, 50]])
    assert np.isclose(m["accord"], 0.9)
    assert np.isclose(m["desaccord_quantite"], 0.1)
    assert np.isclose(m["desaccord_allocation"], 0.0)
    m = pontius([[40, 10], [10, 40]])
    assert np.isclose(m["desaccord_quantite"], 0.0)
    assert np.isclose(m["desaccord_allocation"], 0.2)
    assert np.isclose(m["kappa"], 0.6)


def check_stratum():
    native = np.array([0, 1, 1, 2, 2, 3, 3, 3, 3])
    hbins = np.array([0, 0, 0, 0, 2, 5, 3, 1, HBIN_UNKNOWN])
    cut = {"shrub_min": 1, "tree_min": 4}
    assert stratum(native, hbins).tolist() == native.tolist()
    assert stratum(native, hbins, cut).tolist() == [0, 1, 1, 1, 2, 3, 2, 2, 3]


def check_hbin():
    native = np.array([0, 1, 2, 2, 3, 3, 3])
    chm = np.array([0, 0, 0, 2, 0, 9, CHM_NODATA])
    unknown = HBIN_UNKNOWN
    assert hbin(native, chm).tolist() == [0, 0, unknown, 2, unknown, 5, unknown]


def check_zone_median_height():
    native = np.array([[3, 3, 3, 3, 0, 2, 2], [0, 0, 0, 0, 0, 0, 3]])
    chm = np.array([[20, 1, 18, CHM_NODATA, 0, 2, 5], [0, 0, 0, 0, 0, 0, CHM_NODATA]])
    zone = zone_median_height(native, chm)
    assert zone[0].tolist() == [18, 18, 18, 18, CHM_NODATA, 3, 3]
    assert zone[1, 6] == CHM_NODATA


def check_distance_bins():
    mask = np.zeros((1, 12), dtype=bool)
    mask[0, 1:] = True
    bins = distance_bins(mask, np.array([1, 2, 5]))
    assert bins[0, 1:].tolist() == [0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    assert (distance_bins(np.ones((3, 3), dtype=bool), np.array([1])) == 1).all()


def check_small_patches():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    mask[6:9, 6:9] = True
    mask[0, 7:] = True
    small = small_patches(mask, 5)
    assert small[2:4, 2:4].all()
    assert not small[6:9, 6:9].any()
    assert not small[0, 7:].any()


def check_colorize():
    cats = [(1, "a", "#ff0000"), (2, "b", "#00ff00"), (3, "c", "#0000ff")]
    rgba = colorize(np.array([[1, 2, 3, 255]]), cats, nodata=255, transparent={3})
    assert rgba[0, 0].tolist() == [255, 0, 0, 255]
    assert rgba[0, 1].tolist() == [0, 255, 0, 255]
    assert rgba[0, 2, 3] == 0
    assert rgba[0, 3, 3] == 0


def check_pick():
    df = pd.DataFrame({"niveau": ["vegetation"], "v": [1]})
    assert pick(df, niveau="vegetation")["v"].tolist() == [1]
    try:
        pick(df, niveau="strates")
    except KeyError as e:
        assert "strates" in str(e)
    else:
        raise AssertionError


if __name__ == "__main__":
    check_colorize()
    check_pick()
    check_pontius()
    check_stratum()
    check_hbin()
    check_zone_median_height()
    check_distance_bins()
    check_small_patches()
    print("selfcheck ok")
