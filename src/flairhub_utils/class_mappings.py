import numpy as np

# Original 19 FLAIR-HUB classes
FLAIR_CLASSES = {
    0: "building",
    1: "greenhouse",
    2: "swimming_pool",
    3: "impervious surface",
    4: "pervious surface",
    5: "bare soil",
    6: "water",
    7: "snow",
    8: "herbaceous vegetation",
    9: "agricultural land",
    10: "plowed land",
    11: "vineyard",
    12: "deciduous",
    13: "coniferous",
    14: "brushwood",
    15: "clear cut",
    16: "ligneous",
    17: "mixed",
    18: "undefined",
}

# Simplified 4-class system matching our dataset
SIMPLIFIED_CLASSES = {
    0: "else",  # Everything else (buildings, roads, water, soil, etc.)
    1: "herbaceous",  # Grass and herbaceous vegetation
    2: "hedge",  # Bushes, shrubs, hedges
    3: "trees",  # Trees (deciduous, coniferous, mixed)
}

# Mapping from 19 FLAIR classes to 4 simplified classes
CLASS_REMAP_19_TO_4 = {
    0: 0,  # building -> else
    1: 0,  # greenhouse -> else
    2: 0,  # swimming_pool -> else
    3: 0,  # impervious surface -> else
    4: 0,  # pervious surface -> else
    5: 0,  # bare soil -> else
    6: 0,  # water -> else
    7: 0,  # snow -> else
    8: 1,  # herbaceous vegetation -> herbaceous
    9: 1,  # agricultural land -> herbaceous
    10: 1,  # plowed land -> herbaceous
    11: 0,  # vineyard -> else
    12: 3,  # deciduous -> trees
    13: 3,  # coniferous -> trees
    14: 2,  # brushwood -> hedge
    15: 0,  # clear cut -> else
    16: 3,  # ligneous -> trees
    17: 3,  # mixed -> trees
    18: 0,  # undefined -> else
}


def remap_to_4_classes(class_map_19: np.ndarray) -> np.ndarray:
    """
    Remap 19-class predictions to 4 simplified classes.

    Args:
        class_map_19: Class map with values 0-18 (19 FLAIR classes)

    Returns:
        Class map with values 0-3 (4 simplified classes)
    """
    class_map_4 = np.zeros_like(class_map_19, dtype=np.uint8)

    for old_class, new_class in CLASS_REMAP_19_TO_4.items():
        class_map_4[class_map_19 == old_class] = new_class

    return class_map_4


def get_class_name(class_id: int, simplified: bool = True) -> str:
    """
    Get class name from ID.

    Args:
        class_id: Class ID
        simplified: If True, use simplified classes (default: True)

    Returns:
        Class name
    """
    if simplified:
        return SIMPLIFIED_CLASSES.get(class_id, "unknown")
    else:
        return FLAIR_CLASSES.get(class_id, "unknown")


def list_classes(simplified: bool = True) -> dict[int, str]:
    """
    List all available classes.

    Args:
        simplified: If True, return simplified classes (default: True)

    Returns:
        Dictionary mapping class IDs to names
    """
    if simplified:
        return SIMPLIFIED_CLASSES.copy()
    else:
        return FLAIR_CLASSES.copy()
