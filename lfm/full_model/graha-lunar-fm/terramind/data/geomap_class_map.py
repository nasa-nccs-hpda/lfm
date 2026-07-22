"""Geomap class reduction mapping from DN values to time-type classes.

The geomap data has 49 geological classes with DN values 1-49, plus index 0 for background/no-data.
This gives us 50 total indices (0-49) that need to be mapped to 16 classes (15 geological + 1 background).

Source: Unified_Geologic_Map_of_the_Moon_RASTER/Dn_GeologicUnit_RGBcolor.csv
"""

# The 16 target classes: 15 time-type geological classes + 1 background class
TARGET_CLASSES = [
    "Cc",  # 0: Copernican crater
    "EIt",  # 1: Eratosthenian-Imbrian terra
    "Ec",  # 2: Eratosthenian crater
    "Em",  # 3: Eratosthenian mare
    "INt",  # 4: Imbrian-Nectarian terra
    "Ib",  # 5: Imbrian basin
    "Ic",  # 6: Imbrian crater
    "Im",  # 7: Imbrian mare
    "It",  # 8: Imbrian terra
    "Nb",  # 9: Nectarian basin
    "Nc",  # 10: Nectarian crater
    "Nt",  # 11: Nectarian terra
    "PNb",  # 12: Pre-Nectarian basin
    "PNc",  # 13: Pre-Nectarian crater
    "PNt",  # 14: Pre-Nectarian terra
    "Background",  # 15: Background/no-data
]

# Mapping from geomap DN values (0-49) to 16 reduced class indices
# DN 0 is background/no-data -> class 15
# DN 1-49 are geological units -> classes 0-14
MAPPING_GEOMAP_CLASSES = {
    0: 15,  # Background/no-data -> Background class
    1: 0,  # DN 1: Cc - Crater Unit, Copernican
    2: 0,  # DN 2: Ccc - Crater Cluster Unit, Copernican
    3: 0,  # DN 3: Csc - Secondary Crater Unit, Copernican
    4: 2,  # DN 4: Ec - Crater Unit, Eratosthenian
    5: 2,  # DN 5: Ecc - Crater Cluster Unit, Eratosthenian
    6: 1,  # DN 6: EIp - Plateau Unit, Eratosthenian-Imbrian
    7: 3,  # DN 7: Em - Mare Unit, Eratosthenian
    8: 2,  # DN 8: Esc - Secondary Crater Unit, Eratosthenian
    9: 5,  # DN 9: Ib - Basin Undivided Unit, Imbrian
    10: 5,  # DN 10: Ibm - Basin Massif Unit, Imbrian
    11: 6,  # DN 11: Ic - Crater Undivided Unit, Imbrian
    12: 6,  # DN 12: Ic1 - Lower Crater Unit, Imbrian
    13: 6,  # DN 13: Ic2 - Upper Crater Unit, Imbrian
    14: 6,  # DN 14: Icc - Crater Cluster Unit, Imbrian
    15: 6,  # DN 15: Icf - Crater Fracture Floor Unit, Imbrian
    16: 8,  # DN 16: Id - Dark Mantling Unit, Imbrian
    17: 8,  # DN 17: Ig - Grooved Terrain Unit, Imbrian
    18: 5,  # DN 18: Iia - Imbrium Alpes Formation Unit, Imbrian
    19: 5,  # DN 19: Iiap - Imbrium Apenninus Formation Unit, Imbrian
    20: 6,  # DN 20: Iic - Imbrium Crater Unit, Imbrian
    21: 5,  # DN 21: Iif - Imbrium Fra Mauro Formation Unit, Imbrian
    22: 7,  # DN 22: Im1 - Lower Mare Unit, Imbrian
    23: 7,  # DN 23: Im2 - Upper Mare Unit, Imbrian
    24: 7,  # DN 24: Imd - Mare Dome Unit, Imbrian
    25: 4,  # DN 25: INp - Plains Unit, Imbrian-Nectarian
    26: 4,  # DN 26: INt - Terra Unit, Imbrian-Nectarian
    27: 5,  # DN 27: Iohi - Orientale Hevelius Formation, Inner Facies Unit, Imbrian
    28: 5,  # DN 28: Ioho - Orientale Hevelius Formation, Inner Facies Unit, Imbrian
    29: 6,  # DN 29: Iohs - Orientale Hevelius Formation, Secondary Crater Facies Unit, Imbrian
    30: 5,  # DN 30: Iom - Orientale Maunder Formation Unit, Imbrian
    31: 5,  # DN 31: Iork - Orientale Montes Rook Formation, Knobby Facies Unit, Imbrian
    32: 5,  # DN 32: Iorm - Orientale Montes Rook Formation, Massif Facies Unit, Imbrian
    33: 8,  # DN 33: Ip - Plains Unit, Imbrian
    34: 6,  # DN 34: Isc - Secondary Crater Unit, Imbrian
    35: 8,  # DN 35: It - Terra Unit, Imbrian
    36: 8,  # DN 36: Itd - Terra Dome Unit, Imbrian
    37: 9,  # DN 37: Nb - Basin Undivided Unit, Nectarian
    38: 9,  # DN 38: Nbl - Basin Lineated Unit, Nectarian
    39: 9,  # DN 39: Nbm - Basin Massif Unit, Nectarian
    40: 10,  # DN 40: Nbsc - Basin Secondary Crater Unit, Nectarian
    41: 10,  # DN 41: Nc - Crater Unit, Nectarian
    42: 9,  # DN 42: Nnj - Nectaris Janssen Formation Unit, Nectarian
    43: 11,  # DN 43: Np - Plains Unit, Nectarian
    44: 11,  # DN 44: Nt - Terra Unit, Nectarian
    45: 11,  # DN 45: Ntp - Plains and Mantling, Terra Unit, Nectarian
    46: 12,  # DN 46: pNb - Basin Undivided Unit, Pre-Nectarian
    47: 12,  # DN 47: pNbm - Basin Massif Unit, Pre-Nectarian
    48: 13,  # DN 48: pNc - Crater Unit, Pre-Nectarian
    49: 14,  # DN 49: pNt - Terra Unit, Pre-Nectarian
}


def get_mapping():
    return MAPPING_GEOMAP_CLASSES.copy()


def get_target_classes():
    return TARGET_CLASSES.copy()
