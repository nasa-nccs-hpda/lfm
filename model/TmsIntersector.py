
from pathlib import Path
from typing import List

from lfm.model.TmsTileDef import TmsTileDef
from lfm.model.TmsZoneDef import TmsZoneDef


# ----------------------------------------------------------------------------
# Class TmsIntersector
# ----------------------------------------------------------------------------
class TmsIntersector:

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):

        self._zones = {}

        # Load all zone JSON files
        for jsonFile in TmsTileDef.JSON_DIR.glob('*.json'):

            try:

                zoneDef = TmsZoneDef(jsonFile)
                self._zones[zoneDef.zone] = zoneDef

            except Exception as e:
                print(f'Error loading {jsonFile}: {e}')

    # ------------------------------------------------------------------------
    # getTids
    # ------------------------------------------------------------------------
    def getTids(self,
                ulLat: float,
                ulLon: float,
                lrLat: float,
                lrLon: float,
                zoomLevel: int,
                minOverlapMeters: float = 10.0) -> List[str]:

        '''
        This returns all tiles in all zones that intersect the input bounding
        box.
        '''

        allTiles = []

        for zone, zoneDef in self._zones.items():

            # Check if bbox intersects this zone
            if zoneDef.intersectsBbox(ulLat, ulLon, lrLat, lrLon):

                print('Found intersection:', zone)

                # Get overlapping tile indices
                indices = zoneDef.getIntersectingTiles(
                    ulLat, ulLon, lrLat, lrLon, zoomLevel, minOverlapMeters)

                # Convert indices to tile IDs
                for col, row in indices:

                    tid = {'tileX': col,
                           'tileY': row,
                           'zone': zone,
                           'zoomLevel': zoomLevel}

                    allTiles.append(tid)

        return allTiles

    # -----------------------------------------------------------------------
    # zones
    # ------------------------------------------------------------------------
    @property
    def zones(self) -> str:

        return self._zones

