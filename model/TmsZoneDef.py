from pathlib import Path
from typing import List
import json

from osgeo import osr

from lfm.model.TmsTileDef import TmsTileDef


# ----------------------------------------------------------------------------
# Class TmsZoneDef
# ----------------------------------------------------------------------------
class TmsZoneDef:
    
    CRS = 'crs'
    ZONE_ID = 'id'

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, zoneFile: Path):
        
        with open(zoneFile, 'r') as f:
            self._zoneDef: dict = json.load(f)
            
        self._srs = osr.SpatialReference()
        self._srs.ImportFromWkt(self._zoneDef[TmsZoneDef.CRS])
        
        self._zone = zoneFile.stem.split('_')[2][:3]
        self._tileDefs = {}  # Cache tile definitions by zoom level
        
    # ------------------------------------------------------------------------
    # getIntersectingTiles
    # ------------------------------------------------------------------------
    def getIntersectingTiles(self,
                             ulLat: float,
                             ulLon: float,
                             lrLat: float,
                             lrLon: float,
                             zoomLevel: int,
                             minOverlapMeters: float=10.0) -> List[tuple]:
        
        # Get or create tile definition for this zoom level
        tileDef = self.getTileDef(zoomLevel)
        
        # Use TmsTileDef's method to find overlapping tiles
        return tileDef.getOverlappingTiles(ulLat, 
                                           ulLon, 
                                           lrLat, 
                                           lrLon, 
                                           minOverlapMeters)

    # ------------------------------------------------------------------------
    # getTileDef
    # ------------------------------------------------------------------------
    def getTileDef(self, zoomLevel: int) -> TmsTileDef:
        
        # Check cache first
        if zoomLevel not in self._tileDefs:
            
            # Create and cache the tile definition
            self._tileDefs[zoomLevel] = \
                TmsTileDef.initFromParams(self._zone, zoomLevel)
        
        return self._tileDefs[zoomLevel]
        
    # ------------------------------------------------------------------------
    # id
    # ------------------------------------------------------------------------
    @property
    def id(self) -> str:
        
        return self._zoneDef[TmsZoneDef.ZONE_ID]
        
    # ------------------------------------------------------------------------
    # intersectsBbox
    #
    # This manual intersection is supposed to be more efficient than asking
    # GDAL to do it.
    # ------------------------------------------------------------------------
    def intersectsBbox(self, 
                       ulLat: float, 
                       ulLon: float, 
                       lrLat: float, 
                       lrLon: float) -> bool:

        zMinLat, zMinLon, zMaxLat, zMaxLon = self.zoneBbox()
        
        # Check for overlap with inclusive boundaries
        latOverlaps = (ulLat >= zMinLat and lrLat <= zMaxLat)
        lonOverlaps = (lrLon >= zMinLon and ulLon <= zMaxLon)
            
        return latOverlaps and lonOverlaps

    # -----------------------------------------------------------------------
    # srs
    # ------------------------------------------------------------------------
    @property
    def srs(self) -> osr.SpatialReference:
        
        return self._srs

    # -----------------------------------------------------------------------
    # zone
    # ------------------------------------------------------------------------
    @property
    def zone(self) -> str:
        
        return self._zone

    # ------------------------------------------------------------------------
    # zoneBbox
    # ------------------------------------------------------------------------
    def zoneBbox(self) -> List[float]:

        aou = self.srs.GetAreaOfUse()

        minLat = aou.south_lat_degree
        minLon = aou.west_lon_degree
        maxLat = aou.north_lat_degree
        maxLon = aou.east_lon_degree
        
        return minLat, minLon, maxLat, maxLon