
from typing import List

from osgeo import osr

from model.TmsTileDef import TmsTileDef


# ----------------------------------------------------------------------------
# Class TmsZoneDef
# ----------------------------------------------------------------------------
class TmsZoneDef:
    
    CRS = 'crs'
    TILE_MATRICES = 'tileMatrices'
    ZONE_ID = 'id'

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, zoneFile: Path):
        
        with open(zoneFile, 'r') as f:
            self._zoneDef: dict = json.load(f)
            
        self._srs = osr.SpatialReference()
        self._srs.ImportFromWkt(self._zoneDef[TmsZoneDef.CRS])
        
        self._zone = int(zoneFile.name.split('_')[2][:2])
        
    # ------------------------------------------------------------------------
    # getIntersectingTiles
    # ------------------------------------------------------------------------
    def getIntersectingTiles(self, 
                             ulLat: float, 
                             ulLon: float, 
                             lrLat: float, 
                             lrLon: float, 
                             zoomLevel: int, 
                             minOverlapMeters: float=10.0):

        zoomLevels: list = tms[TmsZoneDef.TILE_MATRICES]

        tileJson = [zl for zl in zoomLevels \
            if int(zl[TileDef.ID]) == zoomLevel][0]

        tileDef = TmsTileDef(tileJson, self.zone, zoomLevel)

        # Calculate tile size in meters
        tilePixelSize = tileDef.cellSize * tileDef.tileWidth

        # Clip bbox to zone coverage
        zMinLat, zMinLon, zMaxLat, zMaxLon = self.zoneBbox()

        clippedMinLat = max(lrLat, zMinLat)
        clippedMaxLat = min(ulLat, zMaxLat)
        clippedMinLon = max(ulLon, zMinLon)
        clippedMaxLon = min(lrLon, zMaxLon)

        # If no overlap after clipping, return empty
        if clippedMinLat > clippedMaxLat or clippedMinLon > clippedMaxLon:
            return []





        # tm = zone['tileMatrices'][zoomStr]
        # originX, originY = tm['pointOfOrigin']
        # cellSize = tm['cellSize']
        # tileWidth = tm['tileWidth']
        # tileHeight = tm['tileHeight']
        # matrixWidth = tm['matrixWidth']
        # matrixHeight = tm['matrixHeight']
        
        # Calculate tile size in meters
        # tilePixelSize = tileDef.cellSize * tileDef.tileWidth
        
        # Clip bbox to zone coverage
        # zoneBbox = zone['bbox']
        # clippedMinLat = max(min(ulLat, lrLat), zoneBbox['minLat'])
        # clippedMaxLat = min(max(ulLat, lrLat), zoneBbox['maxLat'])
        # clippedMinLon = max(min(ulLon, lrLon), zoneBbox['minLon'])
        # clippedMaxLon = min(max(ulLon, lrLon), zoneBbox['maxLon'])
        
        # If no overlap after clipping, return empty
        # if clippedMinLat > clippedMaxLat or clippedMinLon > clippedMaxLon:
        #     return []
        
        # Get transformer
        transformer = self._getTransformer(zoneId)
        
        # Transform all four corners of clipped bbox to projected coordinates
        # With OAMS_TRADITIONAL_GIS_ORDER: pass (lon, lat) for geographic coords
        ulX, ulY, _ = transformer.TransformPoint(clippedMinLon, clippedMaxLat)
        urX, urY, _ = transformer.TransformPoint(clippedMaxLon, clippedMaxLat)
        llX, llY, _ = transformer.TransformPoint(clippedMinLon, clippedMinLat)
        lrX, lrY, _ = transformer.TransformPoint(clippedMaxLon, clippedMinLat)
        
        # Get bbox extents in projected space
        minEasting = min(ulX, urX, llX, lrX)
        maxEasting = max(ulX, urX, llX, lrX)
        minNorthing = min(ulY, urY, llY, lrY)
        maxNorthing = max(ulY, urY, llY, lrY)
        
        # Store original query bbox for final filtering
        queryMinLat = min(ulLat, lrLat)
        queryMaxLat = max(ulLat, lrLat)
        queryMinLon = min(ulLon, lrLon)
        queryMaxLon = max(ulLon, lrLon)
        
        # Find which tiles intersect this projected bbox
        tileIds = []
        
        for row in range(matrixHeight):
            for col in range(matrixWidth):
                # Calculate this tile's bounds in projected coordinates
                tileMinX = originX + col * tilePixelSize
                tileMaxX = originX + (col + 1) * tilePixelSize
                tileMaxY = originY - row * tilePixelSize
                tileMinY = originY - (row + 1) * tilePixelSize
                
                # Calculate overlap dimensions in projected space
                overlapMinX = max(tileMinX, minEasting)
                overlapMaxX = min(tileMaxX, maxEasting)
                overlapMinY = max(tileMinY, minNorthing)
                overlapMaxY = min(tileMaxY, maxNorthing)
                
                # Calculate overlap width and height
                overlapWidth = max(0, overlapMaxX - overlapMinX)
                overlapHeight = max(0, overlapMaxY - overlapMinY)
                
                # Check if both dimensions meet minimum overlap requirement
                if overlapWidth >= minOverlapMeters and overlapHeight >= minOverlapMeters:
                    # Final check: verify tile's geographic bounds overlap original query bbox
                    tileBounds = self._getTileBoundsInternal(zoneId, col, row, tm)
                    
                    # Check geographic overlap
                    tileMinLon = min(tileBounds['ulLon'], tileBounds['lrLon'])
                    tileMaxLon = max(tileBounds['ulLon'], tileBounds['lrLon'])
                    tileMinLat = min(tileBounds['ulLat'], tileBounds['lrLat'])
                    tileMaxLat = max(tileBounds['ulLat'], tileBounds['lrLat'])
                    
                    geoLatOverlap = not (tileMaxLat < queryMinLat or tileMinLat > queryMaxLat)
                    geoLonOverlap = not (tileMaxLon < queryMinLon or tileMinLon > queryMaxLon)
                    
                    if geoLatOverlap and geoLonOverlap:
                        tileId = f'{zoneId}_{zoomLevel}_{col}_{row}'
                        tileIds.append(tileId)
        
        return tileIds

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
    def intersectsBbox(self, zoneId, ulLat, ulLon, lrLat, lrLon) -> bool:

        zMinLat, zMinLon, zMaxLat, zMaxLon = self.zoneBbox()
        
        # Check for overlap with inclusive boundaries
        latOverlaps = (ulLat >= zMinLat and lrLat <= zMaxLat)
        lonOverlaps = (lrLon >= zMinLon and ulLon <= zMaxLon)
            
        return latOverlaps and lonOverlaps

    # -----------------------------------------------------------------------
    # srs
    # ------------------------------------------------------------------------
    @property
    def srs(self) -> str:
        
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
    
