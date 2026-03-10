import json
import math
from pathlib import Path

from osgeo import gdal
from osgeo import osr

from model.TmsZoneDef import TmsZoneDef


# ----------------------------------------------------------------------------
# Class TmsIntersector
# ----------------------------------------------------------------------------
class TmsIntersector:
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, zoneJsonDir):

        self._zones = {}
        
        # Load all JSON files in directory
        for jsonFile in TileDef.JSON_DIR.glob('*.json'):

            try:

                zoneDef = TmsZoneDef(jsonFile)
                self._zones[zoneDef.id] = zoneDef
                



                # with open(jsonFile, 'r') as f:
                #     zoneConfig = json.load(f)
                #
                # zoneId = zoneConfig['id']
                #
                # # Create spatial references
                # crs = osr.SpatialReference()
                # geoCrs = osr.SpatialReference()
                #
                # # Import CRS definitions
                # crs.ImportFromWkt(zoneConfig['crs'])
                # geoCrs.ImportFromWkt(zoneConfig['_geographic_crs'])
                #
                # # Set axis mapping to traditional GIS order (lon, lat) or (x, y)
                # crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                # geoCrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                #
                # self._zones[zoneId] = {
                #     'config': zoneConfig,
                #     'crs': crs,
                #     'geoCrs': geoCrs,
                #     'tileMatrices': {tm['id']: tm for tm in zoneConfig['tileMatrices']},
                #     'bbox': self._extractZoneBbox(zoneConfig['crs']),
                #     'transformer': None,
                #     'inverseTransformer': None
                # }
                
            except Exception as e:
                print(f'Error loading {jsonFile}: {e}')
    
    # ------------------------------------------------------------------------
    # extractZoneBbox
    # ------------------------------------------------------------------------
    # def _extractZoneBbox(self, crsWkt):
    #
    #     bboxStart = crsWkt.find('BBOX[') + 5
    #     bboxEnd = crsWkt.find(']', bboxStart)
    #     bboxParts = crsWkt[bboxStart:bboxEnd].split(',')
    #
    #     return {
    #         'minLat': float(bboxParts[0]),
    #         'minLon': float(bboxParts[1]),
    #         'maxLat': float(bboxParts[2]),
    #         'maxLon': float(bboxParts[3])
    #     }
    
    # ------------------------------------------------------------------------
    # getTransformer
    # ------------------------------------------------------------------------
    def _getTransformer(self, zoneId):

        zone = self.zones[zoneId]
        if zone['transformer'] is None:
            zone['transformer'] = osr.CoordinateTransformation(zone['geoCrs'], zone['crs'])
        return zone['transformer']
    
    # ------------------------------------------------------------------------
    # getInverseTransformer
    # ------------------------------------------------------------------------
    def _getInverseTransformer(self, zoneId):
        """
        Get or create inverse transformer for a zone (projected to geographic).
        
        Args:
            zoneId: Zone identifier
        
        Returns:
            osr.CoordinateTransformation object
        """
        zone = self.zones[zoneId]
        if zone['inverseTransformer'] is None:
            zone['inverseTransformer'] = osr.CoordinateTransformation(zone['crs'], zone['geoCrs'])
        return zone['inverseTransformer']
    
    # def _intersectsBbox(self, zoneId, ulLat, ulLon, lrLat, lrLon):
    #     """
    #     Check if the given bbox intersects a zone's coverage area.
    #
    #     Args:
    #         zoneId: Zone identifier
    #         ulLat: Upper-left latitude
    #         ulLon: Upper-left longitude
    #         lrLat: Lower-right latitude
    #         lrLon: Lower-right longitude
    #
    #     Returns:
    #         True if bbox intersects zone coverage
    #     """
    #     zoneBbox = self.zones[zoneId]['bbox']
    #
    #     queryMinLat = min(ulLat, lrLat)
    #     queryMaxLat = max(ulLat, lrLat)
    #     queryMinLon = min(ulLon, lrLon)
    #     queryMaxLon = max(ulLon, lrLon)
    #
    #     # Check for overlap with inclusive boundaries
    #     latOverlap = not (queryMaxLat < zoneBbox['minLat'] or
    #                      queryMinLat > zoneBbox['maxLat'])
    #     lonOverlap = not (queryMaxLon < zoneBbox['minLon'] or
    #                      queryMinLon > zoneBbox['maxLon'])
    #
    #     return latOverlap and lonOverlap
    
    def _getTileBoundsInternal(self, zoneId, col, row, tileMatrix):
        """
        Get tile bounds in geographic coordinates.
        
        Args:
            zoneId: Zone identifier
            col: Tile column index
            row: Tile row index
            tileMatrix: Tile matrix dictionary
        
        Returns:
            Dictionary with 'ulLat', 'ulLon', 'lrLat', 'lrLon'
        """
        originX, originY = tileMatrix['pointOfOrigin']
        cellSize = tileMatrix['cellSize']
        tileWidth = tileMatrix['tileWidth']
        
        tilePixelSize = cellSize * tileWidth
        
        # Calculate projected coordinates of tile corners
        minEasting = originX + col * tilePixelSize
        maxEasting = originX + (col + 1) * tilePixelSize
        maxNorthing = originY - row * tilePixelSize
        minNorthing = originY - (row + 1) * tilePixelSize
        
        # Transform back to geographic
        inverseTransformer = self._getInverseTransformer(zoneId)
        
        # Transform upper-left corner (minEasting, maxNorthing)
        # With OAMS_TRADITIONAL_GIS_ORDER: (x, y) = (lon, lat) for geographic
        ulLon, ulLat, _ = inverseTransformer.TransformPoint(minEasting, maxNorthing)
        
        # Transform lower-right corner (maxEasting, minNorthing)
        lrLon, lrLat, _ = inverseTransformer.TransformPoint(maxEasting, minNorthing)
        
        return {
            'ulLat': ulLat,
            'ulLon': ulLon,
            'lrLat': lrLat,
            'lrLon': lrLon
        }
    
    # def intersectOneZone(self, zoneId, ulLat, ulLon, lrLat, lrLon, zoomLevel, minOverlapMeters=10.0):
    #     """
    #     Find all tile IDs in a single zone that intersect the given bounding box.
    #     Includes a final geographic bounds check to filter tiles that don't
    #     actually overlap the query region (important for coarse zoom levels).
    #
    #     Args:
    #         zoneId: Zone identifier
    #         ulLat: Upper-left latitude (degrees)
    #         ulLon: Upper-left longitude (degrees)
    #         lrLat: Lower-right latitude (degrees)
    #         lrLon: Lower-right longitude (degrees)
    #         zoomLevel: Zoom level (as string or int, e.g., '5' or 5)
    #         minOverlapMeters: Minimum overlap in meters required to include a tile (default: 10.0)
    #
    #     Returns:
    #         List of tile IDs in format 'zoneId_zoom_col_row'
    #     """
    #     if zoneId not in self.zones:
    #         raise ValueError(f'Zone {zoneId} not found')
    #
    #     zone = self.zones[zoneId]
    #     zoomStr = str(zoomLevel)
    #
    #     # Get zoom level parameters
    #     if zoomStr not in zone['tileMatrices']:
    #         raise ValueError(f'Zoom level {zoomLevel} not found in zone {zoneId}')
    #
    #     tm = zone['tileMatrices'][zoomStr]
    #     originX, originY = tm['pointOfOrigin']
    #     cellSize = tm['cellSize']
    #     tileWidth = tm['tileWidth']
    #     tileHeight = tm['tileHeight']
    #     matrixWidth = tm['matrixWidth']
    #     matrixHeight = tm['matrixHeight']
    #
    #     # Calculate tile size in meters
    #     tilePixelSize = cellSize * tileWidth
    #
    #     # Clip bbox to zone coverage
    #     zoneBbox = zone['bbox']
    #     clippedMinLat = max(min(ulLat, lrLat), zoneBbox['minLat'])
    #     clippedMaxLat = min(max(ulLat, lrLat), zoneBbox['maxLat'])
    #     clippedMinLon = max(min(ulLon, lrLon), zoneBbox['minLon'])
    #     clippedMaxLon = min(max(ulLon, lrLon), zoneBbox['maxLon'])
    #
    #     # If no overlap after clipping, return empty
    #     if clippedMinLat > clippedMaxLat or clippedMinLon > clippedMaxLon:
    #         return []
    #
    #     # Get transformer
    #     transformer = self._getTransformer(zoneId)
    #
    #     # Transform all four corners of clipped bbox to projected coordinates
    #     # With OAMS_TRADITIONAL_GIS_ORDER: pass (lon, lat) for geographic coords
    #     ulX, ulY, _ = transformer.TransformPoint(clippedMinLon, clippedMaxLat)
    #     urX, urY, _ = transformer.TransformPoint(clippedMaxLon, clippedMaxLat)
    #     llX, llY, _ = transformer.TransformPoint(clippedMinLon, clippedMinLat)
    #     lrX, lrY, _ = transformer.TransformPoint(clippedMaxLon, clippedMinLat)
    #
    #     # Get bbox extents in projected space
    #     minEasting = min(ulX, urX, llX, lrX)
    #     maxEasting = max(ulX, urX, llX, lrX)
    #     minNorthing = min(ulY, urY, llY, lrY)
    #     maxNorthing = max(ulY, urY, llY, lrY)
    #
    #     # Store original query bbox for final filtering
    #     queryMinLat = min(ulLat, lrLat)
    #     queryMaxLat = max(ulLat, lrLat)
    #     queryMinLon = min(ulLon, lrLon)
    #     queryMaxLon = max(ulLon, lrLon)
    #
    #     # Find which tiles intersect this projected bbox
    #     tileIds = []
    #
    #     for row in range(matrixHeight):
    #         for col in range(matrixWidth):
    #             # Calculate this tile's bounds in projected coordinates
    #             tileMinX = originX + col * tilePixelSize
    #             tileMaxX = originX + (col + 1) * tilePixelSize
    #             tileMaxY = originY - row * tilePixelSize
    #             tileMinY = originY - (row + 1) * tilePixelSize
    #
    #             # Calculate overlap dimensions in projected space
    #             overlapMinX = max(tileMinX, minEasting)
    #             overlapMaxX = min(tileMaxX, maxEasting)
    #             overlapMinY = max(tileMinY, minNorthing)
    #             overlapMaxY = min(tileMaxY, maxNorthing)
    #
    #             # Calculate overlap width and height
    #             overlapWidth = max(0, overlapMaxX - overlapMinX)
    #             overlapHeight = max(0, overlapMaxY - overlapMinY)
    #
    #             # Check if both dimensions meet minimum overlap requirement
    #             if overlapWidth >= minOverlapMeters and overlapHeight >= minOverlapMeters:
    #                 # Final check: verify tile's geographic bounds overlap original query bbox
    #                 tileBounds = self._getTileBoundsInternal(zoneId, col, row, tm)
    #
    #                 # Check geographic overlap
    #                 tileMinLon = min(tileBounds['ulLon'], tileBounds['lrLon'])
    #                 tileMaxLon = max(tileBounds['ulLon'], tileBounds['lrLon'])
    #                 tileMinLat = min(tileBounds['ulLat'], tileBounds['lrLat'])
    #                 tileMaxLat = max(tileBounds['ulLat'], tileBounds['lrLat'])
    #
    #                 geoLatOverlap = not (tileMaxLat < queryMinLat or tileMinLat > queryMaxLat)
    #                 geoLonOverlap = not (tileMaxLon < queryMinLon or tileMinLon > queryMaxLon)
    #
    #                 if geoLatOverlap and geoLonOverlap:
    #                     tileId = f'{zoneId}_{zoomLevel}_{col}_{row}'
    #                     tileIds.append(tileId)
    #
    #     return tileIds
    
    # -----------------------------------------------------------------------
    # getTids
    # ------------------------------------------------------------------------
    def getTids(self, 
                ulLat, 
                ulLon, 
                lrLat, 
                lrLon, 
                zoomLevel, 
                minOverlapMeters=10.0):

        allTiles = []
        
        for zoneDef in self._zones:
            
            # Check if bbox intersects this zone.
            if zoneDef.intersectsBbox(ulLat, ulLon, lrLat, lrLon):
                
                # Get the zone's tiles that intersect the bbox.
                tiles = zoneDef.getIntersectingTiles(ulLat, 
                                                     ulLon, 
                                                     lrLat, 
                                                     lrLon, 
                                                     zoomLevel, 
                                                     minOverlapMeters)
                
                allTiles.extend(tiles)
        
        return allTiles
    
    # -----------------------------------------------------------------------
    # getTileBounds
    # ------------------------------------------------------------------------
    def getTileBounds(self, tileId):

        parts = tileId.split('_')
        if len(parts) < 4:
            raise ValueError(f'Invalid tile ID format: {tileId}')
        
        # Reconstruct zone ID (may contain underscores)
        zoomIdx = -3  # zoom is 3rd from end
        zoneId = '_'.join(parts[:zoomIdx])
        zoom = parts[zoomIdx]
        col = int(parts[-2])
        row = int(parts[-1])
        
        if zoneId not in self.zones:
            raise ValueError(f'Zone {zoneId} not loaded')
        
        zone = self.zones[zoneId]
        tm = zone['tileMatrices'][zoom]
        
        return self._getTileBoundsInternal(zoneId, col, row, tm)

