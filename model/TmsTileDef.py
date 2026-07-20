
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List
from typing import Tuple

from osgeo import osr


# ----------------------------------------------------------------------------
# Class TmsTileDef
# ----------------------------------------------------------------------------
class TmsTileDef:

    CELL_SIZE = 'cellSize'
    CRS = 'crs'
    GEO_CRS = '_geographic_crs'
    ID = 'id'
    MATRIX_HEIGHT = 'matrixHeight'
    MATRIX_WIDTH = 'matrixWidth'
    POINT_OF_ORIGIN = 'pointOfOrigin'
    TILE_HEIGHT = 'tileHeight'
    TILE_MATRICES = 'tileMatrices'
    TILE_WIDTH = 'tileWidth'

    CUR_FILE_PARENT: Path = Path(__file__).resolve().parent.parent
    TMS_DIR: Path = CUR_FILE_PARENT / 'TMS'
    JSON_DIR: Path = TMS_DIR / 'RG'
    DB_PATH: Path = JSON_DIR / 'tile_database.gpkg'

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self,
                 tileDef: dict,
                 srs: osr.SpatialReference,
                 geoSrs: osr.SpatialReference,
                 zone: int,
                 zoomLevel: int):

        self._tileDef: dict = tileDef  # Includes the single tile definition.
        self._srs: osr.SpatialReference = None
        self._geoSrs: osr.SpatialReference = None
        self._zone: str = zone
        self._zoomLevel: int = zoomLevel

        self.srs = srs
        self.geoSrs = geoSrs

    # ------------------------------------------------------------------------
    # initFromJson
    # ------------------------------------------------------------------------
    @classmethod
    def initFromJson(cls,
                     tileJson: str,
                     srs: osr.SpatialReference,
                     geoSrs: osr.SpatialReference,
                     zone: str,
                     zoomLevel: int) -> TmsTileDef:

        zoomLevels: list = tileJson[TmsTileDef.TILE_MATRICES]

        tileDef = [zl for zl in zoomLevels \
                   if int(zl[TmsTileDef.ID]) == zoomLevel][0]

        if len(tileDef) == 0:

            raise RuntimeError('Unable to find zoom level ' +
                               str(zoomLevel) +
                               ' in ' +
                               str(tmsPath))

        tileDef[TmsTileDef.CRS] = tileJson[TmsTileDef.CRS]

        return TmsTileDef(tileDef, srs, geoSrs, zone, zoomLevel)

    # ------------------------------------------------------------------------
    # initFromParams
    # ------------------------------------------------------------------------
    @classmethod
    def initFromParams(cls,
                       zone: str,
                       zoomLevel: int) -> TmsTileDef:

        tmsPath = TmsTileDef.getTmsFilePath(zone)

        with open(tmsPath, 'r') as f:
            tms = json.load(f)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(tms[TmsTileDef.CRS])

        geoSrs = osr.SpatialReference()
        geoSrs.ImportFromWkt(tms[TmsTileDef.GEO_CRS])

        return TmsTileDef.initFromJson(tms, srs, geoSrs, zone, zoomLevel)

    # ------------------------------------------------------------------------
    # cellSize
    # ------------------------------------------------------------------------
    @property
    def cellSize(self) -> float:

        return self._tileDef[TmsTileDef.CELL_SIZE]

    # ------------------------------------------------------------------------
    # geoSrs
    # ------------------------------------------------------------------------
    @property
    def geoSrs(self) -> osr.SpatialReference:

        return self._geoSrs

    # ------------------------------------------------------------------------
    # geoSrs
    # ------------------------------------------------------------------------
    @geoSrs.setter
    def geoSrs(self, value: osr.SpatialReference) -> None:

        if not value:

            self._geoSrs = osr.SpatialReference()
            self._geoSrs.ImportFromWkt(self._tileDef[TmsTileDef.GEO_CRS])

        else:
            self._geoSrs = value

        self._geoSrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # ------------------------------------------------------------------------
    # getOverlappingTiles
    # ------------------------------------------------------------------------
    def getOverlappingTiles(self,
                            ulLat: float,
                            ulLon: float,
                            lrLat: float,
                            lrLon: float,
                            minOverlapMeters=10.0) -> list:

        originX, originY = self.pointOfOrigin
        cellSize = self.cellSize
        tileWidth = self.tileWidth
        matrixWidth = self.matrixWidth
        matrixHeight = self.matrixHeight

        tilePixelSize = cellSize * tileWidth

        # Transform corners to projected coordinates
        ulX, ulY = self.llToLtm(ulLat, ulLon)
        lrX, lrY = self.llToLtm(lrLat, lrLon)

        # Get extent in projected space
        minEasting = min(ulX, lrX)
        maxEasting = max(ulX, lrX)
        minNorthing = min(ulY, lrY)
        maxNorthing = max(ulY, lrY)

        # ---
        # Calculate tile range, using epsilon to avoid floating point
        # truncation errors at tile boundaries.
        # ---
        epsilon = 1e-6

        minCol = max(0, math.floor((minEasting - originX) /
                                   tilePixelSize + epsilon))

        maxCol = min(matrixWidth - 1,
                     math.floor((maxEasting - originX) /
                                tilePixelSize + epsilon))

        minRow = max(0, math.floor((originY - maxNorthing) /
                                   tilePixelSize + epsilon))

        maxRow = min(matrixHeight - 1,
                     math.floor((originY - minNorthing) /
                                tilePixelSize + epsilon))

        # Store original query bbox for geographic filtering
        queryMinLat = lrLat
        queryMaxLat = ulLat
        queryMinLon = ulLon
        queryMaxLon = lrLon

        # Generate tile indices, filtering by actual overlap
        indices = []

        for row in range(minRow, maxRow + 1):

            for col in range(minCol, maxCol + 1):

                # Get tile bounds in projected space
                ulx, uly, lrx, lry = self.getTileBbox(col, row)

                # ---
                # Calculate overlap in projected space. Note that uly > lry
                # since Y decreases downward, so min/max must be applied
                # accordingly.
                # ---
                overlapMinX = max(ulx, minEasting)
                overlapMaxX = min(lrx, maxEasting)
                overlapMinY = max(lry, minNorthing)
                overlapMaxY = min(uly, maxNorthing)

                # Calculate overlap width and height
                overlapWidth = max(0, overlapMaxX - overlapMinX)
                overlapHeight = max(0, overlapMaxY - overlapMinY)

                # Check if both dimensions meet minimum overlap requirement
                if overlapWidth >= minOverlapMeters and \
                    overlapHeight >= minOverlapMeters:

                    # ---
                    # Final check: verify tile's geographic bounds overlap
                    # the original query bbox.
                    # ---
                    tUlLat, tUlLon, tLrLat, tLrLon = \
                        self._getTileBboxGeo(col, row)

                    tileMinLon = min(tUlLon, tLrLon)
                    tileMaxLon = max(tUlLon, tLrLon)
                    tileMinLat = min(tUlLat, tLrLat)
                    tileMaxLat = max(tUlLat, tLrLat)

                    geoLatOverlap = (tileMaxLat >= queryMinLat and \
                                     tileMinLat <= queryMaxLat)

                    geoLonOverlap = (tileMaxLon >= queryMinLon and \
                                     tileMinLon <= queryMaxLon)

                    if geoLatOverlap and geoLonOverlap:
                        indices.append((col, row))

        return indices

    # ------------------------------------------------------------------------
    # getTileBbox
    # ------------------------------------------------------------------------
    def getTileBbox(self, tileX: int, tileY: int) -> list[float]:

        '''
        Given a tile index, return its bounding box in LTM.
        '''

        gridOrigin = self.pointOfOrigin
        cellSize = self.cellSize
        width = self.tileWidth
        height = self.tileHeight

        xmin = gridOrigin[0] + tileX * width * cellSize
        xmax = gridOrigin[0] + (tileX + 1) * width * cellSize
        ymax = gridOrigin[1] - tileY * height * cellSize
        ymin = gridOrigin[1] - (tileY + 1) * height * cellSize

        assert(abs(((xmax - xmin) / cellSize) - width) < 1e-6)
        assert(abs(((ymax - ymin) / cellSize) - height) < 1e-6)

        ulx = xmin
        uly = ymax
        lrx = xmax
        lry = ymin

        return [ulx, uly, lrx, lry]

    # ------------------------------------------------------------------------
    # _getTileBboxGeo
    # ------------------------------------------------------------------------
    def _getTileBboxGeo(self, col: int, row: int) -> List[float]:

        # Get projected bounds
        ulx, uly, lrx, lry = self.getTileBbox(col, row)

        ulLat, ulLon = self.ltmToLatLon(ulx, uly)
        lrLat, lrLon = self.ltmToLatLon(lrx, lry)

        return ulLat, ulLon, lrLat, lrLon

    # ------------------------------------------------------------------------
    # getTmsFilePath
    # ------------------------------------------------------------------------
    @staticmethod
    def getTmsFilePath(zone: str) -> Path:

        tmsFileName = 'tms_LTM_' + zone + 'RG.json'
        tmsPath = TmsTileDef.JSON_DIR / tmsFileName
        return tmsPath

    # ------------------------------------------------------------------------
    # llToLtm
    # ------------------------------------------------------------------------
    def llToLtm(self, lat: float, lon: float) -> Tuple(float, float):

        xform = osr.CoordinateTransformation(self.geoSrs, self.srs)
        x, y, _ = xform.TransformPoint(lon, lat)

        return x, y

    # ------------------------------------------------------------------------
    # llToTileIndex
    # ------------------------------------------------------------------------
    def llToTileIndex(self, lat: float, lon: float) -> Tuple(float, float):

        # Calculate tile size in meters
        tilePixelSize = self.cellSize * self.tileWidth

        # Transform point to projected coordinates
        # xform = osr.CoordinateTransformation(self.geoSrs, self.srs)
        # easting, northing, _ = xform.TransformPoint(lat, lon)
        x, y = self.llToLtm(lat, lon)

        # Calculate tile indices
        originX, originY = self.pointOfOrigin
        epsilon = 1e-6
        col = math.floor((x - originX) / tilePixelSize + epsilon)
        row = math.floor((originY - y) / tilePixelSize + epsilon)

        # Check if within matrix bounds
        if col < 0 or col >= self.matrixWidth or \
            row < 0 or row >= self.matrixHeight:

            return None

        return col, row

    # ------------------------------------------------------------------------
    # ltmToLatLon
    # ------------------------------------------------------------------------
    def ltmToLatLon(self, x: float, y: float) -> tuple[float, float]:

        xform = osr.CoordinateTransformation(self.srs, self.geoSrs)
        lon, lat, _ = xform.TransformPoint(x, y)
        return lat, lon

    # ------------------------------------------------------------------------
    # ltmToTileIndex
    # ------------------------------------------------------------------------
    def ltmToTileIndex(self, inX: float, inY: float) -> tuple[int, int]:

        # Extract TMS grid parameters
        originX, originY = self.pointOfOrigin
        cellSize = self.cellSize
        tileWidth = self.tileWidth
        tileHeight = self.tileHeight

        # Calculate tile size in meters
        tileSize = tileWidth * cellSize

        # ---
        # Use epsilon to avoid floating point truncation errors at tile
        # boundaries.
        # ---
        epsilon = 1e-6
        x = math.floor((inX - originX) / tileSize + epsilon)
        y = math.floor((originY - inY) / tileSize + epsilon)

        return x, y

    # ------------------------------------------------------------------------
    # maxtrixHeight
    # ------------------------------------------------------------------------
    @property
    def matrixHeight(self) -> int:

        return self._tileDef[TmsTileDef.MATRIX_HEIGHT]

    # ------------------------------------------------------------------------
    # maxtrixWidth
    # ------------------------------------------------------------------------
    @property
    def matrixWidth(self) -> int:

        return self._tileDef[TmsTileDef.MATRIX_WIDTH]

    # ------------------------------------------------------------------------
    # pointOfOrigin
    # ------------------------------------------------------------------------
    @property
    def pointOfOrigin(self) -> tuple[float, float]:

        return self._tileDef[TmsTileDef.POINT_OF_ORIGIN]

    # ------------------------------------------------------------------------
    # srs
    # ------------------------------------------------------------------------
    @property
    def srs(self) -> osr.SpatialReference:

        if self._srs is None:

            self._srs = osr.SpatialReference()
            self._srs.ImportFromWkt(self._tileDef[TmsTileDef.CRS])

        return self._srs

    # ------------------------------------------------------------------------
    # srs
    # ------------------------------------------------------------------------
    @srs.setter
    def srs(self, value: osr.SpatialReference) -> None:

        if not value:

            self._srs = osr.SpatialReference()
            self._srs.ImportFromWkt(self._tileDef[TmsTileDef.CRS])

        else:
            self._srs = value

        self._srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # ------------------------------------------------------------------------
    # tileHeight
    # ------------------------------------------------------------------------
    @property
    def tileHeight(self) -> int:

        return self._tileDef[TmsTileDef.TILE_HEIGHT]

    # ------------------------------------------------------------------------
    # tileWidth
    # ------------------------------------------------------------------------
    @property
    def tileWidth(self) -> int:

        return self._tileDef[TmsTileDef.TILE_WIDTH]

    # ------------------------------------------------------------------------
    # zone
    # ------------------------------------------------------------------------
    @property
    def zone(self) -> int:

        return self._zone

    # ------------------------------------------------------------------------
    # zoomLevel
    # ------------------------------------------------------------------------
    @property
    def zoomLevel(self) -> int:

        return self._zoomLevel
