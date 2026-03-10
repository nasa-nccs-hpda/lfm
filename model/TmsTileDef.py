
import json
from pathlib import Path

from osgeo import osr

from model.Conversions import Conversions


# ----------------------------------------------------------------------------
# Class TmsTileDef
# ----------------------------------------------------------------------------
class TmsTileDef:
    
    CELL_SIZE = 'cellSize'
    CRS = 'crs'
    ID = 'id'
    POINT_OF_ORIGIN = 'pointOfOrigin'
    TILE_HEIGHT = 'tileHeight'
    # TILE_MATRICES = 'tileMatrices'
    TILE_WIDTH = 'tileWidth'
    
    CUR_FILE_PARENT: Path = Path(__file__).resolve().parent.parent
    TMS_DIR: Path = CUR_FILE_PARENT / 'TMS'
    JSON_DIR: Path = TMS_DIR / 'RG'
    DB_PATH: Path = JSON_DIR / 'tile_database.gpkg'
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 tileDef: dict = None,
                 srs: osr.SpatialReference = None
                 zone: int = None,
                 zoomLevel: int = None):
        
        self._tileDef: dict = tileDef
        self._srs: osr.SpatialReference() = srs
        self._zone: str = zone
        self._zoomLevel: int = zoomLevel
        
    # ------------------------------------------------------------------------
    # initFromJson
    # ------------------------------------------------------------------------
    @classmethod
    def initFromJson(tileJson: str, 
                     srs: osr.SpatialReference,
                     zone: str, 
                     zoomLevel: int) -> TmsTileDef:
        
        return TmsTileDef(tileJson, srs, zone, zoomLevel)
        
    # ------------------------------------------------------------------------
    # initFromParams
    # ------------------------------------------------------------------------
    @classmethod
    def initFromParams(zone: str, 
                       zoomLevel: int) -> TmsTileDef:
        
        tmsFileName = 'tms_LTM_' + zone + 'RG.json'
        tmsPath = TileDef.JSON_DIR / tmsFileName

        with open(tmsPath, 'r') as f:
            tms = json.load(f)

        # ---
        # If we made it here, we must have the correct zone, etc.  No need 
        # to validate.
        # ---
        zoomLevels: list = tms[TileDef.TILE_MATRICES]
        
        tileDef = [zl for zl in zoomLevels \
            if int(zl[TileDef.ID]) == zoomLevel][0]

        if len(tileDef) == 0:
        
            raise RuntimeError('Unable to find zoom level ' + 
                               str(zoomLevel) + 
                               ' in ' + 
                               str(tmsPath))

        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.tileDef[TileDef.CRS])
        
        return TmsTileDef(tileJson, srs, zone, zoomLevel)
        
    # ------------------------------------------------------------------------
    # cellSize
    # ------------------------------------------------------------------------
    @property
    def cellSize(self) -> float:
        
        return self._tileDef[TileDef.CELL_SIZE]
        
    # -----------------------------------------------------------------------
    # getTileBbox
    # ------------------------------------------------------------------------
    def getTileBbox(self, tileX: int, tileY: int) -> tuple:

        gridOrigin = self.pointOfOrigin
        cellSize = self.cellSize
        width = self.tileWidth
        height = self.tileHeight

        ulx = gridOrigin[0] + tileX * width * cellSize
        uly = gridOrigin[1] - tileY * height * cellSize
        llx = ulx
        lly = uly - height * cellSize
        urx = ulx + width * cellSize
        ury = uly

        assert(((urx - llx) / cellSize) - width < 1)
        assert(((ury - lly) / cellSize) - height < 1)
    
        return ((llx, lly), (urx, ury))

    # ------------------------------------------------------------------------
    # ltmToTileIndex
    # ------------------------------------------------------------------------
    def ltmToTileIndex(self, easting: float, northing: float) -> tuple[int, int]:

        # Extract TMS grid parameters
        originX, originY = self.pointOfOrigin
        cellSize = self.cellSize
        tileWidth = self.tileWidth
        tileHeight = self.tileHeight
        
        # Calculate tile size in meters
        tileSize = tileWidth * cellSize
    
        # Convert LTM coordinates to tile indices
        x = int((easting - originX) / tileSize)
        y = int((originY - northing) / tileSize)
    
        return x, y

    # ------------------------------------------------------------------------
    # pointOfOrigin
    # ------------------------------------------------------------------------
    @property
    def pointOfOrigin(self) -> tuple[float, float]:
        
        return self._tileDef[TileDef.POINT_OF_ORIGIN]

    # -----------------------------------------------------------------------
    # srs
    # ------------------------------------------------------------------------
    @property
    def srs(self) -> osr.SpatialReference:
        
        return self._srs
        
    # ------------------------------------------------------------------------
    # tileHeight
    # ------------------------------------------------------------------------
    @property
    def tileHeight(self) -> int:
        
        return self._tileDef[TileDef.TILE_HEIGHT]

    # ------------------------------------------------------------------------
    # tileWidth
    # ------------------------------------------------------------------------
    @property
    def tileWidth(self) -> int:
        
        return self._tileDef[TileDef.TILE_WIDTH]

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
