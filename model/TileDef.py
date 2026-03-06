
from pathlib import Path

from osgeo import osr

from model.Conversion import Conversion


# ----------------------------------------------------------------------------
# Class TileDef
# ----------------------------------------------------------------------------
class TileDef:
    
    CELL_SIZE = 'cellSize'
    CRS = 'crs'
    ID = 'id'
    POINT_OF_ORIGIN = 'pointOfOrigin'
    TILE_HEIGHT = 'tileHeight'
    TILE_MATRICES = 'tileMatrices'
    TILE_WIDTH = 'tileWidth'
    
    CUR_FILE_PARENT: Path = Path(__file__).resolve().parent
    TMS_DIR: Path = CUR_FILE_PARENT / 'TMS' / 'RG'
    JSON_DIR: Path = TMS_DIR / 'RG'
    DB_PATH: Path = TMS_DIR / 'somedbname.gpkg'
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, zone: str, zoomLevel: int):
        
        tmsFileName = 'tms_LTM_' + zone + 'RG.json'
        tmsPath = Conversion.TMS_DIR / tmsFileName

        with open(tmsPath, 'r') as f:
            tms = json.load(f)

        # ---
        # If we made it here, we must have the correct zone, etc.  No need 
        # to validate.
        # ---
        zoomLevels: list = tms[Conversion.TILE_MATRICES]
        
        tileDef = [zl for zl in zoomLevels \
            if int(zl[Conversion.ID]) == zoomLevel][0]

        if len(tileDef) == 0:
        
            raise RuntimeError('Unable to find zoom level ' + 
                               str(zoomLevel) + 
                               ' in ' + 
                               str(tmsPath))

        tileDef[Conversion.CRS] = tms[Conversion.CRS]
        
        self._tileDef: dict = tileDef
        self._zone: str = zone
        self._zoomLevel: int = zoomLevel
        
    # ------------------------------------------------------------------------
    # cellSize
    # ------------------------------------------------------------------------
    @property
    def cellSize(self) -> float:
        
        return self._tileDef[Conversion.CELL_SIZE]
        
    # -----------------------------------------------------------------------
    # getTileBbox
    # ------------------------------------------------------------------------
    def getTileBbox(self, tileIndex: tuple) -> tuple:

        tileX = tileIndex[0]
        tileY = tileIndex[1]

        gridOrigin = self.pointOfOrigin()
        cellSize = self.cellSize()
        width = self.tileWidth()
        height = self.tileHeight()

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
        originX, originY = self.pointOfOrigin()
        cellSize = self.cellSize()
        tileWidth = self.tileWidth()
        tileHeight = self.tileHeight()
        
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
        
        return self._tileDef[Conversion.POINT_OF_ORIGIN]

    # -----------------------------------------------------------------------
    # srs
    # ------------------------------------------------------------------------
    @property
    def srs(self) -> osr.SpatialReference:
        
        ltmSRS = osr.SpatialReference()
        ltmSRS.ImportFromWkt(self._tileDef[Conversion.CRS])
        return ltmSRS
        
    # ------------------------------------------------------------------------
    # tileHeight
    # ------------------------------------------------------------------------
    @property
    def tileHeight(self) -> int:
        
        return self._tileDef[Conversion.TILE_HEIGHT]

    # ------------------------------------------------------------------------
    # tileWidth
    # ------------------------------------------------------------------------
    @property
    def tileWidth(self) -> int:
        
        return self._tileDef[Conversion.TILE_WIDTH]

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
