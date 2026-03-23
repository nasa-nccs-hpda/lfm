
import unittest

from lfm.model.TmsTileDef import TmsTileDef


# -----------------------------------------------------------------------------
# class TmsTileDefTestCase
#
# cd nse-da
# python -m unittest lfm.model.tests.test_TmsTileDef
# python -m unittest lfm.model.tests.test_TmsTileDef.TmsTileDefTestCase.testInit
# -----------------------------------------------------------------------------
class TmsTileDefTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._zone = 21
        tmsFileName = 'tms_LTM_' + self._zone + 'RG.json'
        tmsPath = TmsTileDef.JSON_DIR / tmsFileName

        with open(tmsPath, 'r') as f:
            tms = json.load(f)

        self._srs = osr.SpatialReference()
        self._srs.ImportFromWkt(tms[TmsTileDef.CRS])

        self._geoSrs = osr.SpatialReference()
        self._geoSrs.ImportFromWkt(tms[TmsTileDef.GEO_CRS])
        
        self._zoomLevel = 1
        
        zoomLevels: list = tms[TmsTileDef.TILE_MATRICES]
        
        self._tileDef = [zl for zl in zoomLevels \
                        if int(zl[TmsTileDef.ID]) == zoomLevel][0]
        
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        # Default initialization
        tmsTD = TmsTileDef()
        
        # Initialization parameters
        tmsTD = TmsTileDef(self._tileDef, 
                           self._srs,
                           self._geoSrs,
                           self._zone,
                           self._zoomLevel)
        
        self.assertEqual(tmsTD._tileDef, self._tileDef)
        self.assertEqual(tmsTD._srs, self._srs)
        self.assertEqual(tmsTD._geoSrs, self._geoSrs)
        self.assertEqual(tmsTD._zone, self._zone)
        self.assertEqual(tmsTD._zoomLevel, self._zoomLevel)                   
        
        