
import json
import unittest

from osgeo import osr

from lfm.model.TmsTileDef import TmsTileDef


# -----------------------------------------------------------------------------
# class TmsTileDefTestCase
#
# python -m unittest lfm.model.tests.test_TmsTileDef
# python -m unittest lfm.model.tests.test_TmsTileDef.TmsTileDefTestCase.testInit
# -----------------------------------------------------------------------------
class TmsTileDefTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._zone = '42N'
        tmsFileName = 'tms_LTM_' + self._zone + 'RG.json'
        tmsPath = TmsTileDef.JSON_DIR / tmsFileName

        with open(tmsPath, 'r') as f:
            self._tms = json.load(f)

        self._srs = osr.SpatialReference()
        self._srs.ImportFromWkt(self._tms[TmsTileDef.CRS])

        self._geoSrs = osr.SpatialReference()
        self._geoSrs.ImportFromWkt(self._tms[TmsTileDef.GEO_CRS])
        
        self._zoomLevel = 5
        
        zoomLevels: list = self._tms[TmsTileDef.TILE_MATRICES]
        
        self._tileDef = [zl for zl in zoomLevels \
                        if int(zl[TmsTileDef.ID]) == self._zoomLevel][0]

        self._tileDef[TmsTileDef.CRS] = self._tms[TmsTileDef.CRS]
        
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
        self.assertTrue(tmsTD.srs.IsSame(self._srs))
        self.assertTrue(tmsTD._geoSrs.IsSame(self._geoSrs))
        self.assertEqual(tmsTD._zone, self._zone)
        self.assertEqual(tmsTD._zoomLevel, self._zoomLevel)                   

    # -------------------------------------------------------------------------
    # testInitFromJson
    # -------------------------------------------------------------------------
    def testInitFromJson(self):

        tmsTD = TmsTileDef.initFromJson(self._tms,
                                        self._srs,
                                        self._geoSrs,
                                        self._zone,
                                        self._zoomLevel)

        self.assertEqual(tmsTD._tileDef, self._tileDef)
        self.assertTrue(tmsTD.srs.IsSame(self._srs))
        self.assertTrue(tmsTD._geoSrs.IsSame(self._geoSrs))
        self.assertEqual(tmsTD._zone, self._zone)
        self.assertEqual(tmsTD._zoomLevel, self._zoomLevel)                   

    # -------------------------------------------------------------------------
    # testInitFromParams
    # -------------------------------------------------------------------------
    def testInitFromParams(self):

        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        
        self.assertEqual(tmsTD._tileDef, self._tileDef)
        self.assertTrue(tmsTD.srs.IsSame(self._srs))
        self.assertTrue(tmsTD._geoSrs.IsSame(self._geoSrs))
        self.assertEqual(tmsTD._zone, self._zone)
        self.assertEqual(tmsTD._zoomLevel, self._zoomLevel)                   

    # -------------------------------------------------------------------------
    # testLlToUtm
    # -------------------------------------------------------------------------
    def testLlToUtm(self):
        
        lat = 1.2
        lon = 149.8
        gdalXformE = 183353.592403412 
        gdalXformN = 36378.4385950911

        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        x, y = tmsTD.llToLtm(lat, lon)
        
        self.assertAlmostEqual(gdalXformE, x)
        self.assertAlmostEqual(gdalXformN, y)
        
        lat = 1.3
        lon = 149.7
        gdalXformE = 180325.275307587  
        gdalXformN = 39412.6753393454

        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        x, y = tmsTD.llToLtm(lat, lon)
        
        self.assertAlmostEqual(gdalXformE, x)
        self.assertAlmostEqual(gdalXformN, y)

        lat = 1.1
        lon = 149.9
        gdalXformE = 186382.131946831  
        gdalXformN = 33344.7187973687

        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        x, y = tmsTD.llToLtm(lat, lon)
        
        self.assertAlmostEqual(gdalXformE, x)
        self.assertAlmostEqual(gdalXformN, y)

    # -------------------------------------------------------------------------
    # testGetTileBbox
    # -------------------------------------------------------------------------
    def testGetTileBbox(self):
        
        tileX = 1
        tileY = 62
        
        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        ulx, uly, lrx, lry = tmsTD.getTileBbox(tileX, tileY)
        
        chatXmin = 167551.3878924654
        chatYmin = 38822.0460662215
        chatXmax = 206373.4338767631
        chatYmax = 77644.0920505192
        
        self.assertAlmostEqual(ulx, chatXmin, 5)
        self.assertAlmostEqual(uly, chatYmax, 2)
        self.assertAlmostEqual(lrx, chatXmax, 5)
        self.assertAlmostEqual(lry, chatYmin, 2)

    # -------------------------------------------------------------------------
    # testGetTileBoundsGeo
    # -------------------------------------------------------------------------
    def testGetTileBboxGeo(self):
        
        col = 1
        row = 62
        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        ulLat, ulLon, lrLat, lrLon = tmsTD._getTileBboxGeo(col, row)
        
        gdalXformUlLat = 2.56021010127471
        gdalXformUlLon = 149.276599903094
        gdalXformLrLat = 1.28114577330334
        gdalXformLrLon = 150.559639399566
        
        self.assertAlmostEqual(ulLat, gdalXformUlLat)
        self.assertAlmostEqual(ulLon, gdalXformUlLon)
        self.assertAlmostEqual(lrLat, gdalXformLrLat)
        self.assertAlmostEqual(lrLon, gdalXformLrLon)

    # -------------------------------------------------------------------------
    # testLlToTileIndex
    # -------------------------------------------------------------------------
    def testLlToTileIndex(self):
        
        gdalXformUlLat = 2.56021010127471
        gdalXformUlLon = 149.276599903094
        
        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        col, row = tmsTD.llToTileIndex(gdalXformUlLat, gdalXformUlLon)

        self.assertEqual(col, 1)
        self.assertEqual(row, 62)
        
    # -------------------------------------------------------------------------
    # testLtmToTileIndex
    # -------------------------------------------------------------------------
    def testLtmToTileIndex(self):
        
        ulE = 167551.3878924654
        ulN = 77644.0920505192
        
        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        col, row = tmsTD.ltmToTileIndex(ulE, ulN)
        
        self.assertEqual(col, 1)
        self.assertEqual(row, 62)

    # -------------------------------------------------------------------------
    # testGetOverlappingTiles
    # -------------------------------------------------------------------------
    def testGetOverlappingTiles(self):
        
        ulLat = 1.3
        ulLon = 149.7
        lrLat = 1.1
        lrLon = 149.9
        
        tmsTD = TmsTileDef.initFromParams(self._zone, self._zoomLevel)
        indices = tmsTD.getOverlappingTiles(ulLat, ulLon, lrLat, lrLon)
        
        exp = [(1, 62), (1, 63)]
        self.assertEqual(indices, exp)
        