
import json
import unittest

from osgeo import osr

from lfm.model.TmsTileDef import TmsTileDef
from lfm.model.TmsZoneDef import TmsZoneDef


# -----------------------------------------------------------------------------
# class TmsZoneDefTestCase
#
# python -m unittest lfm.model.tests.test_TmsZoneDef
# python -m unittest lfm.model.tests.test_TmsZoneDef.TmsZoneDefTestCase.testInit
# -----------------------------------------------------------------------------
class TmsZoneDefTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._zone = '42N'
        tmsFileName = 'tms_LTM_' + self._zone + 'RG.json'
        self._tmsPath = TmsTileDef.JSON_DIR / tmsFileName

        with open(self._tmsPath, 'r') as f:
            self._tms = json.load(f)

        self._srs = osr.SpatialReference()
        self._srs.ImportFromWkt(self._tms[TmsTileDef.CRS])

        self._zoomLevel = 5

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        zd = TmsZoneDef(self._tmsPath)

        self.assertIsNotNone(zd._zoneDef)
        self.assertTrue(zd.srs.IsSame(self._srs))
        self.assertEqual(zd.zone, self._zone)
        self.assertIsInstance(zd._tileDefs, dict)
        
        # Test a file in a single-digit zone.
        zone = '1N'
        tmsFileName = 'tms_LTM_' + zone + 'RG.json'
        tmsPath = TmsTileDef.JSON_DIR / tmsFileName
        zd = TmsZoneDef(tmsPath)

        self.assertEqual(zd.zone, '1N')

    # -------------------------------------------------------------------------
    # testGetIntersectingTiles
    # -------------------------------------------------------------------------
    def testGetIntersectingTiles(self):

        ulLat = 1.3
        ulLon = 149.7
        lrLat = 1.1
        lrLon = 149.9

        zd = TmsZoneDef(self._tmsPath)

        indices: list = zd.getIntersectingTiles(ulLat,
                                                ulLon,
                                                lrLat,
                                                lrLon,
                                                self._zoomLevel)

        exp = [(1, 62), (1, 63)]
        self.assertEqual(indices, exp)

    # -------------------------------------------------------------------------
    # testGetTileDef
    # -------------------------------------------------------------------------
    def testGetTileDef(self):

        zd = TmsZoneDef(self._tmsPath)
        td: TmsTileDef = zd.getTileDef(self._zoomLevel)

        self.assertIsInstance(td, TmsTileDef)
        self.assertTrue(td.srs.IsSame(self._srs))
        self.assertEqual(td._zone, self._zone)
        self.assertEqual(td._zoomLevel, self._zoomLevel)

    # -------------------------------------------------------------------------
    # testIntersectsBbox
    # -------------------------------------------------------------------------
    def testIntersectsBbox(self):

        ulLat = 1.3
        ulLon = 149.7
        lrLat = 1.1
        lrLon = 149.9

        zd = TmsZoneDef(self._tmsPath)
        intersects = zd.intersectsBbox(ulLat, ulLon, lrLat, lrLon)
        self.assertTrue(intersects)

        # This one does not overlap.
        ulLat = 45
        ulLon = 0
        lrLat = 10
        lrLon = 8
        intersects = zd.intersectsBbox(ulLat, ulLon, lrLat, lrLon)
        self.assertFalse(intersects)

        # This intersects in latitude, but not longitude.
        ulLat = 60
        ulLon = 0
        lrLat = 10
        lrLon = 8
        intersects = zd.intersectsBbox(ulLat, ulLon, lrLat, lrLon)
        self.assertFalse(intersects)

        # This intersects in longitude, but not latitude.
        ulLat = -10
        ulLon = 150
        lrLat = -40
        lrLon = 154
        intersects = zd.intersectsBbox(ulLat, ulLon, lrLat, lrLon)
        self.assertFalse(intersects)

    # -------------------------------------------------------------------------
    # testZoneBbox
    # -------------------------------------------------------------------------
    def testZoneBbox(self):

        zd = TmsZoneDef(self._tmsPath)
        minLat, minLon, maxLat, maxLon = zd.zoneBbox()

        expMinLat = 0
        expMinLon = 148
        expMaxLat = 82
        expMaxLon = 156

        self.assertEqual(minLat, expMinLat)
        self.assertEqual(minLon, expMinLon)
        self.assertEqual(maxLat, expMaxLat)
        self.assertEqual(maxLon, expMaxLon)

