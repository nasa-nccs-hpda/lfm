import unittest

from osgeo import osr

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
        
        with open(tmsPath, 'r') as f:
            self._tms = json.load(f)

        self._srs = osr.SpatialReference()
        self._srs.ImportFromWkt(self._tms[TmsTileDef.CRS])

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        zd = TmsZoneDef(self._tmsPath)
        self.assertIsNotNone(zd._zoneDef)
        self.assertTrue(zd.srs.IsSame(self._srs))
        self.assertEqual(zd.zone, self._zone)
        self.assertIsInstance(zd._tileDefs, dict)
        
        