import json
import unittest

from osgeo import osr

from lfm.model.Conversions import Conversions
from lfm.model.TmsTileDef import TmsTileDef


# -----------------------------------------------------------------------------
# class ConversionsTestCase
#
# python -m unittest lfm.model.tests.test_Conversions
# python -m unittest lfm.model.tests.test_Conversions.ConversionsTestCase.testLatLonToLTM
# -----------------------------------------------------------------------------
class ConversionsTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._zone = '42N'
        tmsFileName = 'tms_LTM_' + self._zone + 'RG.json'
        self._tmsPath = TmsTileDef.JSON_DIR / tmsFileName
        
        with open(self._tmsPath, 'r') as f:
            self._tms = json.load(f)

    # -------------------------------------------------------------------------
    # testLatLonToLTM
    # -------------------------------------------------------------------------
    def testLatLonToLTM(self):
        
        lat = 1.2
        lon = 149.8
        gdalXformE = 183353.592403412 
        gdalXformN = 36378.4385950911
        
        outSrs = osr.SpatialReference()
        outSrs.ImportFromWkt(self._tms[TmsTileDef.CRS])

        easting, northing = Conversions.latLonToLTM(lat, lon, outSrs)
        
        self.assertAlmostEqual(easting, gdalXformE)
        self.assertAlmostEqual(northing, gdalXformN)

    # -------------------------------------------------------------------------
    # testLtmToLatLon
    # -------------------------------------------------------------------------
    # def testLtmToLatLon(self):
    #
    #     lat = 1.2
    #     lon = 149.8
    #     gdalXformE = 183353.592403412
    #     gdalXformN = 36378.4385950911
    #
    #     self._geoSrs = osr.SpatialReference()
    #     self._geoSrs.ImportFromWkt(self._tms[TmsTileDef.GEO_CRS])

        