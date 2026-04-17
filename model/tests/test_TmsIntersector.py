
import unittest

from lfm.model.TmsIntersector import TmsIntersector


# -----------------------------------------------------------------------------
# class TmsIntersectorTestCase
#
# python -m unittest lfm.model.tests.test_TmsIntersector
# python -m unittest lfm.model.tests.test_TmsIntersector.TmsIntersectorTestCase.testInit
# -----------------------------------------------------------------------------
class TmsIntersectorTestCase(unittest.TestCase):


    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        tmsi = TmsIntersector()
        self.assertEqual(len(tmsi.zones), 92)

    # -------------------------------------------------------------------------
    # testGetTids
    # -------------------------------------------------------------------------
    def testGetTids(self):

        ulLat = 1.3
        ulLon = 149.7
        lrLat = 1.1
        lrLon = 149.9

        tmsi = TmsIntersector()

        allTiles = tmsi.getTids(ulLat, ulLon, lrLat, lrLon, 5)

        exp = [{'tileX': 1, 'tileY': 62, 'zone': '42N', 'zoomLevel': 5},
               {'tileX': 1, 'tileY': 63, 'zone': '42N', 'zoomLevel': 5}]

        self.assertEqual(allTiles, exp)
