import unittest

from lfm.model.lunar_crs import (
    LUNAR_GEOGRAPHIC_WKT_PATH,
    load_lunar_geographic_wkt,
)


class LunarCrsTestCase(unittest.TestCase):
    def test_repository_wkt_exists(self):
        self.assertTrue(LUNAR_GEOGRAPHIC_WKT_PATH.is_file())

    def test_repository_wkt_is_iau_30100(self):
        wkt = load_lunar_geographic_wkt()

        self.assertTrue(wkt.startswith("GEOGCRS["))
        compact_wkt = "".join(wkt.split())
        self.assertIn('ID["IAU",30100,2015]', compact_wkt)
        self.assertIn("1737400", wkt)


if __name__ == "__main__":
    unittest.main()
