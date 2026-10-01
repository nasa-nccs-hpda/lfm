import unittest

from lfm.model.lunar_crs import (
    LUNAR_GEOGRAPHIC_WKT_PATH,
    load_lunar_geographic_wkt,
    raster_crs_equivalent,
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

    def test_projected_fallback_ignores_only_raster_axis_metadata(self):
        class ProjectedSrs:
            def __init__(self, proj4):
                self.proj4 = proj4

            def IsSame(self, other):
                return False

            def IsProjected(self):
                return True

            def ExportToProj4(self):
                return self.proj4

        expected = ProjectedSrs(
            "+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 "
            "+x_0=500000 +y_0=500000 +R=1737400 +units=m +axis=enu "
            "+no_defs +type=crs"
        )
        reconstructed = ProjectedSrs(
            "+proj=stere +lat_0=-90.0 +lon_0=0 +k=0.994 "
            "+x_0=500000 +y_0=500000 +R=1737400 +units=m +axis=nnu "
            "+no_defs"
        )
        changed_projection = ProjectedSrs(
            "+proj=stere +lat_0=-90 +lon_0=0 +k=0.995 "
            "+x_0=500000 +y_0=500000 +R=1737400 +units=m +no_defs"
        )

        self.assertTrue(raster_crs_equivalent(expected, reconstructed))
        self.assertFalse(raster_crs_equivalent(expected, changed_projection))

    def test_projected_fallback_does_not_relax_geographic_crs(self):
        class GeographicSrs:
            def IsSame(self, other):
                return False

            def IsProjected(self):
                return False

        self.assertFalse(
            raster_crs_equivalent(GeographicSrs(), GeographicSrs())
        )


if __name__ == "__main__":
    unittest.main()
