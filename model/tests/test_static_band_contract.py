import unittest

from lfm.model.static_band_contract import (
    MINIRF_SOURCE_NODATA,
    MINIRF_SOURCE_NODATA_BANDS,
    STATIC_BAND_NAMES,
    STATIC_OUTPUT_NODATA,
)


class StaticBandContractTestCase(unittest.TestCase):
    def test_static_band_order_is_complete_and_unique(self):
        self.assertEqual(len(STATIC_BAND_NAMES), 63)
        self.assertEqual(len(set(STATIC_BAND_NAMES)), 63)
        self.assertEqual(STATIC_BAND_NAMES[0], "LDRM_32_N_FLOAT.iau")
        self.assertEqual(STATIC_BAND_NAMES[-1], "lola_kaguya_60mpp_slp")

    def test_minirf_source_nodata_bands_are_in_contract(self):
        self.assertEqual(len(MINIRF_SOURCE_NODATA_BANDS), 2)
        self.assertTrue(
            set(MINIRF_SOURCE_NODATA_BANDS).issubset(STATIC_BAND_NAMES)
        )

    def test_static_output_nodata_is_distinct_from_minirf_source_nodata(self):
        self.assertEqual(STATIC_OUTPUT_NODATA, -32768.0)
        self.assertNotEqual(STATIC_OUTPUT_NODATA, MINIRF_SOURCE_NODATA)


if __name__ == "__main__":
    unittest.main()
