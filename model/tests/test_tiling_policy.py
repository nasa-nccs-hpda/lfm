from pathlib import Path
import unittest

from lfm.model.tiling_config import BandNoDataOverride, TileSourceConfig
from lfm.model.tiling_policy import (
    band_nodata_values,
    product_id_from_raster_path,
    select_source_rasters,
    validate_source_selectors,
)
from lfm.model.vector_index import IndexedRaster


class TilingPolicyTestCase(unittest.TestCase):
    def source(self, **overrides) -> TileSourceConfig:
        values = {
            "name": "wac",
            "data_dir": Path("/data/wac"),
            "index_path": Path("/data/wac/index.shp"),
            "selection_mode": "product_id",
        }
        values.update(overrides)
        return TileSourceConfig(**values)

    def test_product_id_from_lunar_filename(self):
        self.assertEqual(
            product_id_from_raster_path("M1187363083CE.prj.uv.mos.tif"),
            "M1187363083CE",
        )

    def test_product_selection_supports_wac_and_nac(self):
        records = [
            IndexedRaster(Path("/data/M100.prj.uv.mos.tif")),
            IndexedRaster(Path("/data/M100.prj.vis.mos.tif")),
            IndexedRaster(Path("/data/M200.ech.cog.tif")),
        ]
        for name in ("wac", "nac"):
            source = self.source(name=name)
            selected = select_source_rasters(source, records, selector="M100")
            self.assertEqual(len(selected), 2)
            self.assertTrue(all("M100" in str(item.path) for item in selected))

    def test_product_selection_requires_selector(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            select_source_rasters(self.source(), [], selector=None)

    def test_all_intersecting_returns_every_record(self):
        source = self.source(name="static", selection_mode="all_intersecting")
        records = [
            IndexedRaster(Path("/data/elevation.tif")),
            IndexedRaster(Path("/data/slope.tif")),
        ]

        self.assertEqual(select_source_rasters(source, records), records)

    def test_all_intersecting_rejects_selector(self):
        source = self.source(name="static", selection_mode="all_intersecting")
        with self.assertRaisesRegex(ValueError, "does not accept"):
            select_source_rasters(source, [], selector="M100")

    def test_source_nodata_defaults_to_metadata(self):
        source = self.source()

        self.assertEqual(
            band_nodata_values(
                source,
                band_name="vis",
                metadata_source_nodata=-9999,
            ),
            (-9999, -9999),
        )

    def test_source_output_nodata_replaces_metadata_value(self):
        source = self.source(output_nodata=-32768)

        self.assertEqual(
            band_nodata_values(
                source,
                band_name="vis",
                metadata_source_nodata=-9999,
            ),
            (-9999, -32768),
        )

    def test_band_override_takes_precedence(self):
        source = self.source(
            output_nodata=-32768,
            band_nodata_overrides=(
                BandNoDataOverride(
                    "delta_cpr",
                    source_value=-3.4e38,
                    preserve_source=True,
                ),
            ),
        )

        self.assertEqual(
            band_nodata_values(
                source,
                band_name="delta_cpr",
                metadata_source_nodata=-9999,
            ),
            (-3.4e38, -3.4e38),
        )

    def test_wac_only_selector_contract(self):
        selectors = validate_source_selectors(
            [self.source(name="wac")],
            {"wac": "M100"},
        )

        self.assertEqual(selectors, {"wac": "M100"})

    def test_nac_only_selector_contract(self):
        selectors = validate_source_selectors(
            [self.source(name="nac")],
            {"nac": "M200"},
        )

        self.assertEqual(selectors, {"nac": "M200"})

    def test_static_only_selector_contract(self):
        static = self.source(name="static", selection_mode="all_intersecting")

        self.assertEqual(validate_source_selectors([static], None), {})

    def test_mixed_source_selector_contract(self):
        static = self.source(name="static", selection_mode="all_intersecting")

        selectors = validate_source_selectors(
            [self.source(name="wac"), static],
            {"wac": "M100"},
        )

        self.assertEqual(selectors, {"wac": "M100"})


if __name__ == "__main__":
    unittest.main()
