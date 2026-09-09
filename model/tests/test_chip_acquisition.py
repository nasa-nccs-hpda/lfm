from pathlib import Path
import tempfile
from unittest import mock
import unittest

from lfm.model.chip_acquisition import (
    SelectorResolutionError,
    acquire_prepared_request,
    derive_source_selectors,
)
from lfm.model.chip_config import (
    AcquisitionGroupConfig,
    ChipConfig,
    OutputModalityConfig,
    SimpleSplitConfig,
    SplitPercentages,
)
from lfm.model.chip_preflight import PreparedChipRequest
from lfm.model.chip_splits import SplitAssignment
from lfm.model.chip_types import (
    ChipPreflight,
    ChipRequest,
    GeographicAOI,
    SourceSelector,
    TargetGrid,
)
from lfm.model.tiling_config import TileConfig, TileSourceConfig
from lfm.model.tiling_results import MissingRequiredSourceError, TileCubeRecord


class ChipAcquisitionTestCase(unittest.TestCase):
    def source(
        self,
        name,
        *,
        selection_mode="all_intersecting",
        required=True,
    ):
        return TileSourceConfig(
            name=name,
            data_dir=Path(f"/data/{name}"),
            index_path=Path(f"/data/{name}/index.gpkg"),
            selection_mode=selection_mode,
            required=required,
        )

    def group(self, name, root, zoom, *sources):
        return AcquisitionGroupConfig(
            name,
            TileConfig(root / "unused", zoom, tuple(sources)),
        )

    def config(self, root, groups):
        modalities = tuple(
            OutputModalityConfig(group.name, source.name, f"{group.name}_{source.name}")
            for group in groups
            for source in group.tile_config.sources
        )
        return ChipConfig(
            output_root=root / "output",
            intermediate_root=root / "intermediate",
            label_source=root / "labels",
            acquisition_groups=tuple(groups),
            output_modalities=modalities,
            split_config=SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
        )

    def request(self, sample_id="M100_r0_c0", *, aoi=None, selectors=()):
        return ChipRequest(
            sample_id=sample_id,
            target_grid=TargetGrid(
                crs_wkt="GEOGCRS[Moon]",
                transform=(149.7, 0.1, 0.0, 1.3, 0.0, -0.1),
                bounds=(149.7, 1.0, 150.0, 1.3),
                width=3,
                height=3,
            ),
            geographic_aoi=aoi or GeographicAOI(1.3, 149.7, 1.0, 150.0),
            split_group_key=sample_id,
            source_selectors=tuple(selectors),
        )

    def prepared(self, request, *, passed=True):
        split = "train" if passed else None
        return PreparedChipRequest(
            request=request,
            assignment=SplitAssignment(
                sample_id=request.sample_id,
                split_group_key=request.split_group_key,
                assigned_split=split,
                source="automatic_percentage" if passed else "unassigned",
            ),
            preflight=ChipPreflight(
                status="passed" if passed else "skipped",
                assigned_split=split,
            ),
        )

    def record(
        self,
        path,
        *,
        source_name="wac",
        zoom=5,
        zone="42N",
        tile_x=1,
        product_id="M100",
    ):
        return TileCubeRecord(
            source_name=source_name,
            zone=zone,
            zoom_level=zoom,
            tile_x=tile_x,
            tile_y=63,
            product_id=product_id,
            path=path,
            band_names=("band",),
            crs_wkt="PROJCRS[test]",
            nodata_values=(-32768.0,),
        )

    def test_built_in_product_selectors_preserve_full_sample_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group(
                "wac_grid",
                root,
                5,
                self.source("wac", selection_mode="product_id"),
                self.source("static"),
            )
            config = self.config(root, (group,))
            first = self.request("M100_r0_c0")
            second = self.request("M100_r300_c0")
            overridden = self.request(
                "M100_r600_c0",
                selectors=(SourceSelector("wac_grid", "wac", "M200"),),
            )

            first_selectors = derive_source_selectors(first, config)
            second_selectors = derive_source_selectors(second, config)
            override_selectors = derive_source_selectors(overridden, config)

        self.assertEqual(first.sample_id, "M100_r0_c0")
        self.assertEqual(second.sample_id, "M100_r300_c0")
        self.assertEqual(first_selectors[0].product_id, "M100")
        self.assertEqual(second_selectors[0].product_id, "M100")
        self.assertEqual(override_selectors[0].product_id, "M200")
        self.assertEqual(len(first_selectors), 1)

    def test_explicit_selector_and_custom_product_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group(
                "custom_grid",
                root,
                7,
                self.source("hyperspectral", selection_mode="product_id"),
            )
            config = self.config(root, (group,))
            request = self.request(
                selectors=(
                    SourceSelector("custom_grid", "hyperspectral", "CUSTOM-7"),
                )
            )

            selectors = derive_source_selectors(request, config)

            self.assertEqual(selectors[0].product_id, "CUSTOM-7")
            with self.assertRaises(SelectorResolutionError):
                derive_source_selectors(self.request(), config)

    def test_all_intersecting_source_rejects_explicit_selector(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group("context", root, 5, self.source("static"))
            config = self.config(root, (group,))
            request = self.request(
                selectors=(SourceSelector("context", "static", "not-valid"),)
            )

            with self.assertRaisesRegex(
                SelectorResolutionError,
                "does not accept",
            ):
                derive_source_selectors(request, config)

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_invalid_selector_becomes_structured_failure(self, create_tiles):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group(
                "custom_grid",
                root,
                7,
                self.source("hyperspectral", selection_mode="product_id"),
            )
            config = self.config(root, (group,))

            result = acquire_prepared_request(
                self.prepared(self.request()),
                config,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.group_results, ())
        self.assertEqual(result.diagnostics[0].code, "invalid_source_selector")
        create_tiles.assert_not_called()

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_antimeridian_queries_deduplicate_structured_records(self, create_tiles):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group(
                "wac_grid",
                root,
                5,
                self.source("wac", selection_mode="product_id"),
            )
            config = self.config(root, (group,))
            request = self.request(
                aoi=GeographicAOI(1.0, 179.8, 0.8, 180.2),
            )
            record = self.record(
                root / "intermediate/M100_r0_c0/wac_grid/cube.tif"
            )
            create_tiles.side_effect = ([record], [record])

            result = acquire_prepared_request(self.prepared(request), config)

        group_result = result.group_results[0]
        self.assertEqual(result.status, "complete")
        self.assertEqual(create_tiles.call_count, 2)
        self.assertEqual(len(group_result.query_parts), 2)
        self.assertEqual(group_result.records, (record,))
        self.assertEqual(len(group_result.record_groups), 1)
        self.assertEqual(
            group_result.output_dir,
            root / "intermediate/M100_r0_c0/wac_grid",
        )
        self.assertEqual(group_result.selector_mapping, {"wac": "M100"})
        for call in create_tiles.call_args_list:
            self.assertEqual(call.args[0].output_dir, group_result.output_dir)
            self.assertEqual(call.kwargs["selectors"], {"wac": "M100"})

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_later_query_failure_retains_earlier_query_records(self, create_tiles):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group(
                "wac_grid",
                root,
                5,
                self.source("wac", selection_mode="product_id"),
            )
            config = self.config(root, (group,))
            request = self.request(
                aoi=GeographicAOI(1.0, 179.8, 0.8, 180.2),
            )
            first = self.record(root / "first.tif", tile_x=1)
            within_failure = self.record(root / "second.tif", tile_x=2)
            error = MissingRequiredSourceError(
                "required WAC is missing",
                source_name="wac",
                zone="1N",
                tile_x=3,
                tile_y=63,
                completed_records=(within_failure,),
            )
            create_tiles.side_effect = ([first], error)

            result = acquire_prepared_request(self.prepared(request), config)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.group_results[0].records, (first, within_failure))
        self.assertEqual(create_tiles.call_count, 2)

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_required_failure_retains_partial_records_and_inventory(self, create_tiles):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_group = self.group(
                "wac_grid",
                root,
                5,
                self.source("static"),
                self.source("wac", selection_mode="product_id"),
            )
            later_group = self.group(
                "context",
                root,
                11,
                self.source("static"),
            )
            config = self.config(root, (first_group, later_group))
            request = self.request()

            def fail(tile_config, **kwargs):
                tile_config.output_dir.mkdir(parents=True)
                partial_path = tile_config.output_dir / "partial.tif"
                partial_path.touch()
                record = self.record(
                    partial_path,
                    source_name="static",
                    product_id=None,
                )
                raise MissingRequiredSourceError(
                    "required WAC is missing",
                    source_name="wac",
                    zone="42N",
                    tile_x=2,
                    tile_y=63,
                    completed_records=(record,),
                )

            create_tiles.side_effect = fail
            result = acquire_prepared_request(self.prepared(request), config)

        group_result = result.group_results[0]
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.group_results), 1)
        self.assertEqual(group_result.records[0].source_name, "static")
        self.assertEqual(group_result.inventory_paths, (group_result.records[0].path,))
        self.assertEqual(group_result.diagnostics[0].code, "missing_required_source")
        self.assertEqual(create_tiles.call_count, 1)

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_optional_sparse_source_does_not_fail_group(self, create_tiles):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group(
                "nac_grid",
                root,
                11,
                self.source("nac", selection_mode="product_id", required=False),
                self.source("static"),
            )
            config = self.config(root, (group,))
            nac_record = self.record(
                root / "nac-1.tif",
                source_name="nac",
                zoom=11,
                tile_x=1,
                product_id="N100",
            )
            static_records = (
                self.record(
                    root / "static-1.tif",
                    source_name="static",
                    zoom=11,
                    tile_x=1,
                    product_id=None,
                ),
                self.record(
                    root / "static-2.tif",
                    source_name="static",
                    zoom=11,
                    tile_x=2,
                    product_id=None,
                ),
            )
            create_tiles.return_value = [nac_record, *static_records]

            result = acquire_prepared_request(
                self.prepared(self.request("N100_r0_c0")),
                config,
            )

        group_result = result.group_results[0]
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(group_result.records), 3)
        self.assertEqual(
            group_result.diagnostics[0].code,
            "incomplete_optional_source_coverage",
        )
        self.assertEqual(group_result.diagnostics[0].tile_x, 2)

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_same_tile_coordinates_in_multiple_zones_remain_distinct(
        self,
        create_tiles,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group("context", root, 5, self.source("static"))
            config = self.config(root, (group,))
            records = (
                self.record(
                    root / "42N.tif",
                    source_name="static",
                    zone="42N",
                    product_id=None,
                ),
                self.record(
                    root / "43N.tif",
                    source_name="static",
                    zone="43N",
                    product_id=None,
                ),
            )
            create_tiles.return_value = list(records)

            result = acquire_prepared_request(
                self.prepared(self.request("SAMPLE_A")),
                config,
            )

        group_result = result.group_results[0]
        self.assertEqual(group_result.records, records)
        self.assertEqual(len(group_result.record_groups), 2)
        self.assertEqual(
            {item.key.zone for item in group_result.record_groups},
            {"42N", "43N"},
        )

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_colliding_tile_addresses_use_separate_sample_directories(
        self,
        create_tiles,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group("context", root, 5, self.source("static"))
            config = self.config(root, (group,))

            def return_record(tile_config, **kwargs):
                return [
                    self.record(
                        tile_config.output_dir / "same-tile.tif",
                        source_name="static",
                        product_id=None,
                    )
                ]

            create_tiles.side_effect = return_record
            first = acquire_prepared_request(
                self.prepared(self.request("SAMPLE_A")),
                config,
            )
            second = acquire_prepared_request(
                self.prepared(self.request("SAMPLE_B")),
                config,
            )

        first_path = first.group_results[0].records[0].path
        second_path = second.group_results[0].records[0].path
        self.assertNotEqual(first_path.parent, second_path.parent)
        self.assertEqual(first_path.name, second_path.name)

    @mock.patch("lfm.model.chip_acquisition.create_tiles_for_aoi")
    def test_multiple_acquisition_zooms_remain_separate(self, create_tiles):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            groups = (
                self.group("coarse", root, 5, self.source("static_coarse")),
                self.group("fine", root, 11, self.source("static_fine")),
            )
            config = self.config(root, groups)

            def return_record(tile_config, **kwargs):
                source_name = tile_config.sources[0].name
                return [
                    self.record(
                        tile_config.output_dir / "cube.tif",
                        source_name=source_name,
                        zoom=tile_config.zoom_level,
                        product_id=None,
                    )
                ]

            create_tiles.side_effect = return_record
            result = acquire_prepared_request(
                self.prepared(self.request("SAMPLE_A")),
                config,
            )

        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [item.zoom_level for item in result.group_results],
            [5, 11],
        )
        self.assertNotEqual(
            result.group_results[0].record_groups[0].key.acquisition_group,
            result.group_results[1].record_groups[0].key.acquisition_group,
        )

    @mock.patch("lfm.model.chip_acquisition.derive_source_selectors")
    def test_ineligible_request_never_derives_selectors(self, derive_selectors):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = self.group("context", root, 5, self.source("static"))
            config = self.config(root, (group,))

            with self.assertRaisesRegex(ValueError, "did not pass"):
                acquire_prepared_request(
                    self.prepared(self.request(), passed=False),
                    config,
                )

        derive_selectors.assert_not_called()


if __name__ == "__main__":
    unittest.main()
