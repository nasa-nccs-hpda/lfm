from pathlib import Path
import unittest

from lfm.model.chip_types import (
    ChipPreflight,
    ChipRequest,
    ChipResult,
    GeographicAOI,
    LabelMismatchError,
    LabelValidationDiagnostic,
    ReferenceSample,
    SourceSelector,
    TargetGrid,
    validate_request_contracts,
)


class ChipTypesTestCase(unittest.TestCase):
    def grid(self, **overrides):
        values = {
            "crs_wkt": "GEOGCRS[Moon]",
            "transform": (149.7, 0.001, 0.0, 1.3, 0.0, -0.001),
            "bounds": (149.7, 1.0, 150.0, 1.3),
            "width": 300,
            "height": 300,
        }
        values.update(overrides)
        return TargetGrid(**values)

    def aoi(self):
        return GeographicAOI(1.3, 149.7, 1.0, 150.0)

    def request(self, sample_id="M1187363083CE_r0_c0", **overrides):
        values = {
            "sample_id": sample_id,
            "target_grid": self.grid(),
            "geographic_aoi": self.aoi(),
            "split_group_key": "M1187363083CE",
            "source_selectors": (
                SourceSelector("wac_grid", "wac", "M1187363083CE"),
            ),
        }
        values.update(overrides)
        return ChipRequest(**values)

    def test_target_grid_requires_complete_valid_metadata(self):
        grid = self.grid()

        self.assertEqual(grid.width, 300)
        self.assertEqual(grid.transform[1], 0.001)
        with self.assertRaisesRegex(ValueError, "crs_wkt"):
            self.grid(crs_wkt="")
        with self.assertRaisesRegex(ValueError, "invertible"):
            self.grid(transform=(149.7, 0.0, 0.0, 1.3, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "left < right"):
            self.grid(bounds=(150.0, 1.0, 149.7, 1.3))
        with self.assertRaisesRegex(ValueError, "positive"):
            self.grid(width=0)

    def test_request_preserves_full_offset_qualified_sample_id(self):
        request = self.request(label_path="/labels/M1187363083CE_r0_c0_label.npy")

        self.assertEqual(request.sample_id, "M1187363083CE_r0_c0")
        self.assertEqual(
            request.source_selectors[0].product_id,
            "M1187363083CE",
        )
        self.assertEqual(
            request.label_path,
            Path("/labels/M1187363083CE_r0_c0_label.npy"),
        )

    def test_requests_may_share_product_but_not_full_sample_id(self):
        first = self.request("M1187363083CE_r0_c0")
        second = self.request("M1187363083CE_r300_c0")

        validate_request_contracts((first, second))
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            validate_request_contracts((first, self.request("m1187363083ce_R0_C0")))

    def test_request_requires_split_group_key(self):
        with self.assertRaisesRegex(ValueError, "split_group_key"):
            self.request(split_group_key=" ")

    def test_request_rejects_duplicate_source_selectors(self):
        selectors = (
            SourceSelector("wac_grid", "wac", "M1"),
            SourceSelector("WAC_GRID", "WAC", "M2"),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            self.request(source_selectors=selectors)

    def test_explicit_split_assignments_must_agree_within_group(self):
        first = self.request(assigned_split="train")
        second = self.request(
            "M1187363083CE_r300_c0",
            assigned_split="test",
        )

        with self.assertRaisesRegex(ValueError, "conflicting explicit assignments"):
            validate_request_contracts((first, second))

    def test_reference_preflight_result_and_typed_label_error(self):
        reference = ReferenceSample(
            path="/references/M1_r0_c0.tif",
            sample_id="M1_r0_c0",
            target_grid=self.grid(),
            geographic_aoi=self.aoi(),
        )
        diagnostic = LabelValidationDiagnostic(
            code="shape_mismatch",
            message="Label shape does not match target grid.",
            expected="(300, 300)",
            actual="(256, 256)",
        )
        preflight = ChipPreflight(
            status="failed",
            assigned_split="val",
            resolved_label_path=Path("/labels/M1_r0_c0_label.npy"),
            label_diagnostics=(diagnostic,),
        )
        result = ChipResult(
            request=self.request(),
            status="failed",
            preflight=preflight,
            message="label mismatch",
            elapsed_seconds=0.25,
        )
        error = LabelMismatchError(
            "label mismatch",
            sample_id="M1_r0_c0",
            diagnostics=(diagnostic,),
        )

        self.assertEqual(reference.path, Path("/references/M1_r0_c0.tif"))
        self.assertEqual(result.preflight.label_diagnostics, (diagnostic,))
        self.assertEqual(error.sample_id, "M1_r0_c0")
        self.assertEqual(error.diagnostics, (diagnostic,))


if __name__ == "__main__":
    unittest.main()
