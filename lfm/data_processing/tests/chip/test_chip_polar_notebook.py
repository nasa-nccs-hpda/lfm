"""Executable notebook validation controls, without HPC imagery or GDAL."""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from lfm.data_processing._paths import REPO_ROOT
from lfm.data_processing.chip import GeographicAOI, LabelInput
from lfm.data_processing.chip.chip_notebook_utils import chip_batch_fingerprints


def notebook_cell(name):
    notebook = json.loads((REPO_ROOT / "notebooks/chip_polar_example.ipynb").read_text())
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == name))


class PolarNotebookValidationTestCase(unittest.TestCase):
    def test_real_presets_both_hemispheres_and_rerun_are_stable(self):
        for hemisphere in ("north", "south"):
            built = []

            def build(**kwargs):
                built.append(kwargs)
                return SimpleNamespace(**kwargs, target_grid=SimpleNamespace(width=8, height=8))

            base = SimpleNamespace(sample_id="base", split_group_key="product")
            namespace = dict(Path=Path, GeographicAOI=GeographicAOI, LabelInput=LabelInput,
                HEMISPHERE=hemisphere, SAMPLE_ID="base", LABEL_PATH=Path("default.gpkg"),
                LABEL_KIND="auto", LABEL_LAYER="craters", CASE_LABEL_PATHS={"antimeridian": "other.gpkg"},
                EXTRA_REAL_CASES=("antimeridian", "seam_antimeridian", "boundary_polar", "boundary_ltm"),
                requests=[base], request=base, source_grid=object(), static_only=False,
                selectors=(), MAX_OUTPUT_PIXELS=100, chip_request_from_aoi=build,
                print=lambda *args: None)
            for _ in range(2):
                exec(notebook_cell("polar_extra_queries"), namespace)
                self.assertEqual(len(namespace["requests"]), 5)
            self.assertEqual(built[0]["label_input"].path, Path("other.gpkg"))
            seam = built[1]["geographic_aoi"]
            self.assertLess(seam.lower_right_latitude, 82 if hemisphere == "north" else -82)
            self.assertGreater(seam.upper_left_latitude, 82 if hemisphere == "north" else -82)
            self.assertGreater(seam.upper_left_longitude, seam.lower_right_longitude)
            namespace["EXTRA_REAL_CASES"] = ("typo",)
            with self.assertRaisesRegex(ValueError, "Unknown extra cases"):
                exec(notebook_cell("polar_extra_queries"), namespace)
            namespace["EXTRA_REAL_CASES"] = ("antimeridian",)
            namespace["MAX_OUTPUT_PIXELS"] = 1
            with self.assertRaisesRegex(ValueError, "MAX_OUTPUT_PIXELS"):
                exec(notebook_cell("polar_extra_queries"), namespace)

    def test_fingerprints_compare_bytes_and_splits_not_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def batch(name, split="train", content=b"chip"):
                folder = root / name
                folder.mkdir()
                chip, label = folder / "chip.tif", folder / "label.npy"
                chip.write_bytes(content)
                label.write_bytes(b"label")
                return SimpleNamespace(results=[SimpleNamespace(
                    request=SimpleNamespace(sample_id="sample"), status="success", message=None,
                    chip_path=chip, label_path=label, preflight=SimpleNamespace(assigned_split=split))])

            first, second = batch("serial"), batch("parallel")
            self.assertEqual(chip_batch_fingerprints(first), chip_batch_fingerprints(second))
            self.assertNotEqual(chip_batch_fingerprints(first), chip_batch_fingerprints(batch("different", content=b"changed")))
            self.assertNotEqual(chip_batch_fingerprints(first), chip_batch_fingerprints(batch("split", split="test")))
            second.results[0].status = "failed"
            with self.assertRaisesRegex(ValueError, "unsuccessful"):
                chip_batch_fingerprints(second)
            with self.assertRaisesRegex(ValueError, "empty"):
                chip_batch_fingerprints(SimpleNamespace(results=[]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                chip_batch_fingerprints(SimpleNamespace(results=first.results * 2))

    def test_regression_cell_logs_failures_and_skips_separately(self):
        import sys
        for code, output, expected in ((0, "Ran 1 test\nOK\n", "PASSED"),
                                     (0, "Ran 1 test\nOK (skipped=1)\n", "INCOMPLETE (skips)"),
                                     (1, "FAILED (errors=1)\n", "FAILED")):
            with tempfile.TemporaryDirectory() as tmp:
                namespace = dict(RUN_REGRESSIONS=True, OUTPUT_BASE_DIR=Path(tmp),
                    datetime=datetime, repo_root=REPO_ROOT, sys=sys, print=lambda *a, **k: None)

                def run(command, **kwargs):
                    self.assertEqual(command[0], sys.executable)
                    self.assertEqual(kwargs["cwd"], REPO_ROOT)
                    kwargs["stdout"].write(output)
                    return SimpleNamespace(returncode=code)

                with patch("subprocess.run", side_effect=run) as runner:
                    if expected == "PASSED":
                        exec(notebook_cell("polar_regressions"), namespace)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "failures/skips"):
                            exec(notebook_cell("polar_regressions"), namespace)
                    self.assertEqual(runner.call_count, 2)
                summary = json.loads(next(Path(tmp).glob("*/summary.json")).read_text())
                self.assertEqual({item["status"] for item in summary.values()}, {expected})

    def test_disabled_validation_never_launches_work(self):
        with patch("subprocess.run") as run:
            exec(notebook_cell("polar_regressions"), dict(RUN_REGRESSIONS=False, print=lambda *a: None))
            run.assert_not_called()
        exec(notebook_cell("polar_replay"), dict(RUN_PARALLEL_REPLAY=False, print=lambda *a: None))

    def test_replay_uses_isolated_directories_and_rejects_mismatch(self):
        from lfm.data_processing.tests.chip.test_chip_creation import ChipCreationRasterTestCase
        for mismatch in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = ChipCreationRasterTestCase().config(root)
                calls = []

                def create(requests, replay_config, max_workers):
                    calls.append((replay_config, max_workers))
                    replay_config.output_root.mkdir(parents=True)
                    return max_workers

                namespace = dict(RUN_PARALLEL_REPLAY=True, REPLAY_WORKERS=2, requests=[1, 2],
                    OUTPUT_ROOT=root, datetime=datetime, chip_config=config, create_chips=create,
                    chip_batch_fingerprints=lambda workers: {"sample": workers if mismatch else "same"},
                    json=json, print=lambda *args: None)
                if mismatch:
                    with self.assertRaisesRegex(AssertionError, "outputs differ"):
                        exec(notebook_cell("polar_replay"), namespace)
                else:
                    exec(notebook_cell("polar_replay"), namespace)
                    self.assertEqual(len(list(root.glob("replay_*/comparison.json"))), 1)
                self.assertEqual([workers for _, workers in calls], [1, 2])
                self.assertNotEqual(calls[0][0].output_root, calls[1][0].output_root)
                for replay_config, _ in calls:
                    self.assertNotEqual(replay_config.intermediate_root, config.intermediate_root)
                    self.assertEqual(replay_config.acquisition_groups, config.acquisition_groups)
