"""Regression checks for the canonical data-processing package layout."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lfm.data_processing._paths import REPO_ROOT


class PackageLayoutTestCase(unittest.TestCase):
    def run_python(self, source, *, pythonpath=REPO_ROOT):
        with tempfile.TemporaryDirectory() as working_dir:
            env = dict(os.environ, PYTHONPATH=str(pythonpath))
            result = subprocess.run(
                [sys.executable, "-c", source], cwd=working_dir,
                env=env, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_public_exports_share_canonical_class_identity(self):
        self.run_python("""
import sys
from lfm.data_processing import chip, tiling
from lfm.data_processing.chip.chip_config import ChipConfig, TileConfig as ChipTileConfig
from lfm.data_processing.tiling.tiling_config import TileConfig

assert chip.ChipConfig is ChipConfig
assert tiling.TileConfig is TileConfig
assert tiling.TileConfig is ChipTileConfig
for package in (chip, tiling):
    assert all(hasattr(package, name) for name in package.__all__)
assert not any(name == 'model' or name.startswith(('model.', 'lfm.model'))
               for name in sys.modules)
""")

    def test_tiling_import_does_not_load_chip_or_widget_modules(self):
        self.run_python("""
import sys
from lfm.data_processing.tiling import TileConfig
assert not any(name.startswith(('lfm.data_processing.chip',
                               'lfm.data_processing.labeling',
                               'lfm.data_processing.clustering', 'ipywidgets'))
               for name in sys.modules)
""")

    def test_resources_resolve_from_unrelated_working_directory(self):
        self.run_python("""
from lfm.data_processing._paths import REPO_ROOT
from lfm.data_processing.tiling.lunar_crs import load_lunar_geographic_wkt
from lfm.data_processing.tiling.grid_registry import default_grid_registry
assert 'Moon' in load_lunar_geographic_wkt()
assert default_grid_registry() is not None
assert (REPO_ROOT / 'notebooks').is_dir()
""")

    def test_checkout_name_does_not_determine_package_name(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "renamed-checkout"
            checkout.symlink_to(REPO_ROOT, target_is_directory=True)
            self.run_python("""
from lfm.data_processing.chip import ChipConfig
from lfm.data_processing.tiling import TileConfig
from lfm.data_processing._paths import REPO_ROOT
assert (REPO_ROOT / 'TMS' / 'IAU_30100_2015.wkt').is_file()
""", pythonpath=checkout)

    def test_types_roundtrip_through_spawned_worker(self):
        self.run_python("""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from lfm.data_processing.tiling.grid_registry import GridFamily
with ProcessPoolExecutor(max_workers=1,
                         mp_context=multiprocessing.get_context('spawn')) as pool:
    assert pool.submit(type, GridFamily.LTM).result() is GridFamily
""")


if __name__ == "__main__":
    unittest.main()
