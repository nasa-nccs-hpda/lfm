"""Static configuration must not initialize the model-training stack."""

import subprocess
import sys
import unittest
from pathlib import Path

from lfm.data_processing._paths import REPO_ROOT
from lfm.data_processing.tiling import (
    make_static_source, STATIC_BAND_NAMES, STATIC_OUTPUT_NODATA,
    MINIRF_SOURCE_NODATA, MINIRF_SOURCE_NODATA_BANDS,
)


class StaticSourceTestCase(unittest.TestCase):
    def test_canonical_configuration_and_overrides(self):
        source = make_static_source(data_dir="static", index_path="index.gpkg",
                                    index_layer="rasters", location_field="path", required=False)
        self.assertEqual(source.name, "static")
        self.assertEqual(source.data_dir, Path("static"))
        self.assertEqual(source.index_path, Path("index.gpkg"))
        self.assertEqual(source.index_layer, "rasters")
        self.assertEqual(source.location_field, "path")
        self.assertFalse(source.required)
        self.assertEqual(source.selection_mode, "all_intersecting")
        self.assertEqual(source.resampling, "bilinear")
        self.assertEqual(source.band_names, STATIC_BAND_NAMES)
        self.assertEqual(source.output_nodata, STATIC_OUTPUT_NODATA)
        self.assertEqual(tuple(o.band_name for o in source.band_nodata_overrides),
                         tuple(MINIRF_SOURCE_NODATA_BANDS))
        self.assertTrue(all(o.source_value == MINIRF_SOURCE_NODATA
                            for o in source.band_nodata_overrides))
        default = make_static_source(data_dir="static", index_path="index.gpkg")
        self.assertTrue(default.required)
        self.assertEqual(default.location_field, "location")

    def test_fresh_import_blocks_training_dependencies(self):
        code = '''
import sys
from importlib.abc import MetaPathFinder
class BlockTraining(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightning', 'pytorch_lightning'} or fullname.startswith('lfm.all_models'):
            raise AssertionError('Training import attempted: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
from lfm.data_processing.tiling import make_static_source
source = make_static_source(data_dir='static', index_path='index.gpkg')
assert len(source.band_names) == 63
'''
        subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       check=True, capture_output=True, text=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
