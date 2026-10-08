import re
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.python.all_tasks.find_polar_nac import (
    NAC_PATTERN, candidates, longitude_span, worker_count, inspect_batch,
    inspection_waves, matches_filters,
)


class PolarNacDiscoveryTestCase(unittest.TestCase):
    def test_depth_case_filter_and_symlink_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = root / "child"
            child.mkdir()
            deep = child / "deep"
            deep.mkdir()
            for path in (root / "M123LE.TIF", child / "M456RE.tiff", deep / "NAC.tif",
                         root / "WAC.tif", root / "NAC.txt"):
                path.touch()
            (root / "NAC_link.tif").symlink_to(root / "M123LE.TIF")
            (child / "loop").symlink_to(root, target_is_directory=True)
            errors = []
            found = list(candidates([root], 2, re.compile(NAC_PATTERN, re.I), errors))
            self.assertEqual({p.name for p in found}, {"M123LE.TIF", "M456RE.tiff"})
            self.assertEqual(errors, [])
            self.assertEqual(len(list(candidates([root], 1, None, []))), 2)

    def test_missing_root_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            errors = []
            self.assertEqual(list(candidates([Path(temp) / "missing"], 10, None, errors)), [])
            self.assertEqual(len(errors), 1)

    def test_workers_from_slurm_and_override(self):
        with patch.dict('os.environ', {'SLURM_CPUS_PER_TASK': '16'}):
            self.assertEqual(worker_count(), 16)
            self.assertEqual(worker_count(2), 2)
        with self.assertRaises(ValueError):
            worker_count(0)

    def test_circular_span_and_filters(self):
        self.assertEqual(longitude_span([179, -179]), 2)
        self.assertEqual(longitude_span([20]), 0)
        self.assertEqual(longitude_span([-40, -20, -30]), 20)
        item = dict(hemispheres=['north'], longitude_span=20)
        self.assertTrue(matches_filters(item, 'north', 10))
        self.assertFalse(matches_filters(item, 'south', 10))
        self.assertFalse(matches_filters(item, 'both', 30))

    def test_batch_isolates_errors(self):
        with patch('scripts.python.all_tasks.find_polar_nac.inspect_raster',
                   side_effect=[{'path': 'a'}, ValueError('bad TIFF'), {'path': 'c'}]):
            results, errors = inspect_batch(['a', 'b', 'c'])
        self.assertEqual(results, [{'path': 'a'}, {'path': 'c'}])
        self.assertEqual(errors, [{'path': 'b', 'error': 'bad TIFF'}])

    def test_serial_batches_inspect_each_path_once(self):
        paths = [str(i) for i in range(7)]
        with patch('scripts.python.all_tasks.find_polar_nac.inspect_raster',
                   side_effect=lambda p: {'path': p}) as inspect:
            waves = list(inspection_waves(paths, 1, 3))
        self.assertEqual([len(r) for r, _ in waves], [3, 3, 1])
        self.assertEqual([call.args[0] for call in inspect.call_args_list], paths)

    def test_spawn_batches_account_for_all_failed_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [str(Path(temp) / f'missing_{i}.tif') for i in range(7)]
            waves = list(inspection_waves(paths, 2, 2))
        self.assertEqual([len(e) for _, e in waves], [4, 3])
        self.assertEqual([e['path'] for _, errors in waves for e in errors], paths)
        self.assertTrue(all(not results for results, _ in waves))


if __name__ == "__main__":
    unittest.main()
