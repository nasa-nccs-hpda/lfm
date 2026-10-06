import re
import tempfile
from pathlib import Path
import unittest

from scripts.python.all_tasks.find_polar_nac import NAC_PATTERN, candidates


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


if __name__ == "__main__":
    unittest.main()
