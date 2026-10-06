"""Check declarative A0 expectations, not the unimplemented production converter.

Run without geospatial dependencies:
    python3 -m unittest discover -s lfm/data_processing/tests/chip -t . -p test_chip_a0_fixtures.py -v
Rectangle fixtures provide an analytic oracle for later GDAL integration tests.
"""

import itertools
import json
import math
from lfm.data_processing._paths import REPO_ROOT
from pathlib import Path
import unittest


ROOT = REPO_ROOT
FIXTURES = ROOT / '.agents/planning_docs/aoi_chip_contract_fixtures.json'


class ChipA0FixtureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(FIXTURES.read_text())

    def test_outward_windows_preserve_source_lattice(self):
        for case in self.fixtures['grid_windows']:
            with self.subTest(case=case['name']):
                values = [round(v) if abs(v - round(v)) <= 1e-8 else v
                          for v in case['pixel_bounds']]
                x, y = math.floor(values[0]), math.floor(values[1])
                width, height = math.ceil(values[2]) - x, math.ceil(values[3]) - y
                self.assertEqual([x, y, width, height], case['window'])
                a, b, c, d, e, f = case['source_affine_gdal']
                self.assertEqual([a + b*x + c*y, b, c, d + e*x + f*y, e, f],
                                 case['output_affine_gdal'])

    def test_static_zone_selection_including_seam_and_ties(self):
        for case in self.fixtures['static_grids']:
            with self.subTest(case=case['name']):
                west, south, east, north = case['aoi_wsen']
                lon = (west + ((east - west) % 360) / 2 + 180) % 360 - 180
                zone = math.floor((lon + 180) / 8) + 1
                hemisphere = 'N' if (north + south) / 2 >= 0 else 'S'
                self.assertEqual(f'{zone}{hemisphere}', case['zone'])
                self.assertTrue((ROOT / f'TMS/RG/tms_LTM_{zone}{hemisphere}RG.json').is_file())

    def test_static_grid_uses_zero_anchor_and_100_metres(self):
        case = self.fixtures['static_rounding']
        left, bottom, right, top = case['projected_bounds_xyxy']
        left, bottom = math.floor(left/100)*100, math.floor(bottom/100)*100
        right, top = math.ceil(right/100)*100, math.ceil(top/100)*100
        self.assertEqual([left, bottom, right, top], case['output_bounds_xyxy'])
        self.assertEqual([(top-bottom)//100, (right-left)//100], case['shape_hw'])
        self.assertEqual([left, 100, 0, top, 0, -100], case['affine_gdal'])

    def test_vector_oracles_for_all_feature_orders(self):
        for case in self.fixtures['vector_cases']:
            height, width = case['shape_hw']
            for features in itertools.permutations(case['rectangles']):
                with self.subTest(case=case['name'], order=[v['id'] for v in features]):
                    mask = [[0]*width for _ in range(height)]
                    ids, boxes, subpixel = [], [], []
                    for feature in sorted(features, key=lambda v: v['id']):
                        x0, y0, x1, y1 = feature['bounds']
                        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        support = [(row, col) for row in range(height) for col in range(width)
                                   if x0 <= col+.5 <= x1 and y0 <= row+.5 <= y1]
                        if not support:
                            subpixel.append(feature['id'])
                            continue
                        ids.append(feature['id'])
                        boxes.append([x0, y0, x1-x0, y1-y0])
                        for row, col in support:
                            mask[row][col] = len(ids)
                    present = {v for row in mask for v in row}
                    occluded = [source for i, source in enumerate(ids, 1) if i not in present]
                    self.assertEqual(mask, case['mask'])
                    self.assertEqual(ids, case['ids'])
                    self.assertEqual(boxes, case['bboxes'])
                    self.assertEqual(occluded, case['occluded'])
                    self.assertEqual(subpixel, case['subpixel'])

    def test_raster_window_and_nearest_oracles(self):
        for case in self.fixtures['raster_cases']:
            with self.subTest(case=case['name']):
                source = case['source']
                if 'window_xywh' in case:
                    x, y, w, h = case['window_xywh']
                    result = [row[x:x+w] for row in source[y:y+h]]
                else:
                    # Analytically mapped centers for a +100 m false-easting shift.
                    result = [[source[math.floor(y)][math.floor(x)] for x, y in row]
                              for row in case['mapped_source_centers']]
                self.assertEqual(result, case['expected'])

    def test_nodata_counts_use_spatial_denominator(self):
        case = self.fixtures['nodata']
        bands = [[v for row in band for v in row] for band in case['valid_masks']]
        size = len(bands[0])
        counts = [sum(not value for value in band) for band in bands]
        union = sum(not all(values) for values in zip(*bands))
        self.assertEqual(counts, case['invalid_per_band'])
        self.assertEqual([100*c/size for c in counts], case['percent_per_band'])
        self.assertEqual(union, case['invalid_union'])
        self.assertEqual(100*union/size, case['percent_union'])


if __name__ == '__main__':
    unittest.main()
