"""Count-aware percentage assignment of existing indivisible groups."""

import json
from pathlib import Path
import tempfile
import unittest

from lfm.data_processing.chip.chip_config import SimpleSplitConfig, SplitPercentages, split_config_from_dict
from lfm.data_processing.chip.chip_splits import plan_splits, SplitTargetWarning
from lfm.data_processing.chip.chip_publication import _split_policy_document
from lfm.data_processing.tests.chip import test_chip_splits as fixtures


class CountAwareSplitTestCase(unittest.TestCase):
    def requests(self, sizes, locks=None):
        helper = fixtures.ChipSplitTestCase()
        result = []
        for group, size in enumerate(sizes):
            start = len(result)
            result.extend(helper.request(start + i, group=f"block_{group}",
                                         split=(locks or {}).get(group)) for i in range(size))
        return result

    def test_reported_371_chip_layout_is_balanced_without_splitting_blocks(self):
        helper = fixtures.ChipSplitTestCase()
        requests = [helper.request(row * 7 + col,
            group=f"M1412665711CE_block4_chip256_r{row // 4}_c{col // 4}")
            for row in range(53) for col in range(7)]
        self.assertEqual(plan_splits(requests, SimpleSplitConfig(seed=42)).realized_counts,
                         {"train": 331, "val": 40, "test": 0})
        config = SimpleSplitConfig(seed=42, assignment_method="count_aware")
        plan = plan_splits(requests, config)
        self.assertEqual(plan.realized_counts, {"train": 296, "val": 39, "test": 36})
        self.assertEqual(plan, plan_splits(list(reversed(requests)), config))
        groups = {}
        for assignment in plan.assignments:
            groups.setdefault(assignment.split_group_key, set()).add(assignment.assigned_split)
        self.assertEqual(len(groups), 28)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertNotEqual(plan, plan_splits(requests, SimpleSplitConfig(seed=43, assignment_method="count_aware")))

    def test_populates_feasible_splits_even_with_tiny_percentages(self):
        for sizes in ([100, 1, 1], [1, 1, 1], [16] * 20):
            plan = plan_splits(self.requests(sizes), SimpleSplitConfig(
                SplitPercentages(.998, .001, .001), assignment_method="count_aware"))
            self.assertTrue(all(plan.realized_counts[s] > 0 for s in ("train", "val", "test")))

    def test_too_few_groups_warn_and_zero_percentage_is_not_filled(self):
        with self.assertWarns(SplitTargetWarning):
            plan = plan_splits(self.requests([4, 4]), SimpleSplitConfig(assignment_method="count_aware"))
        self.assertEqual(len(plan.warnings), 1)
        self.assertEqual(plan.warnings[0].code, "empty_percentage_split")
        plan = plan_splits(self.requests([4, 4]), SimpleSplitConfig(
            SplitPercentages(.5, 0, .5), assignment_method="count_aware"))
        self.assertEqual(plan.realized_counts, {"train": 4, "val": 0, "test": 4})
        self.assertFalse(plan.warnings)
        self.assertFalse(plan_splits([], SimpleSplitConfig(assignment_method="count_aware")).warnings)

    def test_explicit_and_prior_locks_never_move(self):
        requests = self.requests([16] * 10, locks={0: "train", 1: "test"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"samples": [{"split_group_key": "block_2", "assigned_split": "val"}]}))
            config = SimpleSplitConfig(assignment_method="count_aware", prior_manifest_path=path)
            plan = plan_splits(requests, config)
        expected = {"block_0": "train", "block_1": "test", "block_2": "val"}
        for assignment in plan.assignments:
            if assignment.split_group_key in expected:
                self.assertEqual(assignment.assigned_split, expected[assignment.split_group_key])
        with self.assertWarns(SplitTargetWarning):
            plan = plan_splits(self.requests([1, 1, 1], locks={0: "train", 1: "train", 2: "train"}),
                               SimpleSplitConfig(assignment_method="count_aware"))
        self.assertEqual(plan.realized_counts["train"], 3)

    def test_config_and_manifest_record_method(self):
        config = split_config_from_dict({"type": "simple", "percentages": {"train": .8, "val": .1, "test": .1},
                                         "assignment_method": "count_aware"})
        self.assertEqual(config.assignment_method, "count_aware")
        plan = plan_splits(self.requests([1] * 30), config)
        self.assertEqual(_split_policy_document(config, plan)["assignment_method"], "count_aware")
        with self.assertRaises(ValueError):
            SimpleSplitConfig(assignment_method="random_typo")
        self.assertEqual(SimpleSplitConfig().assignment_method, "hash")
