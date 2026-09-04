import json
from pathlib import Path
import tempfile
import unittest

from lfm.model.chip_config import (
    MixedPercentageNumberSplitConfig,
    NoSplitConfig,
    NumberSplitConfig,
    SimpleSplitConfig,
    SplitCounts,
    SplitPercentages,
)
from lfm.model.chip_splits import SplitTargetWarning, plan_splits
from lfm.model.chip_types import ChipRequest, GeographicAOI, TargetGrid


class ChipSplitTestCase(unittest.TestCase):
    def request(self, index, *, group=None, split=None):
        return ChipRequest(
            sample_id=f"M{index}_r0_c0",
            target_grid=TargetGrid(
                crs_wkt="GEOGCRS[Moon]",
                transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
                bounds=(0.0, 0.0, 2.0, 2.0),
                width=2,
                height=2,
            ),
            geographic_aoi=GeographicAOI(2.0, 0.0, 0.0, 2.0),
            split_group_key=str(index if group is None else group),
            assigned_split=split,
        )

    def assignment_map(self, requests, config):
        return {
            item.sample_id: item.assigned_split
            for item in plan_splits(requests, config).assignments
        }

    def test_simple_split_is_stable_to_order_and_unrelated_additions(self):
        config = SimpleSplitConfig(SplitPercentages(0.6, 0.2, 0.2), seed=17)
        requests = tuple(self.request(index) for index in range(30))

        original = self.assignment_map(requests, config)
        reversed_order = self.assignment_map(tuple(reversed(requests)), config)
        expanded = self.assignment_map(
            (*requests, *(self.request(index) for index in range(30, 40))),
            config,
        )

        self.assertEqual(original, reversed_order)
        self.assertEqual(original, {key: expanded[key] for key in original})

    def test_no_split_assigns_every_request_to_one_root_layout(self):
        requests = tuple(self.request(index) for index in range(5))

        plan = plan_splits(tuple(reversed(requests)), NoSplitConfig())

        self.assertEqual(plan.layout, "unsplit")
        self.assertEqual(plan.realized_counts, {"unsplit": 5})
        self.assertEqual(
            tuple(item.sample_id for item in plan.assignments),
            tuple(item.sample_id for item in requests),
        )
        self.assertEqual(
            {item.assigned_split for item in plan.assignments},
            {"unsplit"},
        )
        self.assertEqual(
            {item.source for item in plan.assignments},
            {"no_split"},
        )

    def test_no_split_rejects_caller_partition_assignment(self):
        with self.assertRaisesRegex(ValueError, "caller-assigned"):
            plan_splits((self.request(1, split="test"),), NoSplitConfig())

    def test_changed_seed_changes_automatic_membership(self):
        requests = tuple(self.request(index) for index in range(30))

        first = self.assignment_map(
            requests,
            SimpleSplitConfig(SplitPercentages(0.5, 0.25, 0.25), seed=1),
        )
        second = self.assignment_map(
            requests,
            SimpleSplitConfig(SplitPercentages(0.5, 0.25, 0.25), seed=2),
        )

        self.assertNotEqual(first, second)

    def test_groups_are_atomic_and_explicit_assignments_are_honored(self):
        requests = (
            self.request(1, group="site-a", split="val"),
            self.request(2, group="site-a"),
            self.request(3, group="site-b"),
        )
        plan = plan_splits(
            requests,
            SimpleSplitConfig(SplitPercentages(1.0, 0.0, 0.0)),
        )

        self.assertEqual(plan.assignment_for("M1_r0_c0").assigned_split, "val")
        self.assertEqual(plan.assignment_for("M2_r0_c0").assigned_split, "val")
        self.assertEqual(plan.assignment_for("M1_r0_c0").source, "explicit")

    def test_default_mixed_policy_fills_test_then_splits_remainder(self):
        requests = tuple(self.request(index) for index in range(120))

        plan = plan_splits(requests, MixedPercentageNumberSplitConfig())

        self.assertEqual(plan.realized_counts["test"], 100)
        self.assertEqual(
            plan.realized_counts["train"] + plan.realized_counts["val"],
            20,
        )
        self.assertFalse(plan.warnings)

    def test_insufficient_fixed_count_warns_without_failing(self):
        with self.assertWarns(SplitTargetWarning):
            plan = plan_splits(
                tuple(self.request(index) for index in range(3)),
                MixedPercentageNumberSplitConfig(),
            )

        self.assertEqual(plan.realized_counts["test"], 3)
        self.assertEqual(len(plan.warnings), 1)
        self.assertEqual(plan.warnings[0].requested_count, 100)
        self.assertEqual(plan.warnings[0].reason, "insufficient_samples")

    def test_fixed_priority_and_atomic_group_warning(self):
        requests = (
            self.request(1, group="pair-a"),
            self.request(2, group="pair-a"),
            self.request(3, group="pair-b"),
            self.request(4, group="pair-b"),
        )
        config = NumberSplitConfig(
            fixed_counts=SplitCounts(train=3),
            fixed_priority=("train",),
        )

        with self.assertWarns(SplitTargetWarning):
            plan = plan_splits(requests, config)

        self.assertIn(plan.realized_counts["train"], (2, 4))
        self.assertEqual(plan.warnings[0].reason, "atomic_split_groups")

    def test_number_targets_are_filled_in_configured_priority(self):
        requests = tuple(self.request(index) for index in range(7))
        config = NumberSplitConfig(
            fixed_counts=SplitCounts(train=3, test=2, val=1),
            fixed_priority=("train", "test", "val"),
        )

        plan = plan_splits(requests, config)

        self.assertEqual(plan.realized_counts, {"train": 3, "val": 1, "test": 2})
        self.assertEqual(
            sum(item.assigned_split is None for item in plan.assignments),
            1,
        )
        self.assertFalse(plan.warnings)

    def test_number_remainder_is_unassigned_or_explicitly_routed(self):
        requests = tuple(self.request(index) for index in range(5))
        unassigned = plan_splits(
            requests,
            NumberSplitConfig(
                fixed_counts=SplitCounts(test=2),
                fixed_priority=("test",),
            ),
        )
        routed = plan_splits(
            requests,
            NumberSplitConfig(
                fixed_counts=SplitCounts(test=2),
                fixed_priority=("test",),
                remainder_split="train",
            ),
        )

        self.assertEqual(
            sum(item.assigned_split is None for item in unassigned.assignments),
            3,
        )
        self.assertEqual(routed.realized_counts, {"train": 3, "val": 0, "test": 2})

    def test_prior_manifest_locks_membership_as_inventory_grows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "M1_r0_c0",
                                "split_group_key": "site-a",
                                "assigned_split": "val",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = SimpleSplitConfig(
                SplitPercentages(1.0, 0.0, 0.0),
                prior_manifest_path=path,
            )

            plan = plan_splits(
                (
                    self.request(1, group="site-a"),
                    self.request(2, group="site-a"),
                    self.request(3, group="site-b"),
                ),
                config,
            )

        self.assertEqual(plan.assignment_for("M1_r0_c0").assigned_split, "val")
        self.assertEqual(plan.assignment_for("M2_r0_c0").assigned_split, "val")
        self.assertEqual(plan.assignment_for("M3_r0_c0").assigned_split, "train")

    def test_seed_changes_only_automatic_not_explicit_or_prior_membership(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "M1_r0_c0",
                                "split_group_key": "prior-site",
                                "assigned_split": "val",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            requests = (
                self.request(1, group="prior-site"),
                self.request(2, group="explicit-site", split="test"),
                *(self.request(index) for index in range(3, 33)),
            )
            first = self.assignment_map(
                requests,
                SimpleSplitConfig(
                    SplitPercentages(0.5, 0.25, 0.25),
                    seed=1,
                    prior_manifest_path=path,
                ),
            )
            second = self.assignment_map(
                requests,
                SimpleSplitConfig(
                    SplitPercentages(0.5, 0.25, 0.25),
                    seed=2,
                    prior_manifest_path=path,
                ),
            )

        self.assertEqual(first["M1_r0_c0"], second["M1_r0_c0"])
        self.assertEqual(first["M1_r0_c0"], "val")
        self.assertEqual(first["M2_r0_c0"], second["M2_r0_c0"])
        self.assertEqual(first["M2_r0_c0"], "test")
        automatic_keys = set(first) - {"M1_r0_c0", "M2_r0_c0"}
        self.assertNotEqual(
            {key: first[key] for key in automatic_keys},
            {key: second[key] for key in automatic_keys},
        )


if __name__ == "__main__":
    unittest.main()
