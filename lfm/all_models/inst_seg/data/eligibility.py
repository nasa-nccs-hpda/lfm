"""Skip invalid instance chips during normal loading, without a pixel pre-scan."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class SkippedInstanceChip:
    chip_id: str
    image_path: str
    label_path: str
    reason: str = "no_valid_pixels_in_crop"


class NoEligibleInstanceSamplesError(RuntimeError):
    """A complete traversal found no eligible samples in a required split."""


def _uncollated_samples(samples: list[Any]) -> list[Any]:
    """Keep skip records picklable and visible to the main process with workers."""
    return samples


class EligibleInstanceDataLoader(DataLoader):
    """Refill batches with eligible samples; persist skips in the main process.

    The eligible length is unknown until normal iteration, so this loader has no
    ``len``. Reporting an upper bound as its length would suppress Lightning's
    end-of-epoch validation when chips are skipped. All-invalid traversals raise.
    Only the final eligible batch may be smaller than the configured batch size.
    """

    def __init__(
        self,
        dataset,
        *,
        sample_collate_fn: Callable,
        skip_report_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.sample_collate_fn = sample_collate_fn
        self.skip_report_dir = (
            Path(skip_report_dir) if skip_report_dir is not None else None
        )
        self.skipped_chips: dict[str, SkippedInstanceChip] = {}
        self.eligible_chip_ids: set[str] = set()
        # Lightning can pass the original collator when reconstructing a loader.
        kwargs.pop("collate_fn", None)
        super().__init__(dataset, collate_fn=_uncollated_samples, **kwargs)

    def __len__(self):
        raise TypeError("Eligible batch count is determined during iteration.")

    @property
    def batch_count_upper_bound(self) -> int:
        """Nominal batches from membership only, for explicit scheduler budgeting."""
        return super().__len__()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        split = self.dataset.split_name
        pending = []
        observed_skips = {}
        observed_eligible = set()
        exhausted = False
        events = None
        try:
            for samples in super().__iter__():
                for sample in samples:
                    if isinstance(sample, SkippedInstanceChip):
                        self.skipped_chips[sample.chip_id] = sample
                        observed_skips[sample.chip_id] = asdict(sample)
                        if self.skip_report_dir is not None:
                            if events is None:
                                self.skip_report_dir.mkdir(parents=True, exist_ok=True)
                                events = (
                                    self.skip_report_dir / f"{split}_skips.jsonl"
                                ).open("a", encoding="utf-8")
                            events.write(json.dumps(asdict(sample)) + "\n")
                            events.flush()
                        continue
                    observed_eligible.add(sample["chip_id"])
                    self.eligible_chip_ids.add(sample["chip_id"])
                    pending.append(sample)
                    if len(pending) == self.batch_size:
                        batch = self.sample_collate_fn(pending)
                        pending = []
                        yield batch
            exhausted = True
            if pending and not self.drop_last:
                yield self.sample_collate_fn(pending)
            if not observed_eligible:
                raise NoEligibleInstanceSamplesError(
                    f"No eligible samples in required split {split!r}: "
                    "every loaded crop has no valid pixels."
                )
        finally:
            if events is not None:
                events.close()
            if self.skip_report_dir is not None:
                self.skip_report_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "split": split,
                    "complete_traversal": exhausted,
                    "eligible_chip_ids": sorted(observed_eligible),
                    "skipped_chips": list(observed_skips.values()),
                }
                # A sampled figure pass must not replace a full traversal's report.
                suffix = "" if exhausted else "_partial"
                path = self.skip_report_dir / f"{split}{suffix}.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(path)
