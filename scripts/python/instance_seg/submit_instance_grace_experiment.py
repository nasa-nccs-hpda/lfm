"""Prepare and submit the three-job Grace container experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfm.all_models.experiment_execution import submit
from lfm.all_models.inst_seg.grace_experiment import (
    parse_submission_args,
    submission_request,
)

if __name__ == "__main__":
    receipt = submit(submission_request(parse_submission_args()))
    print(f"Submission manifest: {receipt.manifest}", flush=True)
