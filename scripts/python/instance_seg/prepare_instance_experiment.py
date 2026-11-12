"""Prepare the matched 70-band experiment configuration and file-pair manifest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfm.all_models.inst_seg.paired_experiment import (
    create_paired_experiment_parser,
    prepare_experiment,
    resolve_paired_experiment,
)


def main() -> None:
    args = create_paired_experiment_parser().parse_args()
    experiment = resolve_paired_experiment(**vars(args))
    prepare_experiment(experiment, args.run_root)
    print(f"Prepared experiment {experiment.config_id} in {args.run_root}")
    print(
        "Saved experiment.json and split_manifest.json; no training or jobs submitted."
    )


if __name__ == "__main__":
    main()
