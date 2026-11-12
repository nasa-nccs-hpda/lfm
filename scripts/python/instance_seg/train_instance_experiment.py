"""Train one Toy/GRAHA model and publish best/latest checkpoint provenance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfm.all_models.inst_seg.training import (
    create_training_parser,
    train_prepared_experiment,
)


def main():
    args = create_training_parser().parse_args()
    result = train_prepared_experiment(**vars(args))
    print(f"Training complete; result manifest: {result}", flush=True)


if __name__ == "__main__":
    main()
