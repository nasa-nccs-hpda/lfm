"""Container GPU entry point: record the allocation and run the requested work."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfm.all_models.experiment_execution import run_allocated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("toy", "graha", "comparison"), required=True
    )
    args, forwarded = parser.parse_known_args()
    run_allocated(args.stage, args.metadata, forwarded)


if __name__ == "__main__":
    main()
