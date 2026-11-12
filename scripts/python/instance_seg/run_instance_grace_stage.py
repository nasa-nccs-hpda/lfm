"""Container-side allocation controller; the parent shell launches GPU work."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfm.all_models.experiment_execution import launch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("toy", "graha", "comparison"), required=True
    )
    args = parser.parse_args()
    sys.exit(launch(args.submission, args.stage))
