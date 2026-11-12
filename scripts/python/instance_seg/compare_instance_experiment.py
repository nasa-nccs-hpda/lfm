"""Evaluate both explicit best-validation checkpoints on the full test split."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfm.all_models.inst_seg.comparison import (
    compare_checkpoints,
    create_comparison_parser,
)


def main():
    args = create_comparison_parser().parse_args()
    result = compare_checkpoints(**vars(args))
    print(f"Comparison complete; manifest: {result}", flush=True)


if __name__ == "__main__":
    main()
