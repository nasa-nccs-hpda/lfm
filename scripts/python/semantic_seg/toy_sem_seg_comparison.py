"""Compatibility wrapper for the renamed semantic comparison workflow.

Use ``scripts/python/semantic_seg/semantic_seg_comparison.py`` for new runs.
"""

from semantic_seg_comparison import *  # noqa: F401,F403
from semantic_seg_comparison import main

if __name__ == "__main__":
    main()
