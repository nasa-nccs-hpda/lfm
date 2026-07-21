#!/bin/bash

# Compatibility wrapper for the renamed semantic segmentation fine-tuning script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/sbatch_semantic_seg_finetuning.sh" "$@"

