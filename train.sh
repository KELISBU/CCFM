#!/usr/bin/env bash
# ============================================================================
# Train the CCFM flow-matching model on nuScenes (via trajdata).
#
# CCFM = the flow-matching trajectory model used as the reactive agent.
# Training itself is plain flow matching; the constrained guidance (CCFM) and
# HCS event selection are applied later at simulation time
# (see nuscene_simulation.sh).
# ============================================================================
set -euo pipefail

# ---- Paths (edit for your machine) ----
DATASET=/path/to/nuscenes
OUTPUT_DIR=path/to/outputs

python scripts/train.py \
  --config_name nusc_flowmatching \
  --dataset_path "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --name ccfm_flowmatching
