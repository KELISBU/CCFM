# ---- Paths (edit for your machine) ----
DATASET=/path/to/nuscenes
OUTPUT_ROOT=path/to/results
CKPT_YAML=evaluation/CCFM.yaml

# ---- Shared arguments ----
COMMON_ARGS=(
  --dataset_path="${DATASET}"
  --env=nusc
  --eval_class=StrivePolicy_trajdata
  --agent_eval_class=FM
  --ckpt_yaml="${CKPT_YAML}"
  --render
  --scene_select_mode=collision_all
  # CCFM constrained guidance + HCS periodic event re-selection
  --ccfm
  --hcs_mode=periodic
  --hcs_freq=5
)

# ---- 80 horizon (default sim length) ----
python scripts/run_adv_simulation.py \
  --results_root_dir="${OUTPUT_ROOT}/CCFM_80" \
  --split_dataset \
  "${COMMON_ARGS[@]}"

# ---- 200 horizon ----
python scripts/run_adv_simulation.py \
  --results_root_dir="${OUTPUT_ROOT}/CCFM_200" \
  "${COMMON_ARGS[@]}" \
  --sim-steps=200
