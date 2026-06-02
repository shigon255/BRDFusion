#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_sdedit_pipeline.sh inverse
#   bash run_sdedit_pipeline.sh forward
#   bash run_sdedit_pipeline.sh rotate
#   bash run_sdedit_pipeline.sh sky
#   bash run_sdedit_pipeline.sh verify-inverse
#   bash run_sdedit_pipeline.sh verify-forward
#
# Notes:
# - This script uses absolute output roots by default.
# - It runs from repo root automatically, so relative defaults resolve correctly.
# - Intermediate files are written under the caller's working directory by default.

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Missing mode. Use one of: inverse | forward | rotate | sky | verify-inverse | verify-forward"
  exit 1
fi

CALLER_CWD="$(pwd -P)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}"

# Toggle dry run (1 = print commands only, 0 = execute).
DRY_RUN="${DRY_RUN:-0}"
DRY_FLAG=""
if [[ "$DRY_RUN" == "1" ]]; then
  DRY_FLAG="--dry_run"
fi

# Shared defaults from spec/legacy scripts.
SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-57}"
OVERLAP="${OVERLAP:-50}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
POST_RESIZE_HEIGHT="${POST_RESIZE_HEIGHT:-640}"
POST_RESIZE_WIDTH="${POST_RESIZE_WIDTH:-960}"
GUIDANCE="${GUIDANCE:-0}"
GUIDANCE_START="${GUIDANCE_START:-}"
GUIDANCE_END="${GUIDANCE_END:-}"
NUM_STEPS="${NUM_STEPS:-15}"
S_CHURN="${S_CHURN:-0.0}"
S_NOISE="${S_NOISE:-1.0}"
S_TMIN="${S_TMIN:-0.0}"
S_TMAX="${S_TMAX:-inf}"
SCHEDULER_SIGMA_MIN="${SCHEDULER_SIGMA_MIN:-0.02}"
SCHEDULER_SIGMA_MAX="${SCHEDULER_SIGMA_MAX:-80.0}"
SCHEDULER_RHO="${SCHEDULER_RHO:-7.0}"
STRENGTHS="${STRENGTHS:-}"
SIGMAS="${SIGMAS:-}"
INTRINSICS="${INTRINSICS:-albedo normal normalized_depth roughness metallic}"

# Absolute roots (customize as needed).
INVERSE_INPUT_ROOT="${INVERSE_INPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/intrinsic_refinement/raw_intrinsic}"
INVERSE_OUTPUT_ROOT="${INVERSE_OUTPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/intrinsic_refinement}"
FORWARD_INPUT_ROOT="${FORWARD_INPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/pbr_refinement/raw_render}"
FORWARD_OUTPUT_ROOT="${FORWARD_OUTPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/pbr_refinement}"
ROTATE_INPUT_ROOT="${ROTATE_INPUT_ROOT:-$FORWARD_INPUT_ROOT}"
ROTATE_OUTPUT_ROOT="${ROTATE_OUTPUT_ROOT:-$FORWARD_OUTPUT_ROOT}"
SKY_OUTPUT_ROOT="${SKY_OUTPUT_ROOT:-$FORWARD_OUTPUT_ROOT}"
SKY_VIDEO="${SKY_VIDEO:-$FORWARD_INPUT_ROOT/rgb_sky.mp4}"
OPACITY_VIDEO="${OPACITY_VIDEO:-$FORWARD_INPUT_ROOT/opacity.mp4}"

# Set expected frames for verification.
EXPECTED_FRAMES="${EXPECTED_FRAMES:-198}"

# Intermediate workspace (task-scoped) under workflow output root.
# You can override:
#   TASK_ID=your_task_name
#   TMP_ROOT_BASE=/path/to/workspace
#   TMP_ROOT=/path/to/workspace/your_task_name
TASK_ID="${TASK_ID:-${MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "$MODE" == "inverse" || "$MODE" == "verify-inverse" ]]; then
  OUTPUT_ROOT_FOR_TMP="$INVERSE_OUTPUT_ROOT"
elif [[ "$MODE" == "forward" || "$MODE" == "verify-forward" ]]; then
  OUTPUT_ROOT_FOR_TMP="$FORWARD_OUTPUT_ROOT"
elif [[ "$MODE" == "rotate" ]]; then
  OUTPUT_ROOT_FOR_TMP="$ROTATE_OUTPUT_ROOT"
elif [[ "$MODE" == "sky" ]]; then
  OUTPUT_ROOT_FOR_TMP="$SKY_OUTPUT_ROOT"
else
  OUTPUT_ROOT_FOR_TMP="$CALLER_CWD"
fi
TMP_ROOT_BASE="${TMP_ROOT_BASE:-$OUTPUT_ROOT_FOR_TMP/.sdedit_tmp}"
TMP_ROOT="${TMP_ROOT:-$TMP_ROOT_BASE/$TASK_ID}"

echo "[run_sdedit_pipeline] MODE=$MODE"
echo "[run_sdedit_pipeline] CALLER_CWD=$CALLER_CWD"
echo "[run_sdedit_pipeline] TMP_ROOT=$TMP_ROOT"

if [[ -n "$GUIDANCE_START" && -z "$GUIDANCE_END" ]] || [[ -z "$GUIDANCE_START" && -n "$GUIDANCE_END" ]]; then
  echo "Both GUIDANCE_START and GUIDANCE_END must be provided together, or neither."
  exit 1
fi
GUIDANCE_SCHEDULE_FLAGS=()
if [[ -n "$GUIDANCE_START" && -n "$GUIDANCE_END" ]]; then
  GUIDANCE_SCHEDULE_FLAGS=(--guidance_start "$GUIDANCE_START" --guidance_end "$GUIDANCE_END")
fi

if [[ -n "$SIGMAS" && -n "$STRENGTHS" ]]; then
  echo "Provide only one of SIGMAS or STRENGTHS."
  exit 1
fi
if [[ -n "$SIGMAS" ]]; then
  SD_CONTROL_FLAGS=(--sigmas ${SIGMAS})
  SWEEP_VALUES="${SIGMAS}"
elif [[ -n "$STRENGTHS" ]]; then
  SD_CONTROL_FLAGS=(--strengths ${STRENGTHS})
  SWEEP_VALUES="${STRENGTHS}"
else
  SD_CONTROL_FLAGS=(--strengths 0.4)
  SWEEP_VALUES="0.4"
fi

case "$MODE" in
  inverse)
    $PYTHON_BIN -u scripts/sdedit_pipeline.py inverse \
      --input_root "$INVERSE_INPUT_ROOT" \
      --output_root "$INVERSE_OUTPUT_ROOT" \
      --tmp_root "$TMP_ROOT" \
      --intrinsics ${INTRINSICS} \
      "${SD_CONTROL_FLAGS[@]}" \
      --seed "$SEED" \
      --chunk_size "$CHUNK_SIZE" \
      --overlap "$OVERLAP" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --post_resize_height "$POST_RESIZE_HEIGHT" \
      --post_resize_width "$POST_RESIZE_WIDTH" \
      --guidance "$GUIDANCE" \
      "${GUIDANCE_SCHEDULE_FLAGS[@]}" \
      --num_steps "$NUM_STEPS" \
      --s_churn "$S_CHURN" \
      --s_noise "$S_NOISE" \
      --s_tmin "$S_TMIN" \
      --s_tmax "$S_TMAX" \
      --scheduler_sigma_min "$SCHEDULER_SIGMA_MIN" \
      --scheduler_sigma_max "$SCHEDULER_SIGMA_MAX" \
      --scheduler_rho "$SCHEDULER_RHO" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --offload_diffusion_transformer \
      --offload_tokenizer \
      ${DRY_FLAG}
    ;;

  forward)
    $PYTHON_BIN -u scripts/sdedit_pipeline.py forward \
      --input_root "$FORWARD_INPUT_ROOT" \
      --output_root "$FORWARD_OUTPUT_ROOT" \
      --tmp_root "$TMP_ROOT" \
      "${SD_CONTROL_FLAGS[@]}" \
      --seed "$SEED" \
      --chunk_size "$CHUNK_SIZE" \
      --overlap "$OVERLAP" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --post_resize_height "$POST_RESIZE_HEIGHT" \
      --post_resize_width "$POST_RESIZE_WIDTH" \
      --guidance "$GUIDANCE" \
      "${GUIDANCE_SCHEDULE_FLAGS[@]}" \
      --num_steps "$NUM_STEPS" \
      --s_churn "$S_CHURN" \
      --s_noise "$S_NOISE" \
      --s_tmin "$S_TMIN" \
      --s_tmax "$S_TMAX" \
      --scheduler_sigma_min "$SCHEDULER_SIGMA_MIN" \
      --scheduler_sigma_max "$SCHEDULER_SIGMA_MAX" \
      --scheduler_rho "$SCHEDULER_RHO" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --offload_diffusion_transformer \
      --offload_tokenizer \
      ${DRY_FLAG}
    ;;

  rotate)
    $PYTHON_BIN -u scripts/sdedit_pipeline.py forward \
      --input_root "$ROTATE_INPUT_ROOT" \
      --output_root "$ROTATE_OUTPUT_ROOT" \
      --tmp_root "$TMP_ROOT" \
      "${SD_CONTROL_FLAGS[@]}" \
      --seed "$SEED" \
      --chunk_size 57 \
      --overlap "$OVERLAP" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --post_resize_height "$POST_RESIZE_HEIGHT" \
      --post_resize_width "$POST_RESIZE_WIDTH" \
      --guidance "$GUIDANCE" \
      "${GUIDANCE_SCHEDULE_FLAGS[@]}" \
      --num_steps "$NUM_STEPS" \
      --s_churn "$S_CHURN" \
      --s_noise "$S_NOISE" \
      --s_tmin "$S_TMIN" \
      --s_tmax "$S_TMAX" \
      --scheduler_sigma_min "$SCHEDULER_SIGMA_MIN" \
      --scheduler_sigma_max "$SCHEDULER_SIGMA_MAX" \
      --scheduler_rho "$SCHEDULER_RHO" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --rotate_light \
      --offload_diffusion_transformer \
      --offload_tokenizer \
      ${DRY_FLAG}
    ;;

  sky)
    $PYTHON_BIN -u scripts/postprocess_sdedit_sky.py \
      --output_root "$SKY_OUTPUT_ROOT" \
      --strengths ${SWEEP_VALUES} \
      --sky_video "$SKY_VIDEO" \
      --opacity_video "$OPACITY_VIDEO" \
      --tmp_root "$TMP_ROOT" \
      ${DRY_FLAG}
    ;;

  verify-inverse)
    $PYTHON_BIN -u scripts/verify_sdedit_outputs.py inverse \
      --output_root "$INVERSE_OUTPUT_ROOT" \
      --strengths ${SWEEP_VALUES} \
      --expected_frames "$EXPECTED_FRAMES"
    ;;

  verify-forward)
    $PYTHON_BIN -u scripts/verify_sdedit_outputs.py forward \
      --output_root "$FORWARD_OUTPUT_ROOT" \
      --strengths ${SWEEP_VALUES} \
      --expected_frames "$EXPECTED_FRAMES"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Use one of: inverse | forward | rotate | sky | verify-inverse | verify-forward"
    exit 1
    ;;
esac
