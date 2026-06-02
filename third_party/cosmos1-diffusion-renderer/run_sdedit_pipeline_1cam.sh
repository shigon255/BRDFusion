#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_sdedit_pipeline_1cam.sh inverse
#   bash run_sdedit_pipeline_1cam.sh forward
#   bash run_sdedit_pipeline_1cam.sh rotate
#   bash run_sdedit_pipeline_1cam.sh sky
#   bash run_sdedit_pipeline_1cam.sh verify-inverse
#   bash run_sdedit_pipeline_1cam.sh verify-forward

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
DRY_RUN="${DRY_RUN:-0}"
DRY_FLAG=""
if [[ "$DRY_RUN" == "1" ]]; then
  DRY_FLAG="--dry_run"
fi

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-57}"
OVERLAP="${OVERLAP:-50}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
POST_RESIZE_HEIGHT="${POST_RESIZE_HEIGHT:-}"
POST_RESIZE_WIDTH="${POST_RESIZE_WIDTH:-}"
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

INVERSE_INPUT_ROOT="${INVERSE_INPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/intrinsic_refinement/raw_intrinsic}"
INVERSE_OUTPUT_ROOT="${INVERSE_OUTPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/intrinsic_refinement}"
FORWARD_INPUT_ROOT="${FORWARD_INPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/pbr_refinement/raw_render}"
FORWARD_OUTPUT_ROOT="${FORWARD_OUTPUT_ROOT:-$BRDFUSION_ROOT/outputs/drivestudio/test2/pbr_refinement}"
ROTATE_INPUT_ROOT="${ROTATE_INPUT_ROOT:-$FORWARD_INPUT_ROOT}"
ROTATE_OUTPUT_ROOT="${ROTATE_OUTPUT_ROOT:-$FORWARD_OUTPUT_ROOT}"
SKY_OUTPUT_ROOT="${SKY_OUTPUT_ROOT:-$FORWARD_OUTPUT_ROOT}"
SKY_VIDEO="${SKY_VIDEO:-$FORWARD_INPUT_ROOT/rgb_sky.mp4}"
OPACITY_VIDEO="${OPACITY_VIDEO:-$FORWARD_INPUT_ROOT/opacity.mp4}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-198}"
OFFLOAD_DIFFUSION_TRANSFORMER="${OFFLOAD_DIFFUSION_TRANSFORMER:-1}"
OFFLOAD_TOKENIZER="${OFFLOAD_TOKENIZER:-1}"
OFFLOAD_FLAGS=()
[[ "$OFFLOAD_DIFFUSION_TRANSFORMER" == "1" ]] && OFFLOAD_FLAGS+=(--offload_diffusion_transformer)
[[ "$OFFLOAD_TOKENIZER" == "1" ]] && OFFLOAD_FLAGS+=(--offload_tokenizer)

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

echo "[run_sdedit_pipeline_1cam] MODE=$MODE"
echo "[run_sdedit_pipeline_1cam] CALLER_CWD=$CALLER_CWD"
echo "[run_sdedit_pipeline_1cam] TMP_ROOT=$TMP_ROOT"

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

resolve_post_resize_defaults() {
  local ref_video="$1"
  local inferred_w=""
  local inferred_h=""

  if [[ -f "$ref_video" ]]; then
    local wh
    wh="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$ref_video" 2>/dev/null || true)"
    if [[ "$wh" == *x* ]]; then
      inferred_w="${wh%x*}"
      inferred_h="${wh#*x}"
    fi
  fi

  if [[ -z "$POST_RESIZE_WIDTH" ]]; then
    if [[ -n "$inferred_w" ]]; then
      POST_RESIZE_WIDTH="$inferred_w"
    else
      POST_RESIZE_WIDTH="$WIDTH"
    fi
  fi

  if [[ -z "$POST_RESIZE_HEIGHT" ]]; then
    if [[ -n "$inferred_h" ]]; then
      POST_RESIZE_HEIGHT="$inferred_h"
    else
      POST_RESIZE_HEIGHT="$HEIGHT"
    fi
  fi

  echo "[run_sdedit_pipeline_1cam] POST_RESIZE set to ${POST_RESIZE_WIDTH}x${POST_RESIZE_HEIGHT} (ref=${ref_video})"
}

case "$MODE" in
  inverse)
    resolve_post_resize_defaults "$INVERSE_INPUT_ROOT/gt_rgb.mp4"
    $PYTHON_BIN -u scripts/sdedit_pipeline_1cam.py inverse \
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
      "${OFFLOAD_FLAGS[@]}" \
      ${DRY_FLAG}
    ;;

  forward)
    resolve_post_resize_defaults "$FORWARD_INPUT_ROOT/pbr_rgb.mp4"
    $PYTHON_BIN -u scripts/sdedit_pipeline_1cam.py forward \
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
      "${OFFLOAD_FLAGS[@]}" \
      ${DRY_FLAG}
    ;;

  rotate)
    resolve_post_resize_defaults "$ROTATE_INPUT_ROOT/pbr_rgb.mp4"
    $PYTHON_BIN -u scripts/sdedit_pipeline_1cam.py forward \
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
      "${OFFLOAD_FLAGS[@]}" \
      ${DRY_FLAG}
    ;;

  sky)
    $PYTHON_BIN -u scripts/postprocess_sdedit_sky_1cam.py \
      --output_root "$SKY_OUTPUT_ROOT" \
      --strengths ${SWEEP_VALUES} \
      --sky_video "$SKY_VIDEO" \
      --opacity_video "$OPACITY_VIDEO" \
      --tmp_root "$TMP_ROOT" \
      ${DRY_FLAG}
    ;;

  verify-inverse)
    $PYTHON_BIN -u scripts/verify_sdedit_outputs_1cam.py inverse \
      --output_root "$INVERSE_OUTPUT_ROOT" \
      --strengths ${SWEEP_VALUES} \
      --expected_frames "$EXPECTED_FRAMES"
    ;;

  verify-forward)
    $PYTHON_BIN -u scripts/verify_sdedit_outputs_1cam.py forward \
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
