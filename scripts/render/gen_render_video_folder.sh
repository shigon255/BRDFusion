#!/usr/bin/env bash
set -euo pipefail

# Run DiffusionRenderer forward SDEdit on an existing tools/eval.py video folder.
# Usage:
#   VIDEO_ROOT=/path/to/render/<task>/videos scripts/render/gen_render_video_folder.sh
#   scripts/render/gen_render_video_folder.sh /path/to/render/<task>/videos

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation gen_render --exec "${SCRIPT_PATH}" "$@"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

VIDEO_ROOT="${VIDEO_ROOT:-${1:-}}"
if [[ -z "${VIDEO_ROOT}" ]]; then
  echo "[ERR] Usage: VIDEO_ROOT=/path/to/render/<task>/videos scripts/render/gen_render_video_folder.sh" >&2
  exit 1
fi

resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}

strength_tag() {
  local first="${1%% *}"
  python3 - "$first" <<'PY'
import sys
value = sys.argv[1]
try:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
except ValueError:
    text = value
print(text or "0")
PY
}

VIDEO_ROOT="$(resolve_path "${VIDEO_ROOT}")"
[[ -d "${VIDEO_ROOT}" ]] || { echo "[ERR] VIDEO_ROOT does not exist: ${VIDEO_ROOT}" >&2; exit 1; }

CAM_IDS="${CAM_IDS:-0}"
if [[ -n "${SIGMAS:-}" && -n "${STRENGTHS+x}" ]]; then
  echo "[ERR] Provide only one of STRENGTHS or SIGMAS." >&2
  exit 1
fi
if [[ -z "${SIGMAS:-}" ]]; then
  STRENGTHS="${STRENGTHS:-0.5}"
else
  STRENGTHS=""
fi
TAG_SOURCE="${STRENGTHS:-${SIGMAS}}"
TAG="${GEN_RENDER_TAG:-$(strength_tag "${TAG_SOURCE}")}"
PARENT_ROOT="$(cd "${VIDEO_ROOT}/.." && pwd)"
RAW_ROOT="$(resolve_path "${RAW_ROOT:-${PARENT_ROOT}/raw_render}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT:-${PARENT_ROOT}}")"
FINAL_OUTPUT_ROOT="${OUTPUT_ROOT}/refined_render_w${TAG}_sky"
PYTHON_BIN="${PYTHON_BIN:-python}"
COSMOS_ROOT="${COSMOS_ROOT:-${DIFFUSION_RENDERER_ROOT:-${REPO_ROOT}/third_party/cosmos1-diffusion-renderer}}"
COPY_FLAG=()
[[ "${COPY_INPUTS:-0}" == "1" ]] && COPY_FLAG=(--copy)

cam_count="$(python3 - "$CAM_IDS" <<'PY'
import sys
print(len([item for item in sys.argv[1].split() if item]))
PY
)"
if [[ -n "${ONE_CAM:-}" ]]; then
  if [[ "${ONE_CAM}" == "1" ]]; then
    SDEDIT_SCRIPT="run_sdedit_pipeline_1cam.sh"
  else
    SDEDIT_SCRIPT="run_sdedit_pipeline.sh"
  fi
elif (( cam_count == 1 )); then
  SDEDIT_SCRIPT="run_sdedit_pipeline_1cam.sh"
else
  SDEDIT_SCRIPT="run_sdedit_pipeline.sh"
fi

required_keys=(pbr_colors albedos normals normalized_depths roughnesses metallics rgb_sky opacities)
for key in "${required_keys[@]}"; do
  if ! find "${VIDEO_ROOT}" -type f \( -name "*_${key}.mp4" -o -name "${key}.mp4" \) -print -quit | grep -q .; then
    echo "[ERR] Could not find required ${key} video under ${VIDEO_ROOT}" >&2
    exit 1
  fi
done
[[ -f "${VIDEO_ROOT}/envmap.hdr" ]] || { echo "[ERR] Missing rendered envmap: ${VIDEO_ROOT}/envmap.hdr" >&2; exit 1; }
[[ -d "${COSMOS_ROOT}" ]] || { echo "[ERR] COSMOS_ROOT does not exist: ${COSMOS_ROOT}" >&2; exit 1; }
[[ -f "${COSMOS_ROOT}/${SDEDIT_SCRIPT}" ]] || { echo "[ERR] Missing SDEdit script: ${COSMOS_ROOT}/${SDEDIT_SCRIPT}" >&2; exit 1; }

echo "[gen_render_video_folder] VIDEO_ROOT=${VIDEO_ROOT}"
echo "[gen_render_video_folder] RAW_ROOT=${RAW_ROOT}"
echo "[gen_render_video_folder] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[gen_render_video_folder] FINAL_OUTPUT_ROOT=${FINAL_OUTPUT_ROOT}"
echo "[gen_render_video_folder] CAM_IDS=${CAM_IDS}"
echo "[gen_render_video_folder] SDEDIT_SCRIPT=${SDEDIT_SCRIPT}"

"${PYTHON_BIN}" tools/prepare_sdedit_inputs.py \
  --mode forward \
  --prefer "${PREFER_VIDEO_TOKEN:-raw_render}" \
  --video_root "${VIDEO_ROOT}" \
  --output_root "${RAW_ROOT}" \
  --cam_ids ${CAM_IDS} \
  --envmap_path "${VIDEO_ROOT}/envmap.hdr" \
  "${COPY_FLAG[@]}"

if (( cam_count > 1 )); then
  for cam_id in ${CAM_IDS}; do
    dst="${RAW_ROOT}/envmap_${cam_id}.hdr"
    if [[ ! -e "${dst}" && ! -L "${dst}" ]]; then
      ln -s "${VIDEO_ROOT}/envmap.hdr" "${dst}" 2>/dev/null || cp "${VIDEO_ROOT}/envmap.hdr" "${dst}"
    fi
  done
fi

(
  cd "${COSMOS_ROOT}"
  FORWARD_INPUT_ROOT="${RAW_ROOT}" \
  FORWARD_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  EXPECTED_FRAMES="${EXPECTED_FRAMES:-0}" \
  STRENGTHS="${STRENGTHS}" \
  SIGMAS="${SIGMAS:-}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DRY_RUN="${DRY_RUN:-0}" \
    bash "${SDEDIT_SCRIPT}" forward

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[gen_render_video_folder] DRY_RUN=1: skipping sky postprocess execution because forward dry-run does not materialize refined_render_w${TAG}."
  else
    SKY_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    SKY_VIDEO="${RAW_ROOT}/rgb_sky.mp4" \
    OPACITY_VIDEO="${RAW_ROOT}/opacity.mp4" \
    STRENGTHS="${STRENGTHS}" \
    SIGMAS="${SIGMAS:-}" \
    PYTHON_BIN="${PYTHON_BIN}" \
      bash "${SDEDIT_SCRIPT}" sky
  fi
)
