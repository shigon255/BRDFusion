#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
fi

usage() {
    cat <<'EOF'
Usage:
  ./render_pipeline.sh inverse [args...]
  ./render_pipeline.sh forward [args...]

Inverse example:
  ./render_pipeline.sh inverse \
    --input_rgb asset/example/rgb.mp4 \
    --output_dir outputs/inverse \
    --checkpoint_dir checkpoints \
    --chunk_size 57 \
    --overlap 50 \
    --height 704 \
    --width 1280

Forward example:
  ./render_pipeline.sh forward \
    --basecolor outputs/inverse/basecolor.mp4 \
    --normal outputs/inverse/normal.mp4 \
    --depth outputs/inverse/depth.mp4 \
    --roughness outputs/inverse/roughness.mp4 \
    --metallic outputs/inverse/metallic.mp4 \
    --env_map asset/examples/hdri_examples/sunny_vondelpark_2k.hdr \
    --output_video outputs/relit.mp4 \
    --checkpoint_dir checkpoints \
    --chunk_size 57 \
    --overlap 50 \
    --height 704 \
    --width 1280

Notes:
  - This is a thin bash wrapper around scripts/render_pipeline.py.
  - Run './render_pipeline.sh <mode> --help' to see full Python-side arguments.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

MODE="$1"
shift

case "$MODE" in
    inverse|forward)
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        exit 1
        ;;
esac

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/render_pipeline.py" "$MODE" "$@"
