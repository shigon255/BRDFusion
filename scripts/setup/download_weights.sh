#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

print_third_party_instructions() {
  cat <<'EOF'
Third-party weights should be installed from the vendored third-party projects.

Cosmos DiffusionRenderer:
  conda activate cosmos-predict1
  cd "$BRDFUSION_ROOT/third_party/cosmos1-diffusion-renderer"
  huggingface-cli login
  CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) \
    python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir checkpoints

DiffusionLight-Turbo:
  No explicit download is required. The wrappers pass DiffusionLight/ExposureLoRA
  and DiffusionLight/TurboLoRA to diffusers, which downloads them into the
  HuggingFace cache on first run. Run `huggingface-cli login` first if needed.

This writes the official DiffusionRenderer layout under:
  third_party/cosmos1-diffusion-renderer/checkpoints/
    Diffusion_Renderer_Inverse_Cosmos_7B/model.pt
    Diffusion_Renderer_Forward_Cosmos_7B/model.pt
    Cosmos-Tokenize1-CV8x8x8-720p/{model.pt,encoder.jit,decoder.jit,autoencoder.jit,mean_std.pt,image_mean_std.pt}

EOF
}

case "${1:-third_party}" in
  cosmos|diffusionlight|third_party|all)
    print_third_party_instructions
    ;;
  *)
    echo "Usage: $0 {third_party|cosmos|diffusionlight|all}" >&2
    exit 2
    ;;
esac
