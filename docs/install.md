# Installation

BRDFusion uses multiple environments. This is intentional: the main training
stack, Cosmos DiffusionRenderer, DiffusionLight-Turbo, and segmentation tools
have different dependency constraints.

## Prerequisites

Recommended system:

- Linux
- NVIDIA GPU with a CUDA 11.8-compatible driver
- Conda or Mamba
- `gcc-11` and `g++-11`
- `git`, `rsync`, `ffmpeg`, `cmake`, `ninja`
- enough disk for checkpoints, processed data, and outputs

Set common paths:

```bash
cd /path/to/BRDFusion
export BRDFUSION_ROOT=$PWD
export CONDA_ROOT=/path/to/miniconda3/envs
export THREEDGRUT_ROOT=$BRDFUSION_ROOT/third_party/3dgrut
export DIFFUSION_RENDERER_ROOT=$BRDFUSION_ROOT/third_party/cosmos1-diffusion-renderer
export DIFFUSION_LIGHT_ROOT=$BRDFUSION_ROOT/third_party/DiffusionLight-Turbo
export PYTHONPATH=$BRDFUSION_ROOT
```

## Main Environment

The main environment runs training, evaluation, 3DGS rasterization, PBR
rendering, and 3DGRT tracing. The standard environment name is `brdfusion`.

```bash
conda create -n brdfusion python=3.10 -y
conda activate brdfusion
```

Install 3DGRT dependencies and the vendored 3DGRT package:

```bash
cd "$THREEDGRUT_ROOT"
conda install -y cuda-toolkit cmake ninja -c nvidia/label/cuda-11.8.0
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 "numpy<2.0" "mkl=2023.2.0" -c pytorch -c nvidia/label/cuda-11.8.0 -c conda-forge
pip3 install --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html kaolin==0.17.0
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 -y
git submodule update --init --recursive
python -m pip install --no-build-isolation -r requirements.txt
CC=gcc-11 CXX=g++-11 pip install -e .
```

Install BRDFusion dependencies:

```bash
cd "$BRDFUSION_ROOT"
python -m pip install --no-build-isolation -r requirements.txt
CC=gcc-11 CXX=g++-11 python -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
python -m pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
python -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
cd third_party/smplx
pip install -e .
```

Verify basic Python syntax and config loading:

```bash
cd "$BRDFUSION_ROOT"
PYTHON_BIN=$CONDA_ROOT/brdfusion/bin/python scripts/smoke/check_imports.sh
scripts/smoke/check_configs.sh
```

If `threedgrt_tracer` is missing, reinstall 3DGRT from
`third_party/3dgrut` in the active main environment.

## DiffusionRenderer Environment

Use a separate `cosmos-predict1` environment for inverse priors, generative
refinement, and generative rendering. Install it from the vendored
DiffusionRenderer project:

```bash
cd "$DIFFUSION_RENDERER_ROOT"
# Follow the vendored project's README/environment instructions.
```

After installing, verify that the environment can run the vendored downloader:

```bash
conda run -n cosmos-predict1 python \
  third_party/cosmos1-diffusion-renderer/scripts/download_diffusion_renderer_checkpoints.py \
  --help
```

Then restore weights as described in [assets.md](assets.md).

## DiffusionLight Environment

Use a separate `diffusionlight` environment for HDR envmap priors:

```bash
cd "$DIFFUSION_LIGHT_ROOT"
# Follow the vendored project's README/environment instructions.
```

BRDFusion expects the DiffusionLight model directories to be copied or placed
under `assets/checkpoints/diffusionlight/models/`; see [assets.md](assets.md).

## Segmentation Environment

Use a separate `segformer` environment if you need to generate sky or dynamic
masks. If your downloaded dataset already includes masks, this environment is
not required for training/evaluation.

Relevant wrappers:

```bash
scripts/data/extract_waymo_masks.sh
scripts/data/extract_self_sky_masks.sh
```

## Repository Checks

Run these after installing or editing the repo:

```bash
scripts/smoke/check_scripts.sh
scripts/smoke/check_self_contained_paths.sh
scripts/smoke/check_third_party_layout.sh
scripts/smoke/check_repo_hygiene.sh
PYTHON_BIN=$CONDA_ROOT/brdfusion/bin/python scripts/smoke/check_imports.sh
```

These checks do not prove that large weights and datasets are present. Use
[assets.md](assets.md) and [dataset_layout.md](dataset_layout.md) for that.
