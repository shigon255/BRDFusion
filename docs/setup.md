# Setup Guide

This guide is the recommended path for a new user starting from a fresh clone.
It separates source setup, environments, weights, datasets, validation, and the
first dry-run pipeline.

## 0. Prerequisites

Recommended platform:

- Linux workstation or server
- NVIDIA GPU with a CUDA 11.8-compatible driver
- Conda or Mamba
- `gcc-11` and `g++-11`
- `git`, `rsync`, `ffmpeg`, `cmake`, `ninja`
- enough disk for large checkpoints and datasets

Approximate storage expectations:

- Cosmos DiffusionRenderer checkpoints: about 60 GB
- DiffusionLight-Turbo checkpoints: depends on the source package
- Waymo processed scenes: tens of GB per selected group of scenes
- Self dataset: depends on number of paths/lights; keep it outside git under `data/`

All commands below assume:

```bash
cd /path/to/BRDFusion
export BRDFUSION_ROOT=$PWD
export CONDA_ROOT=/path/to/miniconda3/envs
export PYTHONPATH=$BRDFUSION_ROOT
```

## 1. Check Vendored Source

BRDFusion expects the external source projects to exist inside this repo:

```text
third_party/3dgrut/
third_party/cosmos1-diffusion-renderer/
third_party/DiffusionLight-Turbo/
```

Verify:

```bash
scripts/smoke/check_third_party_layout.sh
```

## 2. Build Environments

Create/use separate environments:

- main environment: training, rendering, evaluation
- `cosmos-predict1`: DiffusionRenderer inverse/forward/SDEdit
- `diffusionlight`: DiffusionLight-Turbo
- `segformer`: optional sky/dynamic mask generation

Install the main environment following [install.md](install.md). Install the
DiffusionRenderer and DiffusionLight environments following the vendored project
docs in:

```text
third_party/cosmos1-diffusion-renderer/
third_party/DiffusionLight-Turbo/
```

Then run:

```bash
PYTHON_BIN=$CONDA_ROOT/brdfusion/bin/python scripts/smoke/check_imports.sh
scripts/smoke/check_configs.sh
scripts/smoke/check_scripts.sh
```

## 3. Restore Weights

Cosmos DiffusionRenderer:

```bash
COSMOS_ENV=cosmos-predict1 scripts/setup/download_weights.sh cosmos
scripts/setup/check_assets.py --group cosmos
```

DiffusionLight-Turbo:

```bash
DIFFUSIONLIGHT_SOURCE=/path/to/prepared/DiffusionLight-Turbo/models \
  scripts/setup/download_weights.sh diffusionlight
scripts/setup/check_assets.py --group diffusionlight
```

Optional BRDFusion example checkpoints:

```bash
BRDFUSION_SOURCE=/path/to/brdfusion/checkpoints \
  scripts/setup/download_weights.sh brdfusion
```

See [assets.md](assets.md) for exact target paths.

## 4. Prepare Datasets

BRDFusion configs use:

```text
data/brdfusion/waymo/scenes/<scene_id>/
data/brdfusion/self/scenes/<canonical_scene_id>/
```

These scene directories may be real folders or symlinks.

For an existing prepared dataset:

```bash
BRDFUSION_WAYMO_SOURCE_ROOT=/path/to/waymo/processed/training \
BRDFUSION_SELF_SOURCE_ROOT=/path/to/self \
python tools/data/link_datasets.py \
  --layout configs/data/brdfusion_layout.yaml \
  --require_source
```

For a downloaded release archive, extract it under `data/sources/` and link:

```bash
mkdir -p data/sources
# Example target after extraction:
#   data/sources/self/path1_fixed_tree_gamma_full/<scene>/
#   data/sources/waymo/processed/training/<scene_id>/

BRDFUSION_SELF_SOURCE_ROOT=data/sources/self \
BRDFUSION_WAYMO_SOURCE_ROOT=data/sources/waymo/processed/training \
python tools/data/link_datasets.py --require_source
```

Inspect:

```bash
python tools/data/inspect_layout.py \
  data/brdfusion/self/scenes \
  data/brdfusion/waymo/scenes
```

Validate one self scene:

```bash
DATASET=self \
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

Validate one Waymo scene:

```bash
DATASET=waymo \
DATA_ROOT=data/brdfusion/waymo/scenes \
SCENE=3 \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

See [dataset_layout.md](dataset_layout.md) and [data.md](data.md).

## 5. Generate Missing Priors

If your dataset already includes DiffusionRenderer priors and HDR envmaps, skip
this step.

Self DiffusionRenderer inverse priors:

```bash
conda activate cosmos-predict1
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
NUM_TIMESTEPS=50 \
scripts/priors/run_dr_self.sh
```

Self DiffusionLight HDR priors:

```bash
conda activate diffusionlight
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
NUM_TIMESTEPS=50 \
scripts/priors/run_dl_self.sh
```

For Waymo, use `scripts/priors/run_dr_waymo.sh` and
`scripts/priors/run_dl_waymo.sh` with `SCENE=<numeric_id>`.

## 6. Dry-Run The Pipeline

The pipeline dry-run writes a manifest and prints commands. It does not launch
heavy training unless `--execute` is passed.

```bash
conda activate brdfusion
python tools/run_pipeline.py \
  --pipeline_config configs/pipeline/self_repro.yaml \
  --output_dir /tmp/brdfusion_pipeline_manifest
```

Review:

```bash
cat /tmp/brdfusion_pipeline_manifest/pipeline_manifest.json
```

## 7. Run A Small Training Smoke Test

Use a short frame window before launching a full run:

```bash
python tools/train.py \
  --config_file configs/omnire.yaml \
  --config_overlay configs/stage/stage1_raster.yaml \
  --output_root work_dirs \
  --project brdfusion_smoke \
  --run_name self_stage1_smoke \
  dataset=self/brdfusion_3cams \
  data.scene_idx=path1_qwantani_moon_noon_puresky_4k \
  data.data_root=data/brdfusion/self/scenes \
  data.start_timestep=0 \
  data.end_timestep=2 \
  trainer.optim.num_iters=2
```

If this succeeds, the main environment, dataset layout, config merge, and basic
training loop are wired correctly.

## 8. Run The Full Schedule

Use [pipeline.md](pipeline.md) for the staged BRDFusion schedule. The high-level
sequence is:

1. Stage 1-1 raster/material optimization
2. generative refinement of G-buffer priors
3. Stage 1-2 raster/material optimization with refined priors
4. Stage 2 light-only PBR optimization
5. Stage 3 joint fine-tuning
6. render/evaluate/generative-render outputs
