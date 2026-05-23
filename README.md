# BRDFusion

BRDFusion is a research codebase for PBR-aware 3D Gaussian scene
reconstruction, relighting, and generative rendering on driving scenes. It
builds on the DriveStudio/OmniRe codebase and adds:

- per-Gaussian PBR material properties: albedo, metallic, roughness, and normals
- physically based rendering through 3DGRT ray tracing
- HDR environment-map light optimization
- DiffusionRenderer inverse priors, generative refinement, and generative rendering
- DiffusionLight-Turbo HDR lighting priors
- staged optimization for geometry/material/light reconstruction
- relighting, local-light rendering, and dynamic-object editing utilities

This repository is organized as a self-contained workspace: source code is
vendored under `third_party/`, while large datasets, priors, checkpoints, and
outputs live in gitignored local folders.

## Repository Layout

```text
configs/        Method, dataset, stage, asset, and pipeline configs
datasets/       Dataset loaders and preprocessing helpers
models/         Gaussian models, renderers, losses, and trainers
scripts/        Stable shell entrypoints for setup, priors, training, eval
tools/          Python entrypoints and utility tools
third_party/    Vendored source for 3DGRT, DiffusionRenderer, DiffusionLight
docs/           Installation, data, pipeline, evaluation, and application docs
```

## Documentation

- [Setup guide](docs/setup.md)
- [Installation](docs/install.md)
- [Assets and weights](docs/assets.md)
- [Dataset layout](docs/dataset_layout.md)
- [Data and priors](docs/data.md)
- [Pipeline](docs/pipeline.md)
- [Evaluation](docs/evaluation.md)
- [Applications](docs/applications.md)
- [Troubleshooting](docs/troubleshooting.md)

## Quick Start

Start with the full [setup guide](docs/setup.md). The condensed sequence is:

1. build the environments from [docs/install.md](docs/install.md)
2. restore checkpoints using [docs/assets.md](docs/assets.md)
3. place or symlink datasets using [docs/dataset_layout.md](docs/dataset_layout.md)
4. validate with smoke checks
5. dry-run `configs/pipeline/self_repro.yaml`

Restore large assets into the local workspace:

```bash
scripts/setup/download_weights.sh cosmos
DIFFUSIONLIGHT_SOURCE=/path/to/DiffusionLight-Turbo/models \
  scripts/setup/download_weights.sh diffusionlight
scripts/setup/check_assets.py --group cosmos
```

Create local dataset links without copying large files:

```bash
BRDFUSION_WAYMO_SOURCE_ROOT=/path/to/waymo/processed/training \
BRDFUSION_SELF_SOURCE_ROOT=/path/to/self \
python tools/data/link_datasets.py \
  --layout configs/data/brdfusion_layout.yaml \
  --require_source
```

Inspect and validate the linked layout:

```bash
python tools/data/inspect_layout.py \
  data/brdfusion/waymo/scenes \
  data/brdfusion/self/scenes

DATASET=self \
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

Dry-run the reproduction pipeline before launching heavy jobs:

```bash
python tools/run_pipeline.py \
  --pipeline_config configs/pipeline/self_repro.yaml \
  --output_dir /tmp/brdfusion_pipeline_manifest
```

## Training

Stage configs are merged in this order:

```text
base config -> dataset config -> stage overlays -> CLI opts
```

Example stage-1 raster/material run:

```bash
python tools/train.py \
  --config_file configs/omnire.yaml \
  --config_overlay configs/stage/stage1_raster.yaml \
  --output_root work_dirs \
  --project brdfusion \
  --run_name stage1a \
  dataset=self/brdfusion_3cams \
  data.scene_idx=path1_qwantani_moon_noon_puresky_4k \
  data.data_root=data/brdfusion/self/scenes
```

The standard BRDFusion schedule is:

1. Stage 1-1: rasterized 3DGS geometry/material optimization
2. DiffusionRenderer generative refinement of G-buffer priors
3. Stage 1-2: rasterized optimization with refined priors
4. Stage 2: freeze geometry/material and optimize light with PBR rendering
5. Stage 3: low-LR joint fine-tuning with all losses

## Smoke Checks

Run these after editing configs, scripts, or docs:

```bash
scripts/smoke/check_scripts.sh
scripts/smoke/check_configs.sh
scripts/smoke/check_self_contained_paths.sh
scripts/smoke/check_third_party_layout.sh
scripts/smoke/check_repo_hygiene.sh
PYTHON_BIN=/path/to/python scripts/smoke/check_imports.sh
```

## Attribution

BRDFusion builds on DriveStudio/OmniRe and incorporates 3DGRT,
DiffusionRenderer, and DiffusionLight-Turbo as vendored source dependencies.
Please cite the corresponding projects when using this codebase.
