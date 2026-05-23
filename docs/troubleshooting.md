# Troubleshooting

Use this page when a setup step fails. For the happy-path sequence, start with
[setup.md](setup.md).

## Python Imports

Set `PYTHONPATH` to the repo root:

```bash
cd /path/to/BRDFusion
export PYTHONPATH=$PWD
```

Check syntax/import-time dependencies:

```bash
PYTHON_BIN=/path/to/env/bin/python scripts/smoke/check_imports.sh
```

If `threedgrt_tracer` or `threedgrut` is missing, activate the main environment
and reinstall the vendored 3DGRT package:

```bash
cd third_party/3dgrut
CC=gcc-11 CXX=g++-11 pip install -e .
```

## Native Extension Builds

Common symptoms:

- `ModuleNotFoundError: No module named torch` during a build
- CUDA extension build failures
- compiler version errors

Use no-build-isolation for packages that compile against the active PyTorch:

```bash
python -m pip install --no-build-isolation -r requirements.txt
CC=gcc-11 CXX=g++-11 pip install -e third_party/3dgrut
```

Confirm the active environment has the intended PyTorch/CUDA stack:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

## Configs

Run this after editing configs:

```bash
scripts/smoke/check_configs.sh
```

Config precedence is:

```text
base config -> dataset config -> --config_overlay files -> CLI opts
```

If a CLI override appears ignored, check whether a later `--config_overlay` or
CLI option overrides the same key.

## Data

Inspect linked scenes:

```bash
python tools/data/inspect_layout.py \
  data/brdfusion/self/scenes \
  data/brdfusion/waymo/scenes
```

Validate self data:

```bash
DATASET=self \
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

Validate Waymo data:

```bash
DATASET=waymo \
DATA_ROOT=data/brdfusion/waymo/scenes \
SCENE=3 \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

If a scene is missing, recreate symlinks:

```bash
python tools/data/link_datasets.py --force --require_source
```

If a prior directory is missing, either generate priors with
`scripts/priors/run_dr_*.sh` or disable prior loading only for the specific
debug run.

## Weights

Check local checkpoint placement:

```bash
scripts/setup/check_assets.py --group cosmos
scripts/setup/check_assets.py --group diffusionlight
```

If Cosmos files are missing, run:

```bash
COSMOS_ENV=cosmos-predict1 scripts/setup/download_weights.sh cosmos
```

If DiffusionLight files are missing, prepare the model directory from the
DiffusionLight release and copy it:

```bash
DIFFUSIONLIGHT_SOURCE=/path/to/DiffusionLight-Turbo/models \
  scripts/setup/download_weights.sh diffusionlight
```

## Out Of Memory

For debug runs:

- reduce `data.end_timestep`
- use `dataset=self/brdfusion_1cam`
- reduce `trainer.tracer.max_num_rays`
- set `trainer.optim.num_iters` to a small value

Example:

```bash
python tools/train.py \
  --config_file configs/omnire.yaml \
  --config_overlay configs/stage/stage1_raster.yaml \
  --output_root work_dirs \
  --project brdfusion_smoke \
  --run_name oom_debug \
  dataset=self/brdfusion_1cam \
  data.scene_idx=path1_qwantani_moon_noon_puresky_4k \
  data.data_root=data/brdfusion/self/scenes \
  data.start_timestep=0 \
  data.end_timestep=2 \
  trainer.optim.num_iters=2 \
  trainer.tracer.max_num_rays=16
```

## Heavy Jobs

Use dry-run manifests first:

```bash
python tools/run_pipeline.py \
  --pipeline_config configs/pipeline/self_repro.yaml \
  --output_dir /tmp/brdfusion_pipeline_manifest
```

Review `pipeline_manifest.json` before adding `--execute`.
