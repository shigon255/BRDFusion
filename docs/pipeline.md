# Pipeline

`tools/run_pipeline.py` provides a manifest-driven view of the staged
BRDFusion workflow. By default it prints commands and writes
`pipeline_manifest.json`; it does not launch heavy jobs unless `--execute` is
passed.

## Dry Run

```bash
python tools/run_pipeline.py \
  --pipeline_config configs/pipeline/self_repro.yaml \
  --output_dir /tmp/brdfusion_pipeline_manifest
```

Review:

```bash
cat /tmp/brdfusion_pipeline_manifest/pipeline_manifest.json
```

## Stages

The default self reproduction pipeline contains:

1. `check_data`: validate the canonical dataset scene.
2. `inverse_prior_initial`: generate initial DiffusionRenderer G-buffer priors.
3. `light_prior`: generate DiffusionLight HDR envmap priors.
4. `train_stage1a`: rasterized 3DGS geometry/material training.
5. `gen_refine_intrinsics`: SDEdit-style generative G-buffer refinement.
6. `train_stage1b`: second raster/material pass with refined priors.
7. `train_stage2_light`: freeze geometry/material and optimize lighting with PBR color loss.
8. `train_stage3_finetune`: low-LR joint fine-tuning.
9. `render_eval`: render checkpoint outputs.
10. `gen_render`: generative rendering from PBR/G-buffers.
11. `compute_metrics`: compute image/G-buffer/relighting metrics.

## Config Merge Order

Training, rendering, and metrics use:

```text
base config -> dataset config -> --config_overlay files -> CLI opts
```

Example stage command:

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

## Stage Configs

Use these overlays for the standard schedule:

```text
configs/stage/stage1_raster.yaml
configs/stage/stage2_light.yaml
configs/stage/stage3_finetune.yaml
```

Method-level overlays live in:

```text
configs/method/pbr.yaml
configs/method/relighting.yaml
configs/method/local_lights.yaml
```

## Executing

After validating weights and data, execute the pipeline:

```bash
python tools/run_pipeline.py \
  --pipeline_config configs/pipeline/self_repro.yaml \
  --output_dir work_dirs/manifests/self_repro \
  --execute
```

For long runs, prefer launching individual manifest commands through your job
scheduler so each environment is explicit.

## Rendering Existing Checkpoints

Render a checkpoint:

```bash
CKPT=/path/to/checkpoint_final.pth \
POSTFIX=_eval \
START=0 \
NUM_FRAMES=50 \
scripts/render/render_checkpoint.sh
```

Relight a checkpoint:

```bash
CKPT=/path/to/checkpoint_final.pth \
NEW_ENVMAP=/path/to/target_envmap.hdr \
CALIBRATE_ENVMAP=1 \
scripts/render/relight_checkpoint.sh
```

Continue PBR training from a checkpoint:

```bash
CKPT=/path/to/checkpoint_final.pth \
OUTPUT_ROOT=work_dirs \
RUN_NAME=stage2_light \
DATASET=self/brdfusion_3cams \
DATA_ROOT=data/brdfusion/self/scenes \
SCENE_IDX=path1_qwantani_moon_noon_puresky_4k \
CONFIG_OVERLAYS=configs/stage/stage2_light.yaml \
scripts/train/train_pbr_from_ckpt.sh
```
