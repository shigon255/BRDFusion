# Data And Priors

This document describes how datasets and generated priors enter BRDFusion after
the local layout has been created. For folder contracts and symlinks, see
[dataset_layout.md](dataset_layout.md).

## Data Families

`self` scenes include:

- RGB renderings
- camera intrinsics and camera pose text files
- GT material/depth/normal buffers when available
- HDR envmaps for the synthetic light condition
- DiffusionRenderer G-buffer priors

`waymo` scenes include:

- RGB images
- camera/lidar calibration and ego poses
- sky/dynamic masks
- lidar scans and object metadata
- DiffusionRenderer G-buffer priors
- optional DiffusionLight envmap priors

## Waymo Raw Preprocessing

Waymo access requires accepting the dataset license through the official Waymo
Open Dataset distribution. Place raw TFRecords under:

```text
data/waymo/raw/
```

Preprocess selected scenes:

```bash
RAW_ROOT=data/waymo/raw \
TARGET_ROOT=data/waymo/processed \
SCENES="3 114" \
WORKERS=8 \
scripts/data/prepare_waymo.sh
```

Alternative selection modes:

```bash
RAW_ROOT=data/waymo/raw TARGET_ROOT=data/waymo/processed \
START=0 NUM_SCENES=4 scripts/data/prepare_waymo.sh

RAW_ROOT=data/waymo/raw TARGET_ROOT=data/waymo/processed \
SPLIT_FILE=data/waymo_example_scenes.txt scripts/data/prepare_waymo.sh
```

Then link processed scenes:

```bash
BRDFUSION_WAYMO_SOURCE_ROOT=data/waymo/processed/training \
python tools/data/link_datasets.py --dataset waymo --require_source
```

## Masks

If masks are missing, generate them with the segmentation environment.

Waymo:

```bash
conda activate segformer
DATA_ROOT=data/brdfusion/waymo/scenes \
SCENES="3 114" \
SEGFORMER_PATH=/path/to/SegFormer \
scripts/data/extract_waymo_masks.sh
```

Self sky masks:

```bash
conda activate segformer
DATA_ROOT=data/brdfusion/self/scenes \
SCENES="path1_qwantani_moon_noon_puresky_4k" \
SEGFORMER_PATH=/path/to/SegFormer \
scripts/data/extract_self_sky_masks.sh
```

Skip this step when downloaded/prepared scenes already include masks.

## DiffusionRenderer Priors

DiffusionRenderer inverse renderer predicts:

- `diffusion_renderer_normal`
- `diffusion_renderer_depth`
- `diffusion_renderer_albedo`
- `diffusion_renderer_roughness`
- `diffusion_renderer_metallic`

Self:

```bash
conda activate cosmos-predict1
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
NUM_TIMESTEPS=50 \
scripts/priors/run_dr_self.sh
```

Waymo:

```bash
conda activate cosmos-predict1
SCENE=3 \
NUM_TIMESTEPS=198 \
scripts/priors/run_dr_waymo.sh
```

DiffusionRenderer uses 57-frame windows. Long sequences are processed with
overlap and then trimmed back to the requested frame count.

## DiffusionLight Priors

DiffusionLight predicts HDR envmaps from RGB images. These priors regularize the
learned environment map during light optimization.

Self:

```bash
conda activate diffusionlight
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
NUM_TIMESTEPS=50 \
scripts/priors/run_dl_self.sh
```

Waymo:

```bash
conda activate diffusionlight
SCENE=3 \
NUM_TIMESTEPS=198 \
INTERVAL=10 \
scripts/priors/run_dl_waymo.sh
```

## Validation

After adding or generating priors, run:

```bash
DATASET=self \
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

For Waymo:

```bash
DATASET=waymo \
DATA_ROOT=data/brdfusion/waymo/scenes \
SCENE=3 \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```
