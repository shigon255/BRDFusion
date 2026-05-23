# Dataset Layout

BRDFusion dataset configs use a single canonical local root:

```text
data/brdfusion/
  waymo/scenes/<scene_id>/
  self/scenes/<canonical_scene_id>/
  dataset_links.json
```

`data/` is gitignored. Scene directories can be normal folders or symlinks to
prepared data elsewhere on disk.

## Why This Layout

The research data originally used different roots for different tasks:

- Waymo: one processed split root with zero-padded scene ids such as `003`
- self data: path/light folders such as
  `path1_fixed_tree_gamma_full/qwantani_moon_noon_puresky_4k`
- relighted or shifted self data: companion folders that may contain only RGB
  images and calibration

The release layout keeps each scene's internal folder contract unchanged and
normalizes only the root and scene ids.

## Waymo Scene Contract

The loader reads `data_root/<scene_id>`. Numeric ids are zero-padded, so
`SCENE=3` resolves to `003`.

Expected folders for normal training/evaluation:

```text
images/                         # {frame:03d}_{cam}.jpg
sky_masks/                      # {frame:03d}_{cam}.png
dynamic_masks/ or fine_dynamic_masks/
lidar/                          # {frame:03d}.bin
ego_pose/                       # {frame:03d}.txt
extrinsics/                     # {cam}.txt
intrinsics/                     # {cam}.txt
instances/
diffusion_renderer_normal/      # {frame:03d}_{cam}.jpg
diffusion_renderer_depth/
diffusion_renderer_albedo/
diffusion_renderer_roughness/
diffusion_renderer_metallic/
```

## Self Scene Contract

The loader reads `data_root/<canonical_scene_id>`.

Expected folders/files:

```text
images/                         # {frame:03d}_{cam}.png
sky_masks/                      # {frame:03d}_{cam}.png
intrinsics/                     # Camera_Center.txt, Cam_Left.txt, Cam_Right.txt
Camera_Center_poses.txt
Cam_Left_poses.txt
Cam_Right_poses.txt
depth/                          # optional GT, {frame:03d}_{cam}.exr
normal/                         # optional GT
albedo/                         # optional GT
roughness/                      # optional GT
metallic/                       # optional GT
diffusion_renderer_normal/      # {frame:03d}_{cam}.jpg
diffusion_renderer_depth/
diffusion_renderer_albedo/
diffusion_renderer_roughness/
diffusion_renderer_metallic/
```

Relighted or shifted companion scenes can be linked under `self/scenes/`, but
they should not be used as primary training scenes unless they include the full
set of required priors and GT files.

## Link Prepared Data

The tracked manifest [configs/data/brdfusion_layout.yaml](../configs/data/brdfusion_layout.yaml)
defines canonical ids and source paths relative to a source root.

Development mode with existing prepared data:

```bash
BRDFUSION_WAYMO_SOURCE_ROOT=/path/to/waymo/processed/training \
BRDFUSION_SELF_SOURCE_ROOT=/path/to/self \
python tools/data/link_datasets.py \
  --layout configs/data/brdfusion_layout.yaml \
  --require_source
```

Release archive mode:

```bash
mkdir -p data/sources
# Extract self archive to data/sources/self/
# Extract or preprocess Waymo to data/sources/waymo/processed/training/

BRDFUSION_SELF_SOURCE_ROOT=data/sources/self \
BRDFUSION_WAYMO_SOURCE_ROOT=data/sources/waymo/processed/training \
python tools/data/link_datasets.py --require_source
```

Only link one dataset family:

```bash
python tools/data/link_datasets.py --dataset self --require_source
python tools/data/link_datasets.py --dataset waymo --require_source
```

Replace existing symlinks:

```bash
python tools/data/link_datasets.py --force --require_source
```

Dry-run:

```bash
python tools/data/link_datasets.py --dry_run
```

## Inspect And Validate

Inspect scene counts and linked targets:

```bash
python tools/data/inspect_layout.py \
  data/brdfusion/self/scenes \
  data/brdfusion/waymo/scenes
```

Validate a self scene:

```bash
DATASET=self \
DATA_ROOT=data/brdfusion/self/scenes \
SCENE=path1_qwantani_moon_noon_puresky_4k \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

Validate a Waymo scene:

```bash
DATASET=waymo \
DATA_ROOT=data/brdfusion/waymo/scenes \
SCENE=3 \
CAM_IDS="0 1 2" \
CHECK_PRIORS=1 \
scripts/data/check_dataset_layout.sh
```

## Dataset Configs

Use these dataset config names in training/evaluation:

- `dataset=self/brdfusion_3cams`
- `dataset=self/brdfusion_1cam`
- `dataset=waymo/brdfusion_3cams`

Example:

```bash
python tools/train.py \
  --config_file configs/omnire.yaml \
  dataset=self/brdfusion_3cams \
  data.scene_idx=path1_qwantani_moon_noon_puresky_4k \
  data.data_root=data/brdfusion/self/scenes
```
