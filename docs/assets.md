# Assets And Weights

Large files are not tracked by git. BRDFusion expects them inside local,
gitignored folders:

```text
assets/checkpoints/cosmos/
assets/checkpoints/diffusionlight/
assets/checkpoints/brdfusion/
data/
priors/
work_dirs/
outputs/
```

The tracked manifest [configs/assets/weights.yaml](../configs/assets/weights.yaml)
records expected checkpoint paths and selected file sizes.

## Cosmos DiffusionRenderer

Required for:

- inverse G-buffer priors from RGB
- generative refinement of G-buffer priors
- generative rendering from PBR/G-buffer outputs

Download into BRDFusion:

```bash
COSMOS_ENV=cosmos-predict1 scripts/setup/download_weights.sh cosmos
```

Expected target:

```text
assets/checkpoints/cosmos/
  Diffusion_Renderer_Inverse_Cosmos_7B/model.pt
  Diffusion_Renderer_Forward_Cosmos_7B/model.pt
  Cosmos-Tokenize1-CV8x8x8-720p/
```

Verify:

```bash
scripts/setup/check_assets.py --group cosmos
```

For a slower but stronger check of files that have hashes in the manifest:

```bash
scripts/setup/check_assets.py --group cosmos --verify_md5
```

## DiffusionLight-Turbo

Required for HDR environment-map priors from input RGB images.

BRDFusion does not currently fetch these weights from a public URL by itself.
Prepare or download the DiffusionLight-Turbo `models/` directory according to
the DiffusionLight release instructions, then copy it into BRDFusion:

```bash
DIFFUSIONLIGHT_SOURCE=/path/to/prepared/DiffusionLight-Turbo/models \
  scripts/setup/download_weights.sh diffusionlight
```

Expected target:

```text
assets/checkpoints/diffusionlight/models/
  ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500/
  rev3/Flickr2K/Flickr2kPlus_extended/checkpoint-230000/
```

Verify:

```bash
scripts/setup/check_assets.py --group diffusionlight
```

## BRDFusion Example Checkpoints

Optional released checkpoints should be placed under:

```text
assets/checkpoints/brdfusion/
```

Copy a prepared checkpoint folder:

```bash
BRDFUSION_SOURCE=/path/to/brdfusion/checkpoints \
  scripts/setup/download_weights.sh brdfusion
```

The `brdfusion` group is optional and has no required files in the manifest by
default.

## Datasets

Datasets are handled separately from weights. Use:

```bash
scripts/setup/download_datasets.sh self
scripts/setup/download_datasets.sh waymo
```

These commands print placement instructions because dataset access often
requires external credentials, license acceptance, or a separate release archive.
After placing data, link it into the canonical BRDFusion layout using
[dataset_layout.md](dataset_layout.md).
