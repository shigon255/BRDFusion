# Applications

Application workflows are rendered from checkpoints and do not rewrite the
checkpoint unless a tool explicitly says so.

## Relighting

```bash
CKPT=/path/to/checkpoint_final.pth \
NEW_ENVMAP=/path/to/target_envmap.hdr \
CALIBRATE_ENVMAP=1 \
scripts/render/relight_checkpoint.sh
```

## Object Editing

Scene editing is moving toward JSON specs. Examples live in
`configs/application/`.

- `scene_edit_example.json`: delete, move, and insert operations.
- `asset_manifest_example.json`: exported dynamic asset metadata.

Current compatibility wrappers:

```bash
SOURCE_CKPT=/path/to/source/checkpoint_final.pth \
OUTPUT_PATH=/path/to/exported_assets \
scripts/assets/export_dynamic_assets.sh

TARGET_CKPT=/path/to/target/checkpoint_final.pth \
ASSET_SPEC=/path/to/exported_assets/manifest.json \
scripts/assets/insert_dynamic_assets.sh
```

## Night And Local Lights

Use `configs/application/local_light_example.json` as the stable spec target.
Existing render-time flags still flow through `tools/eval.py`.

## Dynamic Assets

For object insertion, first export an asset package from a source checkpoint,
then insert that package during target-scene evaluation:

```bash
SOURCE_CKPT=/path/to/source/checkpoint_final.pth \
OUTPUT_PATH=assets/checkpoints/brdfusion/exported_assets \
EXPORT_MODE=per_object \
scripts/assets/export_dynamic_assets.sh

TARGET_CKPT=/path/to/target/checkpoint_final.pth \
ASSET_SPEC=assets/checkpoints/brdfusion/exported_assets/manifest.json \
scripts/assets/insert_dynamic_assets.sh
```

The insertion path is eval-time only unless a tool explicitly writes a new
checkpoint.
