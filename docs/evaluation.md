# Evaluation

BRDFusion reports three metric groups:

- image quality
- intrinsic/G-buffer quality
- relighting image quality

For self data, all three groups are available. For Waymo, only image quality is
expected unless GT G-buffers are provided.

## Train/Test Split

Use `test_image_stride=10` for the standard split. For frames `0-50`, test
frames are `10, 20, 30, 40, 50`; all other frames are training views.

## Command

```bash
CONFIG=configs/omnire.yaml \
DATASET=self/brdfusion_3cams \
VIDEO_ROOT=/path/to/rendered/videos \
IMAGE_JSON=/path/to/image_metrics.json \
INTRINSIC_VIDEO_ROOT=/path/to/intrinsic/videos \
INTRINSIC_JSON=/path/to/intrinsic_metrics.json \
RELIGHT_VIDEO_ROOT=/path/to/relight/videos \
RELIGHT_JSON=/path/to/relight_metrics.json \
START=0 END=50 TEST_STRIDE=10 \
scripts/eval/compute_video_metrics.sh
```

Set `DATASET_SOURCE=external` to evaluate against the self shifted path when the
dataset config has `external_source` enabled.
