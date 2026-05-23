# SDEdit Integration Spec (Draft v1)

## 1) Objective
- Build DiffusionRenderer-side automation for two workflows:
  - SDEdit inverse (refine intrinsic passes)
  - SDEdit forward (refine rendered PBR RGB)
- DriveStudio-side data export is out of scope for this phase.

## 2) Projects and Runtime
- Project A: DriveStudio
  - Path: `/project2/yi-ray/BRDFusion`
  - Role in this phase: only assumed to provide input files in agreed folders.
- Project B: DiffusionRenderer
  - Path: `/project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer`
  - Role in this phase: run SDEdit inverse/forward and produce outputs.
- Runtime model:
  - Run each project from its own repo root.
  - Use separate conda envs via `conda run -n ...`.
  - For DiffusionRenderer execution, preferred cwd is:
    - `/project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer`
  - Reason:
    - current scripts rely on `PYTHONPATH=$(pwd)` and in-repo imports like `cosmos_predict1...`.
  - Absolute path support:
    - input/output/envmap arguments can be absolute paths.
    - but import/runtime assumptions still make DiffusionRenderer repo root the safe cwd.

## 3) Required Reading for Next Session
The next session should read these files before implementing.

### 3.1 Core SDEdit scripts (must read)
- `sdedit_inverse_renderer.py`
- `sdedit_forward_renderer.py`
- `run_scene_inv_full.sh`
- `run_scene_full.sh`
- `split.py`
- `merge_overlap_videos.py`

### 3.2 Current bridge/legacy scripts (read for compatibility)
- `copy_and_split.sh`
- `copy_dr_intrinsic.sh`
- `convert_sdedit_to_dataset.py`
- `merge_overlap_all.sh`
- `run_scene.sh`
- `run_scene_inv.sh`
- `run_scene_rotate.sh`
- `merge.py`
- `resize_vid.py`

### 3.3 Usually skip (unless debugging)
- Large artifacts/outputs:
  - `drivestudio_exp/**`
  - `output/**`
  - `asset/example_results/**`
  - `checkpoints/**`
- Unrelated training framework internals not touched by this integration.

## 4) Shared Assumptions and Defaults
- Camera streams are encoded as one horizontal concat video of 3 cameras.
- Split logic follows `split.py` convention:
  - left -> cam 1
  - middle -> cam 0
  - right -> cam 2
- Sliding window behavior follows existing scripts:
  - chunk size: `57`
  - overlap: `50`
  - step: `7`
  - per-chunk padding behavior:
    - current `sdedit_inverse_renderer.py` and `sdedit_forward_renderer.py` call `_repeat_last_frame(...)`
    - if a chunk has fewer than 57 frames, they pad by repeating the last available frame.
  - important note:
    - current legacy merge scripts do not automatically trim final outputs back to original video length.
    - this integration requires explicit trimming to original frame count.
- Keep current script defaults for SDEdit execution:
  - guidance, num_steps, seed, offload flags, resize/height/width behavior
  - follow defaults from:
    - `run_scene_inv_full.sh` for inverse
    - `run_scene_full.sh` for forward
- Strength input supports a list of strengths in one run.

## 5) Workflow A: SDEdit Inverse

### 5.1 Input contract
- Root: `intrinsic_refinement/raw_intrinsic/`
- Required videos (configurable filenames; same length):
  - GT RGB
  - albedo
  - normal
  - normalized depth
  - roughness
  - metallic
- Videos are concatenated 3-camera videos.

### 5.2 Processing requirements
- Split concat videos into per-camera streams using `split.py` logic.
- Run inverse SDEdit for each requested strength.
- Sliding window:
  - if total frames < 57: repeat last frame for processing
  - if total frames > 57: use overlapping chunks
- Output trimming:
  - final saved outputs must match original frame count
  - remove any padded/repeated tail introduced for processing

### 5.3 Output contract
- Root: `intrinsic_refinement/refined_intrinsic_w{sdedit_strength}/`
- Save intrinsics only (no GT RGB output).
- Saved intrinsic names must be:
  - `albedo` (from basecolor)
  - `normal`
  - `normalized_depth` (from depth)
  - `roughness`
  - `metallic`
- Video outputs:
  - `intrinsic_refinement/refined_intrinsic_w{sdedit_strength}/videos/{intrinsic_name}.mp4`
  - these are concatenated 3-camera videos using the same camera order as input.
- Image outputs:
  - `intrinsic_refinement/refined_intrinsic_w{sdedit_strength}/images/{intrinsic_name}/{frame_id:03d}_{cam_id}.jpg`

## 6) Workflow B: SDEdit Forward

### 6.1 Input contract
- Root: `pbr_refinement/raw_render/`
- Required videos (configurable filenames; same length):
  - rendered PBR RGB
  - albedo
  - normal
  - normalized depth
  - roughness
  - metallic
- Required envmaps:
  - `envmap_0.hdr`
  - `envmap_1.hdr`
  - `envmap_2.hdr`
- Videos are concatenated 3-camera videos.

### 6.2 Processing requirements
- Split concat videos into per-camera streams.
- Run forward SDEdit per camera using matching `envmap_{camid}.hdr`.
- Sliding window and merge behavior should follow `run_scene_full.sh`.
- Strength input supports list of strengths in one run.

### 6.3 Output contract
- Root: `pbr_refinement/refined_render_w{sdedit_strength}/`
- Save video outputs only.
- Separate per-camera videos:
  - `pbr_refinement/refined_render_w{sdedit_strength}/videos/{camid}/...`
- Concatenated visualization videos:
  - `pbr_refinement/refined_render_w{sdedit_strength}/visualization/...`

## 7) Pain Points to Remove
- Manual copy/move/rename between projects.
- Manual split/recombine for each run.
- Non-deterministic naming across scripts.

## 8) Non-Goals in This Phase
- No DriveStudio-side exporter redesign yet.
- No model/training changes.
- No checkpoint management changes.

## 8.1 Implementation Permission for Next Session
- The next session is explicitly allowed to create new scripts for:
  - inverse workflow orchestration
  - forward workflow orchestration
  - staging/splitting/merging/trimming helpers
  - concat visualization generation
- The next session is not required to reuse old run scripts if new scripts are cleaner.
- Constraint:
  - preserve all processing details specified in this document (naming, camera order, chunking, trimming, outputs).
- Existing scripts can be treated as reference behavior, not mandatory entrypoints.

## 9) Acceptance Criteria
- For each requested strength, both workflows produce required outputs in exact target directories.
- Output frame counts equal original input frame counts.
- Camera order remains consistent end-to-end.
- No manual renaming/copying needed once pipeline is run.
 - Supports multiple strengths in one invocation.

## 9.1 Additional Context for Next Session
- Input naming must be configurable (do not hardcode `gt_rgb.mp4`, etc.).
- Camera split/concat mapping must remain deterministic and documented in logs.
- Forward workflow must map `envmap_{camid}.hdr` to the same `camid` video branch.
- Inverse workflow output pass naming must be normalized to:
  - `albedo`, `normal`, `normalized_depth`, `roughness`, `metallic`
- Keep resource-related defaults from legacy scripts:
  - seed/offload/resize-related defaults from `run_scene_inv_full.sh` and `run_scene_full.sh`.
- Pipeline should write enough logs to debug:
  - discovered inputs
  - inferred frame counts
  - chunk start indices
  - trim length applied

## 10) Concrete Examples (for verification)

### 10.1 Example A: Inverse workflow

#### Input example
```text
intrinsic_refinement/raw_intrinsic/
  gt_rgb.mp4
  albedo.mp4
  normal.mp4
  normalized_depth.mp4
  roughness.mp4
  metallic.mp4
```

#### Intermediate example (conceptual staging)
```text
intrinsic_refinement/tmp/split/
  gt_rgb_0.mp4
  gt_rgb_1.mp4
  gt_rgb_2.mp4
  albedo_0.mp4
  ...
intrinsic_refinement/tmp/chunks_w0.4_cam1/
  chunk_start000/
  chunk_start007/
  ...
```

#### Output example (strength = 0.4)
```text
intrinsic_refinement/refined_intrinsic_w0.4/
  videos/
    albedo.mp4
    normal.mp4
    normalized_depth.mp4
    roughness.mp4
    metallic.mp4
  images/
    albedo/
      000_0.jpg
      000_1.jpg
      000_2.jpg
      ...
    normal/
      000_0.jpg
      ...
    normalized_depth/
      000_0.jpg
      ...
    roughness/
      ...
    metallic/
      ...
```

### 10.2 Example B: Forward workflow

#### Input example
```text
pbr_refinement/raw_render/
  pbr_rgb.mp4
  albedo.mp4
  normal.mp4
  normalized_depth.mp4
  roughness.mp4
  metallic.mp4
  envmap_0.hdr
  envmap_1.hdr
  envmap_2.hdr
```

#### Intermediate example (conceptual staging)
```text
pbr_refinement/tmp/split/
  pbr_rgb_0.mp4
  pbr_rgb_1.mp4
  pbr_rgb_2.mp4
  albedo_0.mp4
  ...
pbr_refinement/tmp/chunks_w0.7_cam2/
  chunk_start000/
  chunk_start007/
  ...
```

#### Output example (strength = 0.7)
```text
pbr_refinement/refined_render_w0.7/
  videos/
    0/
      pbr_rgb.mp4
    1/
      pbr_rgb.mp4
    2/
      pbr_rgb.mp4
  visualization/
    pbr_rgb_concat.mp4
```
