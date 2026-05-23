# Codebase Summary: `cosmos1-diffusion-renderer`

## What this repository is
This repository is a working copy of NVIDIA's Cosmos DiffusionRenderer stack, focused on:
- Inverse rendering: RGB input -> G-buffer passes (`basecolor`, `normal`, `depth`, `roughness`, `metallic`)
- Forward rendering / relighting: G-buffer + environment lighting -> relit RGB
- SDEdit-style refinement workflows for both inverse and forward renderer outputs

It combines:
- The reusable `cosmos_predict1` framework code
- Project-specific orchestration scripts for experiments on local datasets/scenes
- Large local artifacts (checkpoints, generated videos, experiment outputs)

## High-level architecture
- Core package: `cosmos_predict1/`
- Main renderer inference entrypoints:
  - `cosmos_predict1/diffusion/inference/inference_inverse_renderer.py`
  - `cosmos_predict1/diffusion/inference/inference_forward_renderer.py`
- Core inference pipeline abstraction:
  - `cosmos_predict1/diffusion/inference/diffusion_renderer_pipeline.py`
- Model/config layers:
  - `cosmos_predict1/diffusion/model/`
  - `cosmos_predict1/diffusion/config/`
  - `cosmos_predict1/diffusion/networks/`
- Shared utilities:
  - `cosmos_predict1/utils/`

## Repo layout (practical view)
- `cosmos_predict1/`: Primary Python package (diffusion, tokenizer, autoregressive, utils)
- `scripts/`: Environment tests, checkpoint downloaders, data prep helpers
- `checkpoints/`: Pretrained model weights (large; ~56G in this workspace)
- `asset/`: Demo inputs/results and media assets (~685M)
- `drivestudio_exp/`: Local experiment outputs and scene-specific assets (~19G)
- `output/`: Generated outputs from local runs (~21M)
- Top-level `*.sh` and `*.py`: Workflow glue for dataset processing, SDEdit sweeps, merging/resizing/evaluation

## Main workflows in this workspace
1. Run inverse renderer on frames/videos to estimate G-buffers.
2. Run forward renderer on estimated G-buffers with envmaps for relighting.
3. Apply SDEdit-style refinement:
   - `sdedit_inverse_renderer.py` (refine G-buffers)
   - `sdedit_forward_renderer.py` (refine relit RGB output)
4. Post-process and aggregate with helper scripts:
   - Chunking/overlap merging
   - Image averaging/resizing
   - Multi-camera stitching and video merges
   - PSNR evaluation scripts

## Important project-specific scripts
- `sdedit_forward_renderer.py`: Forward renderer refinement conditioned on RGB + G-buffer + envmap.
- `sdedit_inverse_renderer.py`: Inverse renderer refinement conditioned on RGB and optional G-buffer priors.
- `run_scene.sh`, `run_scene_full.sh`, `run_scene_rotate.sh`, `run_scene_inv*.sh`: Batch scene experiments and parameter sweeps.
- `inv_dataset*.sh`: Sliding-window inverse rendering over datasets with overlap handling and aggregation.
- `merge_overlap_*.py/.sh`, `merge.py`, `concat.py`, `split.py`: Temporal/chunk/camera merging utilities.
- `compute_psnr_vid.py`, `compute_psnr_vidlist.py`: Video quality evaluation helpers.

## Dependencies and runtime profile
- Python 3.10 and CUDA-enabled PyTorch stack (`torch==2.6.0`, `torchvision==0.21.0`).
- Heavy GPU requirements for full video inference (README recommends >=16GB, often more for long clips).
- Additional NVIDIA ecosystem components (transformer-engine, optional nvdiffrast, etc.).

## Notes for contributors
- The repo mixes framework code with substantial generated media and local experiment state.
- Most top-level scripts are experiment automation rather than reusable library interfaces.
- `test.py` is a local utility/demo script for overlap chunking logic, not a formal test suite.
- If extending functionality, prefer adding reusable logic inside `cosmos_predict1/` and keep top-level scripts as orchestration wrappers.
