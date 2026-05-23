#!/usr/bin/env python3

import argparse
from pathlib import Path

import imageio
import torch

from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.utils import log, misc
from sdedit_forward_renderer import load_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode/decode a frame sequence with the DiffusionRenderer tokenizer.")
    parser.add_argument("--input_frames", type=str, required=True, help="Input directory containing image frames.")
    parser.add_argument("--output_frames", type=str, required=True, help="Output directory for reconstructed image frames.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Forward_Cosmos_7B",
        help="Checkpoint name used to load the tokenizer.",
    )
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--max_frames", type=int, default=57)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--offload_diffusion_transformer", action="store_true")
    parser.add_argument("--offload_tokenizer", action="store_true")
    return parser.parse_args()


def save_frames(output_dir: str, frames: torch.Tensor) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames_np = (frames[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
    for index, frame in enumerate(frames_np):
        imageio.imwrite(output_path / f"frame_{index:04d}.png", frame)


def main() -> None:
    args = parse_args()
    misc.set_random_seed(args.seed)

    input_video = load_video_frames(
        args.input_frames,
        max_frames=args.max_frames,
        height=args.height,
        width=args.width,
        start_frame=args.start_frame,
    )

    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        guidance=0.0,
        num_steps=1,
        height=args.height,
        width=args.width,
        fps=24,
        num_video_frames=args.max_frames,
        seed=args.seed,
    )

    if args.offload_tokenizer:
        pipeline._load_tokenizer()

    video_cuda = input_video.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        latent = pipeline.model.encode(video_cuda)
        reconstructed = pipeline.model.decode(latent)

    reconstructed = (1.0 + reconstructed).clamp(0, 2) / 2
    save_frames(args.output_frames, reconstructed)
    log.info(f"Saved reconstructed frames to {args.output_frames}")


if __name__ == "__main__":
    main()
