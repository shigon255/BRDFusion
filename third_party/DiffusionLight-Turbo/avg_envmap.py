from argparse import ArgumentParser
import os
import numpy as np
import pyexr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .exr files')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the averaged environment map .exr file')
    parser.add_argument("--seed", default="auto", type=str, help="Seed: right now we use single seed instead to reduce the time, (Auto will use hash file name to generate seed)")
    return parser.parse_args()

def read_exr(path, convert_float=True):
    exr = pyexr.read(path)
    if convert_float and exr.dtype == np.uint8:
        exr = exr.astype(np.float32) / 255.0
    return exr

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    seeds = [int(s) for s in args.seed.split(',')] if args.seed != "auto" else None
    
    # get file names
    file_names = [f for f in os.listdir(os.path.join(args.input_dir, f"hdr_seed{seeds[0]}")) if f.endswith('.exr')]
    
    # get envmap size
    test_envmap = read_exr(os.path.join(args.input_dir, f"hdr_seed{seeds[0]}", file_names[0]))
    H, W, C = test_envmap.shape
    
    avg_envmaps = {}
    for file_name in file_names:
        avg_envmaps[file_name] = np.zeros((H, W, C), dtype=np.float32)
        for seed in seeds:
            envmap = read_exr(os.path.join(args.input_dir, f"hdr_seed{seed}", file_name))
            avg_envmaps[file_name] += envmap / len(seeds)
        pyexr.write(os.path.join(args.output_path, "average_" + file_name), avg_envmaps[file_name])
        
    
    