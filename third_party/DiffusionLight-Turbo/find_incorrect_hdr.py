import pyexr
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import itertools

scenes = ("003", "019", "036")
view_times = ("000_0", "023_1", "045_2", "077_0", "100_1", "122_2")
resize_methods=("crop", "full")
seeds=(0, 37, 71, 125, 140)


def get_minmax(hdr_path):
    hdr_data = pyexr.open(hdr_path)
    hdr_img = hdr_data.get()
    return hdr_img.min(), hdr_img.max()

def test(scene, view_time, resize_method, seed):
    assert scene in scenes, f"Invalid scene: {scene}"
    assert view_time in view_times, f"Invalid view_time: {view_time}"
    assert resize_method in resize_methods, f"Invalid resize_method: {resize_method}"
    assert seed in seeds, f"Invalid seed: {seed}"
    
    path = os.path.join(output_dir, f"hdr_seed{seed}", f"{scene}_{view_time}_{resize_method}.exr")
    min_val, max_val = get_minmax(path)
    return max_val >= 20

def load_failed_cases(txt_path):
    failed_set = set()
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('_')
            if len(parts) < 6:
                continue  # skip invalid lines
            scene = parts[0]
            view_time = f"{parts[1]}_{parts[2]}"  # e.g., "000_0"
            resize_method = parts[3]              # "crop" or "full"
            seed = int(parts[-1].replace("seed", ""))  # e.g., "seed0" → 0
            key = (scene, view_time, resize_method, seed)
            failed_set.add(key)
    return failed_set

# Load the failures into a global set (call this once at the top of your script)
FAILED_CASES_TXT = "./invalid_cases.txt"  # <-- update this path
failed_cases = load_failed_cases(FAILED_CASES_TXT)

# Modified test function
def test_dis(scene, view_time, resize_method, seed):
    key = (scene, view_time, resize_method, seed)
    return key not in failed_cases  # return True if success


if __name__ == "__main__":
    # path: {output_dir}/hdr_seed{seed}/{scene}_{view_time}_{resize_method}.exr
    output_dir = "./output"
    
    # ref = "hdr" 
    ref = "dis"
    
    if ref == "hdr":
        test_func = test
        txt_output_dir = "./txt_output"
    elif ref == "dis":
        test_func = test_dis
        txt_output_dir = "./txt_output_dis"
    else:
        raise ValueError("Invalid reference type. Use 'hdr' or 'dis'.")
    
    data = []

    for scene, view_time, resize_method, seed in itertools.product(scenes, view_times, resize_methods, seeds):
        success = test_func(scene, view_time, resize_method, seed)
        data.append({
            'scene': scene,
            'view_time': view_time,
            'resize_method': resize_method,
            'seed': seed,
            'error': int(not success)
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # --- 2. Aggregate failures across all factor combinations ---
    factor_columns = ['scene', 'view_time', 'resize_method', 'seed']
    for r in range(1, 5):  # lengths1, 2, 3, 4
        for factor_subset in itertools.combinations(factor_columns, r):
            subset_name = "_".join(factor_subset)
            
            # Group by the combination of factors
            group = df.groupby(list(factor_subset))['error'].sum().reset_index()
            group = group[group['error'] > 0]  # Only include combinations with failures
            group_sorted = group.sort_values('error', ascending=False).head(10)

            if not group_sorted.empty:
                # Print to console
                print(f"\nTop failures for {subset_name}:")
                print(group_sorted)

                # Save to TXT file
                filename = f"{subset_name}.txt"
                filepath = os.path.join(txt_output_dir, filename)
                os.makedirs(txt_output_dir, exist_ok=True)
                group_sorted.to_csv(filepath, index=False, sep="\t")
    