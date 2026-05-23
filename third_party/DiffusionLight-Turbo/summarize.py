import pyexr
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

scenes = ("003", "019", "036")
view_times = ("000_0", "023_1", "045_2", "077_0", "100_1", "122_2")
resize_methods=("crop", "full")
seeds=(0, 37, 71, 125, 140)
output_dir = "./output"

single_scene = scenes[0]
single_view_time = view_times[1]
single_resize_method = resize_methods[0]
single_seed = seeds[0]

def get_minmax(hdr_path):
    hdr_data = pyexr.open(hdr_path)
    hdr_img = hdr_data.get()
    return hdr_img.min(), hdr_img.max()

def collect_scenes():
    hdr_paths = {}
    single_hdr_paths = {}
    for scene in scenes:
        hdr_paths[scene] = []
        for view_time in view_times:
            for resize_method in resize_methods:
                for seed in seeds:
                    hdr_path = os.path.join(output_dir, f"hdr_seed{seed}", f"{scene}_{view_time}_{resize_method}.exr")
                    if os.path.exists(hdr_path):
                        min_val, max_val = get_minmax(hdr_path)
                        hdr_paths[scene].append((hdr_path, min_val, max_val))
                    else:
                        raise FileNotFoundError(f"File not found: {hdr_path}")
        
        # collect for single setting
        single_hdr_path = os.path.join(output_dir, f"hdr_seed{single_seed}", f"{scene}_{single_view_time}_{single_resize_method}.exr")
        min_val, max_val = get_minmax(single_hdr_path)
        single_hdr_paths[scene] = (single_hdr_path, min_val, max_val)
    return hdr_paths, single_hdr_paths

def collect_view_times():
    hdr_paths = {}
    single_hdr_paths = {}
    for view_time in view_times:
        hdr_paths[view_time] = []
        for scene in scenes:
            for resize_method in resize_methods:
                for seed in seeds:
                    hdr_path = os.path.join(output_dir, f"hdr_seed{seed}", f"{scene}_{view_time}_{resize_method}.exr")
                    if os.path.exists(hdr_path):
                        min_val, max_val = get_minmax(hdr_path)
                        hdr_paths[view_time].append((hdr_path, min_val, max_val))
                    else:
                        raise FileNotFoundError(f"File not found: {hdr_path}")
        
        # collect for single setting
        single_hdr_path = os.path.join(output_dir, f"hdr_seed{single_seed}", f"{single_scene}_{view_time}_{single_resize_method}.exr")
        min_val, max_val = get_minmax(single_hdr_path)
        single_hdr_paths[view_time] = (single_hdr_path, min_val, max_val)
        
    return hdr_paths, single_hdr_paths

def collect_resize_methods():
    hdr_paths = {}
    single_hdr_paths = {}
    for resize_method in resize_methods:
        hdr_paths[resize_method] = []
        for scene in scenes:
            for view_time in view_times:
                for seed in seeds:
                    hdr_path = os.path.join(output_dir, f"hdr_seed{seed}", f"{scene}_{view_time}_{resize_method}.exr")
                    if os.path.exists(hdr_path):
                        min_val, max_val = get_minmax(hdr_path)
                        hdr_paths[resize_method].append((hdr_path, min_val, max_val))
                    else:
                        raise FileNotFoundError(f"File not found: {hdr_path}")
        
        # collect for single setting
        single_hdr_path = os.path.join(output_dir, f"hdr_seed{single_seed}", f"{single_scene}_{single_view_time}_{resize_method}.exr")
        min_val, max_val = get_minmax(single_hdr_path)
        single_hdr_paths[resize_method] = (single_hdr_path, min_val, max_val)
    return hdr_paths, single_hdr_paths

def collect_seeds():
    hdr_paths = {}
    single_hdr_paths = {}
    for seed in seeds:
        hdr_paths[seed] = []
        for scene in scenes:
            for view_time in view_times:
                for resize_method in resize_methods:
                    hdr_path = os.path.join(output_dir, f"hdr_seed{seed}", f"{scene}_{view_time}_{resize_method}.exr")
                    if os.path.exists(hdr_path):
                        min_val, max_val = get_minmax(hdr_path)
                        hdr_paths[seed].append((hdr_path, min_val, max_val))
                    else:
                        raise FileNotFoundError(f"File not found: {hdr_path}")
        # collect for single setting
        single_hdr_path = os.path.join(output_dir, f"hdr_seed{seed}", f"{single_scene}_{single_view_time}_{single_resize_method}.exr")
        min_val, max_val = get_minmax(single_hdr_path)
        single_hdr_paths[seed] = (single_hdr_path, min_val, max_val)
    return hdr_paths, single_hdr_paths

collect_map = {
    "scene": collect_scenes,
    "view_time": collect_view_times,
    "resize_method": collect_resize_methods,
    "seed": collect_seeds
}

if __name__ == "__main__":
    # path: {output_dir}/hdr_seed{seed}/{scene}_{view_time}_{resize_method}.exr
    
    change_param = "scene"
    # change_param = "view_time"
    # change_param = "resize_method"
    # change_param = "seed"
    
    data, single_data = collect_map[change_param]()
    
    stat_output_dir = os.path.join(output_dir, f"stats_{change_param}")
    if not os.path.exists(stat_output_dir):
        os.makedirs(stat_output_dir)
    
    # for each key, output a histogram of min and max values
    for key, values in tqdm(data.items()):
        min_values = [v[1] for v in values]
        max_values = [v[2] for v in values]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(min_values, bins=50, color='blue', alpha=0.7, label='Min Values')
        plt.title(f'Min Values Histogram for {key}')
        plt.xlabel('Min Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.hist(max_values, bins=50, color='red', alpha=0.7, label='Max Values')
        plt.title(f'Max Values Histogram for {key}')
        plt.xlabel('Max Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(stat_output_dir, f"{key}_histogram.png"))
        plt.close()
        
        # also write each path and its min/max values to txt
        with open(os.path.join(stat_output_dir, f"{key}_minmax.txt"), 'w') as f:
            for path, min_val, max_val in values:
                f.write(f"{path}: min={min_val}, max={max_val}\n")
        # print(f"Statistics for {key} saved to {stat_output_dir}")
    
    # Create subplots for max value histograms
    key_num = len(data)
    # Calculate subplot grid dimensions
    cols = min(3, key_num)  # Max 3 columns
    rows = (key_num + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle different cases for axes indexing
    if key_num == 1:
        # Single subplot case
        axes = [axes]
    elif rows == 1 and cols > 1:
        # Single row, multiple columns
        axes = axes.flatten()
    elif rows > 1 and cols == 1:
        # Multiple rows, single column
        axes = axes.flatten()
    elif rows > 1 and cols > 1:
        # Multiple rows and columns - keep 2D structure
        pass
    
    for i, (key, values) in enumerate(data.items()):
        if rows == 1 or cols == 1 or key_num == 1:
            # Use flattened indexing
            ax = axes[i]
        else:
            # Use 2D indexing
            row = i // cols
            col = i % cols
            ax = axes[row, col]
        
        max_values = [v[2] for v in values]
        ax.hist(max_values, bins=30, alpha=0.7, color=plt.cm.viridis(i / key_num))
        ax.set_title(f'{change_param}: {key}')
        ax.set_xlabel('Max Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots if any
    total_subplots = rows * cols
    if total_subplots > key_num:
        for i in range(key_num, total_subplots):
            if rows == 1 or cols == 1 or key_num == 1:
                axes[i].set_visible(False)
            else:
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
    
    plt.suptitle(f'Max Values Histograms for {change_param}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(stat_output_dir, f"subplot_max_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.clf()
    # write single setting to bar chart and copy the images
    single_min_values = [single_data[key][1] for key in single_data]
    single_max_values = [single_data[key][2] for key in single_data]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(single_data.keys(), single_min_values, color='blue', alpha=0.7, label='Min Values')
    # Add value labels on top of each bar
    for bar, value in zip(bars1, single_min_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(single_min_values)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    plt.title(f'Single Setting Min Values')
    plt.xlabel('Setting')
    plt.ylabel('Min Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(single_data.keys(), single_max_values, color='red', alpha=0.7, label='Max Values')
    # Add value labels on top of each bar
    for bar, value in zip(bars2, single_max_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(single_max_values)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    plt.title(f'Single Setting Max Values')
    plt.xlabel('Setting')
    plt.ylabel('Max Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(stat_output_dir, "single_setting_bar.png"))
    plt.close()
    
    # image path: path: {output_dir}/envmap/{scene}_{view_time}_{resize_method}_ev-00_seed{seed}.png
    for key, value in single_data.items():
        if change_param == "scene":
            img_path = os.path.join(output_dir, f"envmap/{key}_{single_view_time}_{single_resize_method}_ev-00_seed{single_seed}.png")
        elif change_param == "view_time":
            img_path = os.path.join(output_dir, f"envmap/{single_scene}_{key}_{single_resize_method}_ev-00_seed{single_seed}.png")
        elif change_param == "resize_method":
            img_path = os.path.join(output_dir, f"envmap/{single_scene}_{single_view_time}_{key}_ev-00_seed{single_seed}.png")
        elif change_param == "seed":
            img_path = os.path.join(output_dir, f"envmap/{single_scene}_{single_view_time}_{single_resize_method}_ev-00_seed{key}.png")
        if os.path.exists(img_path):
            plt.figure(figsize=(6, 6))
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{key} - Min: {value[1]}, Max: {value[2]}")
            plt.savefig(os.path.join(stat_output_dir, f"{key}_image.png"))
            plt.close()
        else:
            print(f"Image not found for {key}: {img_path}")
    
    print("All statistics collected and saved.")