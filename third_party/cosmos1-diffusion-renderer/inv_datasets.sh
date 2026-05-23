# dataset_ids=(245 527 496 157)
# num_timesteps=(199 199 198 197)

# dataset_ids=(172 703)
# num_timesteps=(198 199)

# dataset_ids=(3 19 245 114 172 703)
dataset_ids=(25 34 35 49 53 80 84 86 89 94 96 102 111 222 323 382 402 427 438 546 581 592 620 640 700 754 795 796)
timesteps=()
for scene in "${dataset_ids[@]}"; do
    sceneidx3d=$(printf "%03d" $scene)
    # count number of images in /project2/yi-ray/BRDFusion/data/waymo/processed/training/022/images
    num_images=$(ls -1 "/project2/yi-ray/BRDFusion/data/waymo/processed/training/${sceneidx3d}/images/" | wc -l)
    # timesteps = num_images / 5 (since each timestep has 5 cameras)
    num_timesteps=$((num_images / 5))
    echo "Scene $scene has $num_timesteps timesteps"
    timesteps+=($num_timesteps)
done

for idx in "${!dataset_ids[@]}"; do
    dataset_id=${dataset_ids[$idx]}
    num_timestep=${timesteps[$idx]}
    bash inv_dataset.sh $dataset_id $num_timestep
done

# for idx in "${!dataset_ids[@]}"; do
#     dataset_id=${dataset_ids[$idx]}
#     num_timestep=51
#     bash inv_dataset_multiseed.sh $dataset_id 5 0 1000
# done
