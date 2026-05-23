

scenes=(21 22 25 34 35 49 53 80 84 86 89 94 96 102 111 222 323 382 402 427 438 546 581 592 620 640 700 754 795 796)
timesteps=()
interval=10
for scene in "${scenes[@]}"; do
    sceneidx3d=$(printf "%03d" $scene)
    # count number of images in /project2/yi-ray/BRDFusion/data/waymo/processed/training/022/images
    num_images=$(ls -1 "/project2/yi-ray/BRDFusion/data/waymo/processed/training/${sceneidx3d}/images/" | wc -l)
    # timesteps = num_images / 5 (since each timestep has 5 cameras)
    num_timesteps=$((num_images / 5))
    timesteps+=($num_timesteps)
done

for idx in "${!scenes[@]}"; do
    scene=${scenes[$idx]}
    num_timesteps=${timesteps[$idx]}
    bash run_dataset.sh $scene $num_timesteps $interval
done
