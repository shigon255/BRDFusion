set -e 
device=1
seeds=(0 37 71 125 140)
# input_dir=./example
# output_dir=./example_output
input_dir=./test_images_extra3
output_dir=./output_extra3

seed_str=$(IFS=','; echo "${seeds[*]}")
echo "Using seeds: $seed_str"
CUDA_VISIBLE_DEVICES=$device python inpaint.py \
    --dataset $input_dir --output_dir $output_dir \
    --seed "$seed_str"
CUDA_VISIBLE_DEVICES=$device python ball2envmap.py \
    --ball_dir $output_dir/square --envmap_dir $output_dir/envmap

for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python exposure2hdr.py \
        --input_dir $output_dir/envmap --output_dir $output_dir/hdr_seed${seed} \
        --endwith "_seed${seed}.png" --preview_output \

done


