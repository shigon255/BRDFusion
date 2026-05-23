set -e
dataset_root=/project2/yi-ray/BRDFusion/data/waymo/processed/training

# scenes=("003" "019" "036")
scenes=("016" "245" "527")
view_times=("000_0" "023_1" "045_2" "077_0" "100_1" "122_2")
ext=".jpg"

output_dir="test_images_extra3"
mkdir -p $output_dir

full_tmp="test_image_full"
mkdir -p $full_tmp
crop_tmp="test_image_crop"
mkdir -p $crop_tmp

for scene in "${scenes[@]}"; do
    for view_time in "${view_times[@]}"; do
        input_path="${dataset_root}/${scene}/images/${view_time}${ext}"
        name=${scene}_${view_time}${ext}
        cp $input_path $full_tmp/${name}
        cp $input_path $crop_tmp/${name}
    done
done

python resize_full.py \
    $full_tmp \
    ${full_tmp}_output \

python resize_crop.py \
    $crop_tmp \
    ${crop_tmp}_output

for scene in "${scenes[@]}"; do
    for view_time in "${view_times[@]}"; do
        name=${scene}_${view_time}
        cp ${full_tmp}_output/${name}${ext} $output_dir/${name}_full${ext}
        cp ${crop_tmp}_output/${name}${ext} $output_dir/${name}_crop${ext}
    done
done

rm -r $full_tmp
rm -r $crop_tmp
rm -r ${full_tmp}_output
rm -r ${crop_tmp}_output
