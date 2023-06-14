# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

exp_folder=$3 #exp_folder=exp_ft_`date +"%m-%d-%H-%M-%S"`
echo $exp_folder
# number of workers, this number should match the number of GPUs available in the system
n_workers=1

# machine task
task=segmentation

# dataset directory (YUV information(W and H) will be stored in the '../detectron2' directory)
# image_dir="/dataset/TVD/TVD-02" #the image dataset directory

image_dir=$1 # image_dir="/dataset/TVD/TVD-02"
cuda_dev=$2

# anno_dir=$(realpath ../dataset/annotations_5k) #the directory that includes annotations for validation set

input_file=`basename $image_dir` # used

yuv_dir="${exp_folder}/${input_file}_uncompressed_p_yuv_${task}"
echo $yuv_dir
mkdir -p $yuv_dir #the directory which the extracted stem YUVs are stored into.

# tasks: 'id','image_dir','input_lst','cuda_device'
task_list=("P_extraction_from_TVD","${image_dir}","${yuv_dir}","${input_file}",$cuda_dev,$exp_folder)

# uncompressed data are stored in a different directory with different extension
# input_file has all images in extension name as png, the original images are in jpg format
# convert the input list

function process_task() {
  task=$1
  task_info=$2
  IFS=',' read task_id image_dir yuv_dir input_file cuda_dev exp_folder<<< "${task_info}"

  echo
  echo "Processing ${task_id} ..."

  echo $input_file

  pushd $(pwd)

  CUDA_VISIBLE_DEVICES=${cuda_dev} python 4tvd_extract_p_layers.py \
    --image_dir $image_dir \
    --task $task \
    --yuv_dir $yuv_dir \
    --input_file $input_file \
    --exp_folder $exp_folder

  echo "Task ${task_id} done. "
  echo 

}
export -f process_task


for task_info in "${task_list[@]}"; do
   process_task ${task} ${task_info}
done


echo "All done!"


exit 0

