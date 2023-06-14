#methods=["priorart","remapping","padding"]
method=priorart
feat=p5
cuda_env=0
#################################################################################
exp_folder=exp_ft_`date +"%m-%d-%H-%M-%S"`_${feat}_${method}
mkdir $exp_folder

extrc_proc="bash 4tvd_extract_p_segmentation.sh"
image_dirs=("/dataset/TVD/TVD-01" "/dataset/TVD/TVD-02" "/dataset/TVD/TVD-03")
for image_dir in ${image_dirs[*]}
do
    temp_proc="$extrc_proc $image_dir ${cuda_env} $exp_folder"
    #also change method used in 4tvd_extract_p_layers.py (line193)
    logfile=$exp_folder/`basename $image_dir`_${feat}_${method}_extract.log
    echo $logfile
    echo $temp_proc
    nohup $temp_proc > $logfile 2>&1 &
done

# bash 4extract_p_segmentation.sh /dataset/TVD/TVD-01 0 # cuda_env

# bash 4extract_p_segmentation.sh /dataset/TVD/TVD-02 1

# bash 4extract_p_segmentation.sh /dataset/TVD/TVD-03 1