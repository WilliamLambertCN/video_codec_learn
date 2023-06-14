exp_folder="exp_ft_02-02-13-15-36_p5_priorart"
# input_file="TVD-01"

# exp_folder="exp_2023-01-12-12-39-55"
# input_file="TVD-02"

# exp_folder="exp_2023-01-12-12-40-03"
input_file="TVD-01"
feat="p5_priorart"

task=segmentation

cd ${exp_folder} && pwd

yuv_dir="${input_file}_uncompressed_p_yuv_${task}" # yuv frames to concat

# folder to store the concated yuv files
res_dir=`realpath ${input_file}_res`

#################### safemode_flag: only 0 for actual running##########
if (($#==0)) # nop args input, safemode enabled by default
then 
    safemode_flag=1
else
    safemode_flag=$1
fi

# how many segments available in $yuv_dir folder
declare -i segment_num=`ls $yuv_dir -l | grep "^-" |wc -l`
echo $segment_num

# cd ${yuv_dir} && pwd
cat_proc="cat"
cur_idx=1
while (($cur_idx <= $segment_num))
do
    temp="`printf "%04d" $cur_idx`".yuv""
    cat_proc="$cat_proc ${yuv_dir}/$temp"
    # echo $cur_idx
    cur_idx=`expr $cur_idx + 1`
    # break
done

# cat_proc="nohup $cat_proc > ${input_file}.yuv 2>&1 &"
cat_proc="nohup cat ${yuv_dir}/*.yuv > $res_dir/${input_file}_${feat}_feature.yuv 2>&1 &"
echo $cat_proc

if (($safemode_flag==0))
then
    echo "safemode_flag=$safemode_flag, safemode disabled, make sure there is no error."
    mkdir $res_dir
    nohup cat ${yuv_dir}/*.yuv > $res_dir/${input_file}_${feat}_feature.yuv 2>&1 &
else
    echo "safemode_flag=$safemode_flag, safemode enabled, no actual execution to check path."
fi

# cat_proc="$cat_proc > $res_dir/${input_file}.yuv"
# echo $cat_proc
# $cat_proc