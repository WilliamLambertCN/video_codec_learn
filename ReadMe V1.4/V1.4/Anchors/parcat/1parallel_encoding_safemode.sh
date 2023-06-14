pwd
# folders(./log, ./bin, ./yuv) to store the results
exp_folder=exp_vtm_`date +"%m-%d-%H-%M-%S"`_f${3}_qp${10}_itr${11}_GoP${12}
#################### safemode_flag: only 0 for actual running##########
if (($1==1)) # nop args input, safemode enabled by default
then 
    safemode_flag=1
else
    safemode_flag=$1
    mkdir $exp_folder
    mkdir $exp_folder/log
    mkdir $exp_folder/bin
    mkdir $exp_folder/yuv
fi
#################### File related Settings ############################
# select source file
src_file=$2 # -i

src_filename=`basename $src_file .yuv`
tgt_name_prefix="${src_filename}_" ## ouput file like: TVD_02_0001.txt .bin .yuv
declare -i total_frames=$3
echo $total_frames
# make sure parallel_enconding process num within maximum+1,
# FramesToBeEncoded must be ${IntraPeriod} + 1, when using random access cfg. Otherwise you cannot concate them
cfg_type=${13}
if (($cfg_type==1)) # All Intra, one frame to encode for one time
then
    maximum_encoding=`expr ${total_frames} + 10` # to avoid infinite loop
    echo "All Intra cfg is used, maximum_encoding proccess is ${maximum_encoding}"
else
    maximum_encoding=100
    echo "Random Access cfg is used, maximum_encoding proccess is ${maximum_encoding}"
fi

maximum_concurrent=10 #  to control concuurrent proc
echo "maximum_concurrent proccess on the same time is ${maximum_concurrent}"

sleep 5s

# FramesToBeEncoded=`expr $total_frames / $cpu_cores`
wdt=$4
hgt=$5
InputBitDepth=$6
InputChromaFormat=$7
FrameRate=$8
input_fmt="-wdt $wdt -hgt $hgt --InputBitDepth=$InputBitDepth --InputChromaFormat=$InputChromaFormat"
#################### For Experimental use #################################
# other parameters for experimental use
encoder=$9  # encoder="./tools/EncoderAppStatic"
qp_rate=${10}

IntraPeriod=${11}
GOPSize=${12}
FramesToBeEncoded=`expr ${IntraPeriod} + 1`

if (($cfg_type==0))
then
    encoder_cfg="-c cfg/encoder_randomaccess_vtm.cfg -c cfg/lossless/lossless.cfg --IntraPeriod=${IntraPeriod} --GOPSize=$GOPSize --FrameRate=$FrameRate" # -c ./cfg/xx.cfg
elif (($cfg_type==1))
then
    encoder_cfg="-c cfg/encoder_intra_vtm.cfg -c cfg/lossless/lossless.cfg --FrameRate=1" # -c ./cfg/xx.cf
elif (($cfg_type==2))
then
    encoder_cfg="-c cfg/encoder_randomaccess_vtm.cfg --IntraPeriod=${IntraPeriod} --GOPSize=$GOPSize --FrameRate=$FrameRate" # -c ./cfg/xx.cf
fi
# qp_rate=0
custom_para="-q ${qp_rate} --Level=6.2 --ConformanceWindowMode=1"

###################Fixed No changes here###############################
# initialize slice_num(for outputfile name) and frame_range
declare -i FrameSkip=0 # auto updating every iteration
declare -i num=1
#################################################################
# !!Attention: every process encode video with one frame's overlapping here, or VTM parcatStatic may lose frames

end_flag=0
while (($end_flag==0))
do
    # generate tgt_name
    declare -i end_frame=${FrameSkip}+${FramesToBeEncoded}-1 # not used in process, just to demonstrate
    # printf "($end_frame>${total_frames}-1)\n"
    if (($end_frame>=${total_frames}-1))
    then
        end_frame=${total_frames}-1
        end_flag=1
    fi
    echo "${num}>>frame${FrameSkip}-${end_frame}"
    post_fix=`printf "%04d" $num`
    tgt_name="$tgt_name_prefix${post_fix}"

    #encode range
    encode_rng="--FrameSkip=${FrameSkip} --FramesToBeEncoded=${FramesToBeEncoded}"
    tgtbin_file="$exp_folder/bin/${tgt_name}.bin" #-b
    # tgtyuv_file="$exp_folder/yuv/${tgt_name}.yuv" # -o
    log_file="$exp_folder/log/${tgt_name}.txt" # > $log_file
    echo "logs in $log_file"

    encoding_prc="${encoder} ${encoder_cfg} ${custom_para} ${input_fmt} ${encode_rng} -i $src_file -b $tgtbin_file" # -o $tgtyuv_file"
    echo $encoding_prc #make sure there is no infinite loop using echo
    if (($safemode_flag==0))
    then
        echo $encoding_prc >> "$exp_folder/log/cmd.txt"
        echo "safemode_flag=$safemode_flag, safemode disabled, make sure there is no infinite loop."
        nohup $encoding_prc > $log_file 2>&1 & 
        sleep 0.6s
        top -b -d 10 -i -c -n 1 | grep "Encoder*" | wc -l
    else
        echo "safemode_flag=$safemode_flag, safemode enabled, no actual execution to check whether there is infinite loop."
    fi
    # update frame range and src_filename
    if (($cfg_type==1)) # All Intra, one frame to encode for one time
    then
        FrameSkip=$end_frame+1
    else
        FrameSkip=$end_frame
    fi
    ############################
    if (($num>$maximum_encoding))
    then
        echo "$num encoding-proc running, make sure no infinite loop, or all cpus will be used up"
        break
    fi
    ####################################
    while ((`top -b -d 10 -i -c -n 1 | grep "Encoder*" | wc -l`>`expr $maximum_concurrent - 1`))
    do
        echo `top -b -d 10 -i -c -n 1 | grep "Encoder*" | wc -l` "running Encoder, sleeping for 30s" 
        sleep 30s
    done
    ################3##########3###
    num=$num+1
    echo
done