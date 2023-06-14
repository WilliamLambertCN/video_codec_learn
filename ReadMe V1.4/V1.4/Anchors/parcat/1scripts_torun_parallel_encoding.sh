#################### safemode_flag: only 0 for actual running##########
if (($#==0)) # nop args input, safemode enabled by default
then 
    safemode_flag=1
else
    safemode_flag=$1
fi

#src_file total_frames wdt hgt InputBitDepth InputChromaFormat FrameRate
src_file=/Anchors/parcat/exp_ft_01-31-02-56-59_p5_remapping/TVD-01_res/TVD-01_p5feature.yuv
total_frames=65
file_settings="$src_file $total_frames 5376 4080 10 400 1"

# encoder_path qp_rate IntraPeriod GoP cfg_type
encoder="/Anchors/parcat/tools/EncoderAppStatic" #cfg to modify in 1parallel_encoding_safemode.sh

qp_rate=0

# cfg_type=1 # RA_lossless, AI_lossless, RA
# IntraPeriod=0 # [ATTENTION!] partition is IntraPeriod+1. 
# GOPSize=0 # In All Intra, it should be ZERO !!! and GoP will not be used.

cfg_type=0 # RA_lossless, AI_lossless, RA
IntraPeriod=32 
GOPSize=32

experiment_settings="$encoder $qp_rate $IntraPeriod $GOPSize $cfg_type"

##########################################################################
if (($safemode_flag==0))
then
    para_encoder_proc_UN_safemode="bash 1parallel_encoding_safemode.sh 0 $file_settings $experiment_settings"
    echo ATENTION: unsafemode
    echo $para_encoder_proc_UN_safemode
    $para_encoder_proc_UN_safemode
else
    # safemode
    para_encoder_proc_safemode="bash 1parallel_encoding_safemode.sh 1 $file_settings $experiment_settings"
    echo $para_encoder_proc_safemode
    $para_encoder_proc_safemode
fi
# $para_encoder_proc_UN_safemode
