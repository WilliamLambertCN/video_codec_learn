pwd

exp_folder="/Anchors/parcat/exp_vtm_01-31-08-17-34_f65_qp0_itr0_GoP0"

# folder to store the decoding log and output files(.log .bin .yuv)
mkdir ${exp_folder}/res
src_filename="TVD-02_p2p5feature"

# how many segments available in ${exp_folder}/yuv folder
declare -i segment_num=`ls ${exp_folder}/bin/ -l | grep "^-" |wc -l`
echo $segment_num

parcat_proc=""
cur_idx=1
while (($cur_idx <= $segment_num))
do
    temp="${exp_folder}/bin/${src_filename}_"`printf "%04d" $cur_idx`".bin"
    parcat_proc="$parcat_proc $temp"
    # echo $parcat_proc
    cur_idx=`expr $cur_idx + 1`
    # break
done

outbitfile="${exp_folder}/res/decode_${src_filename}.bin"
outyuvfile="${exp_folder}/res/decode_${src_filename}.yuv"

echo
parcat_proc="tools/parcatStatic $parcat_proc $outbitfile"
echo $parcat_proc
$parcat_proc

echo
############## ./tools/parcatStatic  [<bitstream2> ...] <outbitfile>
decode_log_file="${exp_folder}/res/decode_${src_filename}.log"
decode_proc="tools/DecoderAppStatic -b $outbitfile -o $outyuvfile" 
echo $decode_proc
nohup $decode_proc > $decode_log_file 2>&1 &