cd /home/ctgvar/Desktop/vcm/parcat
pwd
# folders(./log, ./bin, ./yuv) to store the results
mkdir log
mkdir bin
mkdir yuv
################################################################
# select source file
src_file="./TVD-02.yuv" # -i
input_fmt="-wdt 1920 -hgt 1080 --InputBitDepth=8 --InputChromaFormat=420 --FrameRate=50"
##################################################################
# other parameters
encoder="./tools/EncoderAppStatic"
encoder_cfg="-c ./cfg/encoder_randomaccess_vtm.cfg" # -c ./cfg/xx.cfg
qp_rate=52
custom_para="-q ${qp_rate} --Level=6.2 --IntraPeriod=64 --GOPSize=32 --ConformanceWindowMode=1"
##################################################################
# encode range
FrameSkip=0
FramesToBeEncoded=636
encode_rng="--FrameSkip=${FrameSkip} --FramesToBeEncoded=${FramesToBeEncoded}"
##################################################################
# generate tgt_name
tgt_name="TVD-02"
tgtbin_file="./bin/${tgt_name}.bin" #-b
tgtyuv_file="./yuv/${tgt_name}.yuv" # -o
log_file="./log/${tgt_name}.txt" # > $log_file
##################################################################
# execution
encoding_prc="${encoder} ${encoder_cfg} ${custom_para} ${input_fmt} ${encode_rng} -i $src_file -b $tgtbin_file -o $tgtyuv_file"
echo $encoding_prc
nohup $encoding_prc > $log_file 2>&1 &

# ./tools/EncoderAppStatic -c ./cfg/encoder_randomaccess_vtm.cfg -q 52 --Level=6.2 --IntraPeriod=64 --GOPSize=32 --ConformanceWindowMode=1 -wdt 1920 -hgt 1080 --InputBitDepth=8 --InputChromaFormat=420 --FrameRate=50 --FramesToBeEncoded=636 -i "TVD-02.yuv"  -b "bin/TVD-02_lossless.bin" -o "yuv/TVD-02_lossless.yuv"

# ./tools/EncoderAppStatic -c ./cfg/encoder_randomaccess_vtm.cfg -q 52 --Level=6.2 --InputBitDepth=8 --InputChromaFormat=420 --FrameRate=50 --IntraPeriod=64 --GOPSize=32 --ConformanceWindowMode=1 --FrameSkip=0 --FramesToBeEncoded=64 --FramesToBeEncoded=636 -i "TVD-02.yuv" -wdt 1920 -hgt 1080 -b "bin/TVD-02_lossless.bin" -o "yuv/TVD-02_lossless.yuv"