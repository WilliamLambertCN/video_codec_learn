__V1.4__ updating with p2p5 features and priorart integrated. Also adding support features and methods selection and auto logging featnames with methods __exp_ft_date_feat_methods__ in I.2    

__All scripts working in /Anchors/parcat/ and /dataset folder, a few path changes may be required in scripts' head.__ You can simply copy these scripts to other folder, all results will be create on the same level directories.

# I.Feature Extract Process

1. build VTM 12.0 software first, you will need BitstreamExtractorAppStatic  EncoderAppStatic, StreamMergeAppStatic, DecoderAnalyserAppStatic, parcatStatic, SubpicMergeAppStatic, DecoderAppStatic, SEIRemovalAppStatic (in VVCSoftware_VTM-VTM-12.0/bin/*).  
In my case, I copy all from the VVCSoftware_VTM-VTM-12.0/bin/* folder to /Anchors/parcat/tools folder to simplify usage.

2. Use 3MP4_split.py to split tvd mp4 dataset into a few .jpg images.

3. Use __5scripts_torun_parallel_tvd_extract_p_segmentation.sh__ to run __4tvd_extract_p_segmentation.sh__, which will run __4tvd_extract_p_layers.py__.  All results will be saved in __exp_ft_date_feat_methods__ folders. (eg.exp_ft_01-12-12-40-03_p2p5_remapping), according to logfiles in parcat folder(eg. TVD-01_yuvinfo_padding_segmentation.txt)  
P.S. __5scripts_torun_parallel_tvd_extract_p_segmentation.sh__ including TVD dataset path and cuda_env parameters, and specify the log file for different methods.  
__4tvd_extract_p_segmentation.sh__ needn't to modified, usually.


4. Use __6concat_singleyuv.sh__ to concat these single frames' yuv into exp_foler's XX_res folder. (__depanding frames #, may take hours, using top to see backend process__)

# II.Encoding and Decoding Process  

1. Use 1scripts_torun_parallel_encoding.sh to run 1parallel_encoding_safemode.sh.  
P.S. usually no changes required in 1parallel_encoding_safemode.sh. Parameter Settings should be modified in 1scripts_torun_parallel_encoding.sh 
>Examples:  
bash 1scripts_torun_parallel_encoding.sh 1  # anything other than 0, for safemode, no actual running, just to check the commands no infinite loop  
bash 1scripts_torun_parallel_encoding.sh 0 # 0 for actual running
(in case that you want to early stop, referring to II.2)

2. Read and run commands in 0cpu_usage_info.txt in to check cpu usage, if you need. In case to stop encoding, run commands in it to generate EncoderToKill.sh. In case of want to check the file size, look at the commands in 0diskinfo.txt

3. Use 2concat.sh to concat those parallel encoded bitstream files into one file.

4. Look at examples in 3lookpics_transferfiles.txt to transfer files in dokcer into hostmachine, and evaluate its yuv performances.
