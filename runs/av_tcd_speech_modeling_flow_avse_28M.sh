
source ~/.bashrc
conda activate newenv


python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 4 \
	--vfeat_processing_order cut_extract \
	--video_feature_type flow_avse \
	--backbone ncsnpp_flow_avse \
	--run_id av_tcd_speech_modeling_flow_avse_28M