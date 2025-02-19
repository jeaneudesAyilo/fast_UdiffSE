
source ~/.bashrc
conda activate newenv


python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 4 \
	--vfeat_processing_order default \
	--video_feature_type resnet \
	--backbone ncsnpp \
	--audio_only \
	--run_id aonly_tcd_speech_modeling_default_65M