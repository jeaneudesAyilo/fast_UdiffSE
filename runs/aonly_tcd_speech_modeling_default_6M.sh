
source ~/.bashrc
conda activate newenv


python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 4 \
	--vfeat_processing_order default \
	--video_feature_type resnet \
	--backbone ncsnpp6M \
	--audio_only \
	--run_id GTA_aonly_tcd_speech_modeling_default_6M