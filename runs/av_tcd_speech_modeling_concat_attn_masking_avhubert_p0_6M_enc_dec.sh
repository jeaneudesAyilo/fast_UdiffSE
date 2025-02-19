
source ~/.bashrc
conda activate newenv


python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 4 \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert \
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
	--fusion concat_attn_masking \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id GTA_av_tcd_speech_modeling_concat_attn_masking_avhubert_p0_6M_enc_dec