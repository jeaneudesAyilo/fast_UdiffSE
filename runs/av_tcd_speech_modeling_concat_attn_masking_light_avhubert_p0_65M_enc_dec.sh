
source ~/.bashrc
conda activate newenv


python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 4 \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert \
	--backbone ncsnpp_continueconcat_attn_masking_noising \
	--fusion concat_attn_masking_light \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_65M_enc_dec