#!/bin/bash

source ~/.bashrc

conda activate newenv

# Set variables
 
ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/GTA_aonly_tcd_speech_modeling_default_28M/TCD-DEMAND/fudiffse/fudiffuse_bs4"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/GTA_aonly_tcd_speech_modeling_default_28M/TCD-DEMAND/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/TCD-DEMAND/fudiffse/fudiffuse_bs4"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/TCD-DEMAND/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/flow_avse_enh_tcd_demand_best_sisdr/first_stage"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/flow_avse_enh_tcd_demand_best_sisdr/second_stage"


# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/GTA_aonly_tcd_speech_modeling_default_28M/LRS3-NTCD/fudiffse/fudiffuse_bs4"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/GTA_aonly_tcd_speech_modeling_default_28M/LRS3-NTCD/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/LRS3-NTCD/fudiffse/fudiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/LRS3-NTCD/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/flow_avse_enh_lrs3_ntcd_best_sisdr/first_stage"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/flow_avse_enh_lrs3_ntcd_best_sisdr/second_stage"


DATA_DIR="/group_calcul/calcul/users/jayilo/mydiffuse/eval/new_tcd_demand_test.pkl"
# DATA_DIR="/group_corpus/corpus/source_separation/ICP52/LRS3_NTCD/new_lrs3_ntcd_test.pkl"


DATASET="TCD-DEMAND"
# DATASET="LRS3-NTCD"



echo "$ENHANCED_DIR"
echo "$DATASET"
echo "$DATA_DIR"


# Run command, for LRS3-NTCD do not apply dnn_mos (due to the short length of some files)
python eval/statistics/compute_metrics.py \
    --enhanced_dir "$ENHANCED_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$ENHANCED_DIR" \
    --dataset  "$DATASET"
    # --dnn_mos
    # --input_metrics