#!/bin/bash

source ~/.bashrc

conda activate newenv


# Set variables

# DATASET="TCD-DEMAND"
DATASET="LRS3-NTCD"



SEGMENT=$1
TOTAL_SEGMENTS=$2

# CKPT_PATH="path_to_audio_only_model"
CKPT_PATH="path_to_av_model"



# DATA_DIR="/group_storage/corpus/source_separation/ICP52/TCD_DEMAND/new_tcd_demand_test.pkl"
DATA_DIR="/group_storage/corpus/source_separation/ICP52/LRS3_NTCD/new_lrs3_ntcd_test.pkl"


ALGO_TYPE="fudiffse" #"udiffse"   

SAVE_ROOT="/group_cacul/calcul/users/jayilo/ICP_52_results/grat/$DATASET"  


divide_s0hat="no"
set_v_to_zero="no"


if test "$ALGO_TYPE" = "fudiffse"
then

    TAG="fudiffuse_bs4"       
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=1.5


elif test "$ALGO_TYPE" = "udiffse"
then

    TAG="udiffuse"        
    NUM_E=30
    NUM_EM=5
    NBATCH=4
    STARTSTEP=0
    LMBD=1.5

else 
    echo "NOT AVAILABLE ALGO"
    break


fi


# Run command
python eval/evaluation.py \
    --dataset "$DATASET" \
    --segment "$SEGMENT" \
    --num_segments "$TOTAL_SEGMENTS" \
    --ckpt_path "$CKPT_PATH" \
    --algo_type "$ALGO_TYPE" \
    --tag "$TAG" \
    --data_dir "$DATA_DIR" \
    --save_root "$SAVE_ROOT" \
    --num_E "$NUM_E" \
    --num_EM "$NUM_EM" \
    --nbatch "$NBATCH" \
    --divide_s0hat "$divide_s0hat" \
    --set_v_to_zero "$set_v_to_zero" \
    --lambda "$LMBD"