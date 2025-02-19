import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import re
import shutil

import matplotlib.pyplot as plt
import torch
import cv2
import tempfile
import torch
import utils as avhubert_utils
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils

from copy_and_extract_hubert_feature import (
    build_avhubert_extractor,
    extract_avhubert_feature,
)
from torchaudio import load


fps = 30
fs = 16000
hop_length = 128  # "Window hop length. 128 by default.
n_fft = 510  # = win_length ie defautlf config in the data loader


def resample_video(video, target_num):
    ##from  https://github.com/msaadeghii/av-dkf/blob/master/dvae/utils/speech_dataset.py#L47C5-L55
    n, N = video.shape  # (4489, 129)
    ratio = N / target_num
    idx_lst = np.arange(target_num).astype(float)
    idx_lst *= ratio
    res = np.zeros((n, target_num))
    for i in range(target_num):
        res[:, i] = video[:, int(idx_lst[i])]
    return res


def resampling_and_extract_hubert_feature(
    df, clean_dir, ckpt_path, user_dir, is_finetune_ckpt=False
):

    avhubert_extractor = build_avhubert_extractor(
        ckpt_path=ckpt_path, user_dir=user_dir, is_finetune_ckpt=is_finetune_ckpt
    )

    for raw in tqdm(df.to_dict("records")):

        x, _ = load(raw["speech_file"])
        current_len = x.size(-1)
        filename = re.sub("\.\w*", "", os.path.basename(raw["speech_file"]))

        raw_video = np.load(raw["video_file_raw"])

        current_num_frames = (
            current_len / hop_length + 1
        )  # actual number of STFT frames

        resampled_raw_video = resample_video(
            raw_video, current_num_frames
        )  # resample v to the actual number of STFT frames. the shape becomes (4489,current_num_frames)

        resampled_avhubert_feature_file = f'{filename}_{raw["speaker_id"]}_{raw["noise_type"]}_{raw["snr"]}_hubert_resampled.npy'
        resampled_raw_video_file = f'{filename}_{raw["speaker_id"]}_{raw["noise_type"]}_{raw["snr"]}_Raw_resampled.npy'

        np.save(os.path.join(clean_dir, resampled_raw_video_file), resampled_raw_video)

        resampled_avhubert_feature = extract_avhubert_feature(
            model=avhubert_extractor, ndarray=resampled_raw_video
        )

        np.save(
            os.path.join(clean_dir, resampled_avhubert_feature_file),
            resampled_avhubert_feature.cpu().numpy(),
        )


if __name__ == "__main__":

    base_dir = (
        "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo"
    )

    clean_train_dir = os.path.join(base_dir, "train/clean")
    clean_valid_dir = os.path.join(base_dir, "valid/clean")
    clean_test_dir = os.path.join(base_dir, "test/clean")

    avhubert_ckpt_path = os.path.join(base_dir, "pretrained_model/finetune-model.pt")
    avhubert_user_dir = os.path.join(base_dir, "av_hubert/avhubert")

    ##these dataframes contain the original raw video (67*67) path, these paths point to the clean directory

    train_df = pd.read_csv(os.path.join(base_dir, "train_df.csv"))
    valid_df = pd.read_csv(os.path.join(base_dir, "valid_df.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "test_df.csv"))

    resampling_and_extract_hubert_feature(
        train_df,
        clean_train_dir,
        ckpt_path=avhubert_ckpt_path,
        user_dir=avhubert_user_dir,
    )
    print("end for training set...")
    resampling_and_extract_hubert_feature(
        valid_df,
        clean_valid_dir,
        ckpt_path=avhubert_ckpt_path,
        user_dir=avhubert_user_dir,
    )
    print("end for validation set...")
    resampling_and_extract_hubert_feature(
        test_df,
        clean_test_dir,
        ckpt_path=avhubert_ckpt_path,
        user_dir=avhubert_user_dir,
    )
    print("end for test set...")
