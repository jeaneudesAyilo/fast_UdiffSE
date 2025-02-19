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


def load_array(path, ndarray=None):

    if ndarray != None and path == None:
        arr = ndarray  ##(4489,t)
    elif ndarray == None and path != None:
        arr = np.load(path)
    else:
        raise ValueError("Should not specify both path and ndarray")

    arr = np.transpose(arr.reshape(67, 67, -1), axes=(1, 0, 2))
    arr = torch.from_numpy(np.transpose(arr.reshape(67, 67, -1), axes=(2, 0, 1)))
    arr = torch.FloatTensor(arr)
    # plt.imshow(arr[0,...], cmap='gray')
    # plt.show()
    # print(arr.shape); print(arr.max())
    return arr


def load_array_2(path, ind=0):
    import torchvision.transforms as trf

    arr = np.load(path)
    arr = np.transpose(arr.reshape(67, 67, -1), axes=(1, 0, 2))
    arr = trf.ToTensor()(arr)
    plt.imshow(arr[ind, ...], cmap="gray")
    plt.show()


def build_avhubert_extractor(ckpt_path, user_dir, is_finetune_ckpt=False):
    utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

    model = models[0]
    if hasattr(models[0], "decoder"):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model

    model.cuda()

    model.eval()

    return model


def extract_avhubert_feature(model, video_path=None, ndarray=None):
    frames = load_array(path=video_path, ndarray=ndarray)
    # print(f"frames.shape {frames.shape}")

    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()

    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer
        feature, _ = model.extract_finetune(
            source={"video": frames, "audio": None},
            padding_mask=None,
            output_layer=None,
        )
        feature = feature.squeeze(dim=0)

    return feature


def copy_and_extract_hubert_feature(
    df, clean_dir, ckpt_path, user_dir, is_finetune_ckpt=False
):

    avhubert_extractor = build_avhubert_extractor(
        ckpt_path=ckpt_path, user_dir=user_dir, is_finetune_ckpt=is_finetune_ckpt
    )

    for raw in tqdm(df.to_dict("records")):

        filename = re.sub("\.\w*", "", os.path.basename(raw["speech_file"]))

        renamed_raw_video_file = (
            f'{filename}_{raw["speaker_id"]}_{raw["noise_type"]}_{raw["snr"]}_Raw.npy'
        )

        ##copy the raw video files into the clean directory
        shutil.copy(
            raw["video_file_raw"], os.path.join(clean_dir, renamed_raw_video_file)
        )

        avhubert_feature_file = f'{filename}_{raw["speaker_id"]}_{raw["noise_type"]}_{raw["snr"]}_hubert.npy'

        feature = extract_avhubert_feature(
            model=avhubert_extractor, video_path=raw["video_file_raw"]
        )

        np.save(os.path.join(clean_dir, avhubert_feature_file), feature.cpu().numpy())


if __name__ == "__main__":

    base_dir = (
        "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo"
    )

    train_df = pd.read_csv(os.path.join(base_dir, "train_df.csv"))
    valid_df = pd.read_csv(os.path.join(base_dir, "valid_df.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "test_df.csv"))

    clean_train_dir = os.path.join(base_dir, "train/clean")
    clean_valid_dir = os.path.join(base_dir, "valid/clean")
    clean_test_dir = os.path.join(base_dir, "test/clean")

    avhubert_ckpt_path = os.path.join(base_dir, "pretrained_model/finetune-model.pt")
    avhubert_user_dir = os.path.join(base_dir, "av_hubert/avhubert")

    ##just add the original raw video (67*67) path into dataframe, even if it wont be used for the moment. We will copy them to the corresponding clean directory

    for df in [train_df, valid_df, test_df]:
        df["video_file_raw"] = df["video_file"].apply(
            lambda x: os.path.join(
                os.path.dirname(x), os.path.basename(x.replace("RawVF.npy", "Raw.npy"))
            )
        )

    train_df.to_csv(os.path.join(base_dir, "train_df.csv"), index=False)
    valid_df.to_csv(os.path.join(base_dir, "valid_df.csv"), index=False)
    test_df.to_csv(os.path.join(base_dir, "test_df.csv"), index=False)

    copy_and_extract_hubert_feature(
        train_df,
        clean_train_dir,
        ckpt_path=avhubert_ckpt_path,
        user_dir=avhubert_user_dir,
    )
    copy_and_extract_hubert_feature(
        valid_df,
        clean_valid_dir,
        ckpt_path=avhubert_ckpt_path,
        user_dir=avhubert_user_dir,
    )
    copy_and_extract_hubert_feature(
        test_df,
        clean_test_dir,
        ckpt_path=avhubert_ckpt_path,
        user_dir=avhubert_user_dir,
    )

    print("#### End ####")
