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

# import utils as avhubert_utils
from argparse import Namespace

# import fairseq
from fairseq import checkpoint_utils, options, tasks, utils

from copy_and_extract_hubert_feature import (
    build_avhubert_extractor,
    extract_avhubert_feature,
)

# from torchaudio import load


def load_array(path=None, ndarray=None):

    if ndarray is not None and path == None:
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


base_dir = (
    "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo"
)

avhubert_ckpt_path = os.path.join(base_dir, "pretrained_model/finetune-model.pt")
avhubert_user_dir = os.path.join(base_dir, "av_hubert/avhubert")

avhubert_extractor = build_avhubert_extractor(
    avhubert_ckpt_path, avhubert_user_dir, is_finetune_ckpt=False
)

a = torch.randn(4489, 3)
b = extract_avhubert_feature(model=avhubert_extractor, ndarray=a.numpy())
print(b.shape)
print("######### Holy God fire")
