#!/usr/bin/env python
# coding: utf-8

# # Hands-on tutorial for AV-HuBERT
#
# In this notebook, we show-case how to use pre-trained models for:
# * lip reading
# * feature extraction

# ## Preliminaries
# This section installs necessary python packages for the other sections. Run it first.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch

import cv2
import tempfile
import torch
import utils as avhubert_utils
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils

# from IPython.display import HTML


# In[ ]:


# In[2]:


def load_array(path):
    arr = np.load(path)
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


# In[3]:


def extract_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
    utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    # frames = #avhubert_utils.load_video(video_path)
    print(f"task, {task}")
    print(f"saved_cfg, {saved_cfg}")

    frames = load_array(video_path)
    # print(f"frames.shape {frames.shape}")

    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    model = models[0]
    if hasattr(models[0], "decoder"):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")
    model.cuda()
    model.eval()
    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer
        feature, _ = model.extract_finetune(
            source={"video": frames, "audio": None},
            padding_mask=None,
            output_layer=None,
        )
        feature = feature.squeeze(dim=0)
    # print(f"Video feature shape: {feature.shape}")
    return feature


# In[ ]:


# In[4]:


mouth_roi_path = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo/sa1Raw.npy"
ckpt_path = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo/pretrained_model/finetune-model.pt"
user_dir = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo/av_hubert/avhubert"

feature = extract_visual_feature(mouth_roi_path, ckpt_path, user_dir)


# In[5]:


feature.shape


# In[ ]:
