import IPython
from torchaudio import load
from src.utils import show_spec
import os
import numpy as np
from six.moves import cPickle as pickle

from src.enhancement import UDiffSE, fUDiffSE

ckpt_audio = "./checkpoints/aonly_tcd_speech_modeling_default_28M.ckpt"

num_E = 30  # Number of E-step iterations (reverse diffusion process)
verbose = True


udiffse_aonly = UDiffSE(ckpt_path=ckpt_audio, num_E=num_E, verbose=verbose)
s_clean, S_clean = udiffse_aonly.prior_sampler(clean_file = None, vfile_path = None)
