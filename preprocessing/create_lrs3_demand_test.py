import os
import glob
import pandas as pd
import shutil
import random 
import json
import numpy as np
import soundfile as sf
import argparse
import pyloudnorm as pyln
from tqdm import tqdm


from argparse import ArgumentParser
from six.moves import cPickle as pickle 
import librosa

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
 
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di



"""
We create new mixtures of LRS3 test dataset and demand noise dataset. We mix each clean speech with only one combination of noise_types and snrs.  
"""


fs =16000


#Similar to build train 
def generate_noisy_speech_without_duplicate(path, subset, noise_types, snrs, save_dir, saving_name_list):
    
    list_samples = []
   
    clean_files = librosa.util.find_files(os.path.join(path, subset), ext='wav')  ##pattern  : '.../LRS3_audios/test/0Fi83BHQsMA/00002.wav' 

    np.random.seed(42)

    for i,file in enumerate(tqdm(clean_files)):

        info = {}

        id_speaker = file.split("/")[-2]
        filename = file.split("/")[-1].replace(".wav","")  

        s, s_sr = sf.read(file)
        utt_len = len(s) 
        assert s_sr==fs
        speech_power = 1/len(s)*np.sum(s**2)

        noise_type = list(np.random.choice(noise_types, size=1, replace=False))[0]
        
        snr = list(np.random.choice(snrs, size=1, replace=False))[0]

        noise_file = os.path.join(demand_noise_dir, noise_type, 'ch01.wav') 

        n, n_sr = sf.read(noise_file)
        assert n_sr==fs

        noise_start = int(np.random.randint(0, len(n) - utt_len - 1))   

        n = n[noise_start:noise_start+utt_len]

        noise_power = 1/len(n)*np.sum(n**2)
        noise_power_target = speech_power*np.power(10,-snr/10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n


        info['speech_file'] = file
        info['mix_file'] = os.path.join(save_dir, subset, id_speaker, filename + ".wav")
        os.makedirs(os.path.join(save_dir, subset, id_speaker), exist_ok=True) 

        info['video_file_cropped_88'] = os.path.join(LRS3_cropped_silent_video_path, subset, id_speaker, filename+"_mouthcrop.mp4")
        info['video_file'] =  os.path.join(original_lrs3, subset, id_speaker, filename+".mp4")
        
        assert os.path.isfile(info['video_file']) 
        assert os.path.isfile(info['video_file_cropped_88']) 

        info['noise_type'] =  noise_type
        
        info['snr'] =  snr

        info['speaker_id'] = id_speaker
        info['file_name'] = filename

        info['duration'] = utt_len/fs

        info['noise_start'] = noise_start
        info['noise_end'] = noise_start+utt_len

        sf.write(file = info['mix_file'], data = x, samplerate = fs)   
                    
        list_samples.append(info)

    
    save_dict(list_samples, os.path.join(save_dir,saving_name_list))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--demand_noise_dir", type=str, default='/group_storage/corpus/source_separation/DEMAND', help='path to demand noise dataset')

    parser.add_argument("--lrs3_audio_path", type=str, default='/group_storage/corpus/audio_visual/LRS3_audios', help='path to audio files of LRS3')        
   
    parser.add_argument("--lrs3_cropped_silent_video_path", type=str, default="/group_storage/corpus/audio_visual/CROPPED_MOUTH_88/LRS3", help='path to LRS3 cropped ROI of mouth')        

    parser.add_argument("--original_lrs3_path", type=str, default="/group_storage/corpus/audio_visual/LRS3/", help='path to downloaded LRS3 dataset')        

    parser.add_argument("--save_dir", type=str, default="/group_storage/corpus/source_separation/ICP52/LRS3_DEMAND/", help='Directory where the created mixtures will be saved')
  
    parser.add_argument("--save_pickle", type=str, default="/group_calcul/calcul/users/jayilo/fast_UdiffSE/eval",  help='dir of the file that will record the metadata (snr,noise_type,speaker...) of all mixtures')
 
    args = parser.parse_args()


    demand_noise_dir = args.demand_noise_dir 
    LRS3_audio_path = args.lrs3_audio_path
    LRS3_cropped_silent_video_path = args.lrs3_cropped_silent_video_path
    original_lrs3 = args.original_lrs3_path 
    save_dir = args.save_dir

    save_pickle = args.save_pickle

    os.makedirs(save_dir, exist_ok=True)

    TEST_SNRs = [-5, 5]

    TEST_noise_types = ["TMETRO", "OOFFICE", "TBUS", "STRAFFIC", "SPSQUARE"]


    print("Start creating test new noisy lrs3_demand")
    generate_noisy_speech_without_duplicate(path=LRS3_audio_path,
                                            subset="test", 
                                            noise_types=TEST_noise_types, 
                                            snrs=TEST_SNRs, 
                                            save_dir = save_dir , 
                                            saving_name_list="new_lrs3_demand_test.pkl")
    print("End creating test new noisy lrs3_demand")



