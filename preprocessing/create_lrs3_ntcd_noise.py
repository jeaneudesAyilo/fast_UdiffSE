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
We create new mixtures of LRS3 test dataset and noise extracted from NTCD dataset, re-using the noise types and snrs used of the
ntcd-timit test. Thus, it is mismatched conditions. we mix each clean speech with only one combination of noise_types and snrs.  
"""



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="/group_storage/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/clean", help='path to clean speech in NTCD dataset')

    parser.add_argument("--test_LRS3_audio_path", type=str, default="/group_storage/corpus/audio_visual/LRS3_audios/test", help='path to audio files of LRS3')
    parser.add_argument("--save_dir", type=str, default="/group_storage/corpus/source_separation/ICP52/LRS3_NTCD/", help='Directory where the created mixtures will be saved')
    parser.add_argument("--save_pickle", type=str, default="/group_storage/corpus/source_separation/ICP52/LRS3_NTCD",  help='dir of the file that will record the metadata (snr,noise_type,speaker...) of all mixtures')

    parser.add_argument("--new_lrs3_demand_test_pickle", type=str, default="/group_storage/corpus/source_separation/ICP52/LRS3_DEMAND/new_lrs3_demand_test.pkl", help='path to the file that records the metadata of all mixtures made of LRS3 + DEMAND. It is created firstly')        
    parser.add_argument("--noise_dir", type=str, default="/group_storage/corpus/audio_visual/NTCD-TIMIT-noisy/{noise_type}/0/volunteers",  help='Path to NTCD noisy speech files per noise type')


    args = parser.parse_args()


    ntcd_test_speakers = ["09F", "24M", "26M", "27M", "33F", "40F", "47M", "47M", "49F", "56M"]

    TEST_noise_types = ["Babble", "Cafe", "Car", "LR", "White"]

    TEST_SNRs = [-5, 5]

    clean_dir = args.clean_dir 
    test_LRS3_audio_path = args.test_LRS3_audio_path
    save_dir = args.save_dir 

    noise_dir = args.noise_dir

    save_pickle = args.save_pickle 

    os.makedirs(save_dir, exist_ok=True)

    fs = 16000

    new_lrs3_demand_test = load_dict(args.new_lrs3_demand_test_pickle)


    subset="test"

    list_samples = []    
        
    np.random.seed(42)
    for i, info in enumerate(tqdm(new_lrs3_demand_test)):
        
        ##choose noise 
        noise_type = list(np.random.choice(TEST_noise_types, size=1, replace=False))[0]    
       
        noise_dir = noise_dir.format(noise_type=noise_type)


        #choose a noisy speech in ntcd test set
        ntcd_speaker_id = list(np.random.choice(ntcd_test_speakers, size=1, replace=False))[0]


        noisy_speech_tcd_set = librosa.util.find_files(os.path.join(noise_dir, ntcd_speaker_id, "straightcam"), ext='wav')
        
        noisy_speech_tcd =  list(np.random.choice(noisy_speech_tcd_set, size=1, replace=False))[0]
        
        ntcd_filename = noisy_speech_tcd.split("/")[-1]

        #get the clean speech corresponding to the noisy speech selected in ntcd test set
        clean_speech_tcd = os.path.join(clean_dir,ntcd_speaker_id,ntcd_filename) 
        
        assert os.path.isfile(clean_speech_tcd)

        ##get the noise by substracting the 
        s_tcd, s_sr = sf.read(clean_speech_tcd)
        assert fs==s_sr
        n_tcd, n_sr = sf.read(noisy_speech_tcd)
        assert fs==n_sr

        noise = n_tcd - s_tcd

        ##lrs3 clean speech
        s_lrs3, s_sr_lrs3 = sf.read(info['speech_file'])
        utt_len = len(s_lrs3) 
        assert s_sr_lrs3==fs
        speech_power = 1/len(s_lrs3)*np.sum(s_lrs3**2)     


        n_duplicate =1
        duplicate_noise = False


        while len(noise)-(utt_len) <=0:
            noise=np.concatenate((noise, noise))   
            n_duplicate +=1
            duplicate_noise = True


        assert len(noise)>utt_len, f"noise tcd : {noisy_speech_tcd}  speech tcd {clean_speech_tcd} speech lrs3 {info['speech_file']}"
        noise_start = int(np.random.randint(0, len(noise) - utt_len ))   

        noise = noise[noise_start:noise_start+utt_len]


        snr = list(np.random.choice(TEST_SNRs, size=1, replace=False))[0]

        noise_power = 1/len(noise)*np.sum(noise**2)
        noise_power_target = speech_power*np.power(10,-snr/10)
        k = noise_power_target / noise_power
        noise = noise * np.sqrt(k)
        x = s_lrs3 + noise



        info['noise_type'] =  noise_type
        
        info['snr'] =  snr 

        ##collect information about how the noise is built
        info['ntcd_clean_speech'] = clean_speech_tcd

        info['ntcd_noisy_speech'] = noisy_speech_tcd

        info['duplicate_noise'] = duplicate_noise

        info['n_duplicate'] = n_duplicate       

        info['noise_start'] = noise_start
        info['noise_end'] = noise_start+utt_len

        
        info['mix_file'] = os.path.join(save_dir, subset, info['speaker_id'], info['file_name'] + ".wav")
        os.makedirs( os.path.join(save_dir, subset , info['speaker_id']) , exist_ok=True) 
        sf.write(file = info['mix_file'], data = x, samplerate = fs)   
                    
        list_samples.append(info)


    save_dict(list_samples, os.path.join(save_dir,"new_lrs3_ntcd_test.pkl")) 





