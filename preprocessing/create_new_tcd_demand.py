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


def getTIMITclean(
    subset,
    data_dir="/group_storage/corpus/audio_visual/TCD-TIMIT/",
    t="_data_NTCD",
):
    if subset == "valid":
        subset = "val"
    t1 = subset + t
    if subset == "test":
        t1 = os.path.join(t1, "clean")
    clean_files = sorted(
        [
            os.path.join(root, name)
            for root, dirs, files in os.walk(os.path.join(data_dir, t1))
            for name in files
            if name.endswith(".wav")
        ]
    )
    return clean_files

"""
We create new mixtures of tcd-timit dataset and demand noise dataset. In the training/val, we mix each clean speech  
with only one combination of noise_types and snrs. In the test, we select just 10 clean speech per test speakers, 
and mix each of these 10 files with all the combinations of snrs and noise types chosen for test. """



fs =16000

#Build train 
def generate_noisy_speech_without_duplicate(subset, noise_types, snrs, save_dir, saving_name_list):
    
    list_samples = []

    np.random.seed(42)
    clean_files = getTIMITclean(subset)  ##pattern  : '.../train_data_NTCD/01M/sa2.wav'

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

        info['video_file'] = os.path.join(tcd_timit_original,subset,id_speaker,"straightcam", filename+".mp4")
        assert os.path.isfile(info['video_file']) 

        info['noise_type'] =  noise_type
        
        info['snr'] =  snr

        info['speaker_id'] = id_speaker
        info['file_name'] = filename

        info['duration'] = utt_len/fs

        info['noise_start'] = noise_start
        info['noise_end'] = noise_start+utt_len

        sf.write(file = info['mix_file'], data = x, samplerate = fs)   
                    
        list_samples.append(info)

    
    save_dict(list_samples, os.path.join(save_dir,saving_name_list)) #"test_tcd_demand.pkl"



##Build test

def generate_noisy_speech_test_with_duplicate(path, subset, noise_types, snrs, save_dir, saving_name_list):

    list_samples = []
    
    id_speakers= os.listdir(path)


    #cartesian product
    all_combinations_test = []
    for a in noise_types:
        for b in snrs:
            all_combinations_test.append((a,b))

    np.random.seed(42)
    for i, id_speaker in enumerate(tqdm(id_speakers)):
        
        speaker_path = os.path.join(path, id_speaker)

        clean_speaker_files = librosa.util.find_files(speaker_path, ext='wav')

        selected_files = list(np.random.choice(clean_speaker_files, size=10, replace=False))

        for file in selected_files:

            for (noise_type, snr) in all_combinations_test :

                info = {}

                filename = file.split("/")[-1].replace(".wav","")
                s, s_sr = sf.read(file)
                utt_len = len(s) 
                assert s_sr==fs
                speech_power = 1/len(s)*np.sum(s**2)

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
                info['mix_file'] = os.path.join(save_dir, subset, id_speaker, f"{filename}_{noise_type}_{snr}" + f".wav")
                os.makedirs(os.path.join(save_dir, subset, id_speaker), exist_ok=True) 

                info['video_file'] = os.path.join(tcd_timit_original,subset,id_speaker,"straightcam", filename+".mp4")
                assert os.path.isfile(info['video_file']) 

                info['noise_type'] =  noise_type
                
                info['snr'] =  snr

                info['speaker_id'] = id_speaker
                info['file_name'] = filename

                info['duration'] = utt_len/fs

                info['noise_start'] = noise_start
                info['noise_end'] = noise_start+utt_len

                sf.write(file = info['mix_file'], data = x, samplerate = fs)

                list_samples.append(info)

    save_dict(list_samples, os.path.join(save_pickle,saving_name_list))         




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--demand_noise_dir", type=str, default='/group_storage/corpus/source_separation/DEMAND', help='path to demand noise dataset')
    parser.add_argument("--tcd_timit_original", type=str, default='/group_storage/corpus/audio_visual/TCD-TIMIT_original/volunteers', help='path to downloaded TCD-TIMIT dataset')        
    parser.add_argument("--save_dir", type=str, default="/group_storage/corpus/source_separation/ICP52/TCD_DEMAND/", help='Directory where the created mixtures will be saved')
    parser.add_argument("--save_pickle", type=str, default="/group_calcul/calcul/users/jayilo/fast_UdiffSE/eval",  help='dir of the file that will record the metadata (snr,noise_type,speaker...) of all mixtures')
    parser.add_argument("--tcd_test_data_path", type=str, default="/group_storage/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/clean",  help='Path to clean speech dir of NTCD')
 
    args = parser.parse_args()


    demand_noise_dir = args.demand_noise_dir
    tcd_timit_original = args.tcd_timit_original
    save_dir = args.save_dir
    save_pickle = args.save_pickle

    tcd_test_data_path = args.tcd_test_data_path

    os.makedirs(save_dir, exist_ok=True)

    TRAINVAL_SNRs = [-10, 0, 10]
    TEST_SNRs = [-5, 5]

    TRAINVAL_noise_types = ["DKITCHEN", "OMEETING", "PRESTO", "PSTATION", "NPARK"] 
    TEST_noise_types = ["TMETRO", "OOFFICE", "TBUS", "STRAFFIC", "SPSQUARE"]



    print("Start creating train new noisy  tcd_demand")
    generate_noisy_speech_without_duplicate(subset="train", 
                                            noise_types=TRAINVAL_noise_types, 
                                            snrs=TRAINVAL_SNRs, 
                                            save_dir = save_dir , 
                                            saving_name_list="new_tcd_demand_train.pkl")
    print("End creating train new noisy  tcd_demand")



    print("Start creating val new noisy  tcd_demand")
    generate_noisy_speech_without_duplicate(subset="valid", 
                                            noise_types=TRAINVAL_noise_types, 
                                            snrs=TRAINVAL_SNRs, 
                                            save_dir = save_dir , 
                                            saving_name_list="new_tcd_demand_val.pkl")
    print("End creating val new noisy tcd_demand")


    print("Start creating test new noisy  tcd_demand")
    generate_noisy_speech_test_with_duplicate(path=tcd_test_data_path, 
                                            subset="test", 
                                            noise_types=TEST_noise_types, 
                                            snrs=TEST_SNRs, 
                                            save_dir=save_dir, 
                                            saving_name_list='new_tcd_demand_test.pkl')
    print("End creating test new noisy  tcd_demand")


