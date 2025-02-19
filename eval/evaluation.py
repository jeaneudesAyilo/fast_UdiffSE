#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import sys
import shutil
from glob import glob
from tqdm import tqdm
sys.path.append(".")
import json
import torch
from src.enhancement import UDiffSE, fUDiffSE
from src.utils import load_dict, save_dict, collect_rtf_pickle_files
import argparse
from datetime import datetime
import time
import soundfile as sf
from torchaudio import load
import subprocess



device = "cuda" if torch.cuda.is_available() else "cpu"


def speech_enhance(params):
    if params["dataset"] == "WSJ0":
        # Load file json
        with open(params["data_dir"], "r") as f:
            dataset = json.load(f)
    elif params["dataset"] in ["TCD-TIMIT", "LRS3-NTCD"]:
        dataset = load_dict(params["data_dir"])
             

    seg_id = params["segment"]
    ind_start = 0
    ind_end = len(dataset)

    if seg_id >= 0:
        num_files = len(dataset)
        segment_size = num_files // params["num_segments"]
        ind_start, ind_end = seg_id * segment_size, (seg_id + 1) * segment_size
        if seg_id == params["num_segments"] - 1:
            ind_end = num_files
        print(f"Evaluating files at [{ind_start},{ind_end}]")

    # Skip files that have already been processed
    files_processed = [
        x[:-4] for x in os.listdir(params["save_dir"]) if x.endswith(".wav")
    ]

    # Init evaluation
    print(f"\nTotal number of files to evaluate: {ind_end - ind_start}\n")

    verbose = False
    compute_rtf = True

    if params["algo_type"] == "udiffse":
        enhance = UDiffSE(
            ckpt_path=params["ckpt_path"], num_E=params["num_E"], verbose=verbose, set_v_to_zero=params["set_v_to_zero"]
        )
    
    
    elif params["algo_type"] == "fudiffse":        
        enhance = fUDiffSE(
            ckpt_path=params["ckpt_path"], num_E=params["num_E"], verbose=verbose, set_v_to_zero=params["set_v_to_zero"]
        )


    if params["dataset"] in ["WSJ0","VB"]:

        for ind_i, (ind_mix, mix_info) in (enumerate(dataset.items())):
            if (
                ind_start <= ind_i < ind_end
                and mix_info["utt_name"] not in files_processed
            ):
                utt_name = mix_info["utt_name"]
                mix_file = mix_info["noisy_wav"].format(noisy_root=params["noisy_root"])
                clean_file = mix_info["clean_wav"].format(
                    clean_root=params["clean_root"]
                )
                recon_file = os.path.join(params["save_dir"], utt_name + ".wav")

                start_time = time.time() # datetime.now()

                # Enhance algo, clean_file only used if we run monitor performance
                s_hat, _ = enhance.run(
                    mix_file=mix_file,
                    clean_file=None,
                    nmf_rank=params["nmf_rank"],
                    num_EM=params["num_EM"],
                    lmbd=params["lambda"],
                    nbatch=params["nbatch"],
                    startstep=params["startstep"],
                    
                )
                end_time = time.time() #datetime.now()

                duration = end_time-start_time
                # duration = duration.total_seconds()    
                if compute_rtf:
                    save_dir_rtf = params["save_dir"]
                    mix,sr = load(mix_file)  
                    assert sr==sr
                    rtf = (duration)/(mix.shape[1]/sr)   

                    di_ = {'speaker_id':mix_info['p_id'],
                        'file_name': utt_name,
                        'noise_type': mix_info['noise_type'],
                        'snr':mix_info['snr'],
                        'enhanced':utt_name,
                        'rtf':rtf
                        }  

                    rtf_file = os.path.join(save_dir_rtf, utt_name + ".pkl")

                    save_dict(di_, rtf_file) 

                sf.write(recon_file, s_hat, 16000)

        if compute_rtf:
            collect_rtf_pickle_files(save_dir_rtf, dataset)
        
        if len([x[:-4] for x in os.listdir(params["save_dir"]) if x.endswith(".wav")]) == len(dataset):

            if params["datasets"] in ["WSJ0"]  :                    
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {params["save_dir"]}  --data_dir {params["data_dir"]} --save_dir {params["save_dir"]}  --dataset  {params["dataset"]} --dnn_mos', shell=True)
                stdout, stderr = output.communicate() 

            else:
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {params["save_dir"]}  --data_dir {params["data_dir"]} --save_dir {params["save_dir"]}  --dataset  {params["dataset"]}', shell=True)
                stdout, stderr = output.communicate() 


    elif params["dataset"] in ["TCD-TIMIT", "LRS3-NTCD"]:

        for ind_i, mix_info in enumerate(dataset):
            if (
                ind_start <= ind_i < ind_end
                and f"{mix_info['speaker_id']}_{mix_info['noise_type']}_{mix_info['snr']}_{mix_info['file_name']}"
                not in files_processed
            ):
                mix_file = mix_info["mix_file"]
                clean_file = mix_info["speech_file"]
                save_name = f"{mix_info['speaker_id']}_{mix_info['noise_type']}_{mix_info['snr']}_{mix_info['file_name']}"
                

                ##collect the suitable video file
                if not enhance.audio_only :  

                    if enhance.vfeat_processing_order in ["cut_extract"]:
                    
                        if params["dataset"] in ["TCD-TIMIT"]:

                            if enhance.video_feature_type in  ["resnet", "avhubert", "raw_image"]: 
                                video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/TCD-TIMIT/test/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"                
                            
                            elif enhance.video_feature_type in  ["flow_avse"]: 
                                video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/TCD-TIMIT/test/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"                                             


                        elif params["dataset"] in ["LRS3-NTCD"]:                            

                            if enhance.video_feature_type in  ["resnet", "avhubert", "raw_image"]: 
                                video_path  =  "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/LRS3/test/{speaker_id}/{filename}_mouthcrop.mp4"             
                            
                            elif enhance.video_feature_type in  ["flow_avse"]: 
                                video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/LRS3/test/{speaker_id}/{filename}_mouthcrop.mp4"
                    
                            
                        video_file = video_path.format(speaker_id=mix_info['speaker_id'], filename=mix_info['file_name'])
                            

                else:
                    video_file = None


                recon_file = os.path.join(params["save_dir"], save_name + ".wav")
                start_time = time.time() #datetime.now()
                s_hat, _ = enhance.run(
                    mix_file=mix_file,
                    clean_file=clean_file,
                    video_file = video_file,
                    nmf_rank=params["nmf_rank"],
                    num_EM=params["num_EM"],
                    lmbd=params["lambda"],
                    nbatch=params["nbatch"],
                    startstep=params["startstep"],
                                       
                )
                end_time = time.time() #datetime.now()

                duration = end_time-start_time
                # duration = duration.total_seconds()                                

                ##compute rtf and save it as a file
                if compute_rtf:
                    save_dir_rtf = params["save_dir"]
                    mix,sr = load(mix_file)  
                    assert sr==sr
                    rtf = (duration)/(mix.shape[1]/sr)      
            
                    di_ = {'speaker_id':mix_info['speaker_id'],
                        'file_name': mix_info['file_name'],
                        'noise_type': mix_info['noise_type'],
                        'snr':mix_info['snr'],
                        'enhanced':save_name,
                        'rtf':rtf
                        }    
                    
                    rtf_file = os.path.join(save_dir_rtf, save_name + ".pkl")

                    save_dict(di_, rtf_file) 
                
                sf.write(recon_file, s_hat, 16000)  

        if compute_rtf:
            collect_rtf_pickle_files(save_dir_rtf,dataset)


        if len([x[:-4] for x in os.listdir(params["save_dir"]) if x.endswith(".wav")]) == len(dataset):

            if "TCD" in params["dataset"] and "LRS3" not in params["dataset"]: ## files are long enough to compute dnn mos metrics
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {params["save_dir"]}  --data_dir {params["data_dir"]} --save_dir {params["save_dir"]}  --dataset  {params["dataset"]} --dnn_mos', shell=True)

            elif "LRS3" in params["dataset"] :
                output = subprocess.Popen(f'python eval/statistics/compute_metrics.py  --enhanced_dir {params["save_dir"]}  --data_dir {params["data_dir"]} --save_dir {params["save_dir"]}  --dataset  {params["dataset"]}', shell=True)
            
            stdout, stderr = output.communicate() 
        
                
                
class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument(
            "--segment",
            type=int,
            choices=[-1] + list(range(0, 1000)),
            default=-1,
            help="Segment ID of the test files to evaluate on",
        )
        self.parser.add_argument(
            "--num_segments",
            type=int,
            default=10,
            help="total number of segments to evaluate on",
        )
        self.parser.add_argument(
            "--nbatch",
            type=int,
            default=4,
            help="number of batches for parallel estimation",
        )
        self.parser.add_argument(
            "--lambda",
            type=float,
            default=1.5,
            help="weight parameter for posterior sampler",
        )

        self.parser.add_argument(
            "--dataset",
            type=str,
            default="WSJ0",
            choices=["TCD-TIMIT", "WSJ0","VB", "LRS3-NTCD"],
            help="dataset",
        )
        self.parser.add_argument(
            "--ckpt_path",
            type=str,
            required=True,
            help="path to the ckpt",
        )
        self.parser.add_argument(
            "--algo_type",
            type=str,
            default="udiffse",
            choices=["udiffse","fudiffse"],
            help="SE algorithm: udiffse or fudiffse",
        )
        self.parser.add_argument(
            "--tag",
            type=str,
            default="orig",
            help="Tag given to the specific version of SE algorithm",
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="wsj_test.json",
            help="json or pickle file for audios to be enhanced",
        )
        self.parser.add_argument(
            "--save_root", type=str, default="/tmp", help="path to denoised data"
        )

        self.parser.add_argument("--nmf_rank", type=int, default=4, help="NMF rank")

        self.parser.add_argument(
            "--num_EM", type=int, required=True, help="number of EM iterations. default for udiffse :5, for  fudiffse: 1"
        )
        self.parser.add_argument(
            "--num_E", type=int, required=True, help="number of iterations in the E-step. default for udiffse and fudiffse: 30"
        )
        self.parser.add_argument(
            "--startstep", type=int, default=0, help="start step (0 or greater)"
        )

        self.parser.add_argument(
            "--divide_s0hat", type=str, choices=("yes", "no"), default="no", help="whether to divide s0hat by gamma_t or not, this applies only for fudiffse"
        )    

        self.parser.add_argument(
            "--set_v_to_zero", type=str, choices=("yes", "no"), default="no", 
            help="whether to set v to 0 or not. If yes, then the video feature v=0, if no use the right value of v. But it is better to use an audio-only model, this was purely for experimentation purpose."
        )         
   

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params


if __name__ == "__main__":
    params = Options().get_params()

    if params["dataset"] == "VB":
        params["noisy_root"] = "/group_storage/corpus/source_separation/VoiceBankDEMAND/noisy_testset_wav_16k"
        params["clean_root"] = "/group_storage/corpus/source_separation/VoiceBankDEMAND/clean_testset_wav_16k"
    elif params["dataset"] == "WSJ0":
        params["noisy_root"] = (
            "/group_storage/corpus/source_separation/QUT_WSJ0/test"
        )
        params["clean_root"] = (
            "/group_storage/corpus/source_separation/WSJ0_SE/wsj0_si_et_05"
        )

    params["save_dir"] = str(
        os.path.join(
            params["save_root"],
            params["ckpt_path"].split("/")[-2], #os.path.basename(params["ckpt_path"])[:-5],
            params["dataset"],
            params["algo_type"],
            str(params["lambda"]),
            params["tag"],
        )
    )

    if not os.path.isdir(params["save_dir"]):
        os.makedirs(params["save_dir"], exist_ok=True)

    # save the input args
    args_file = f"{params['save_dir']}/commandline_args.txt"
    with open(args_file, "w") as f:
        json.dump(params, f, indent=2)

    source_file = "src/enhancement.py"
    destination_directory = params["save_dir"]
    if not os.path.exists(os.path.join(destination_directory, "enhancement.py")):
        shutil.copy(source_file, destination_directory)

    speech_enhance(params)
