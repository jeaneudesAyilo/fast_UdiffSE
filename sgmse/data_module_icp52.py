import os
from os.path import join
import torch
import re
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
from random import randrange
import json
import librosa
from sgmse.util.utils_video import load_array, resample_video_numpy, get_mouthroi_audio_pair_load_raw_video


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


def getTIMITnoisy(
    subset,
    data_dir="/group_storage/corpus/audio_visual/NTCD-TIMIT-noisy/",
    t="Babble/15/volunteers",
):
    if subset == "train":
        exclude_folders = [
            "06M",
            "09F",
            "17F",
            "20M",
            "24M",
            "26M",
            "27M",
            "33F",
            "35M",
            "38F",
            "40F",
            "42M",
            "47M",
            "49F",
            "52M",
            "56M",
            "59F",
        ]
        noisy_files = sorted(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(os.path.join(data_dir, t))
                for name in files
                if (
                    name.endswith(".wav")
                    and not any(
                        exclude_folder in root for exclude_folder in exclude_folders
                    )
                )
            ]
        )
    elif subset == "valid":
        include_folders = ["06M", "17F", "20M", "35M", "38F", "42M", "52M", "59F"]
        noisy_files = sorted(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(os.path.join(data_dir, t))
                for name in files
                if (
                    name.endswith(".wav")
                    and any(
                        include_folder in root for include_folder in include_folders
                    )
                )
            ]
        )
    elif subset == "test":
        include_folders = [
            "09F",
            "24M",
            "26M",
            "27M",
            "33F",
            "40F",
            "47M",
            "49F",
            "56M",
        ]
        noisy_files = sorted(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(os.path.join(data_dir, t))
                for name in files
                if (
                    name.endswith(".wav")
                    and any(
                        include_folder in root for include_folder in include_folders
                    )
                )
            ]
        )
    else:
        raise NotImplementedError(f"Subset format {subset} unknown!")
    return noisy_files


def get_window(window_type, window_length):
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(
        self,
        data_dir,
        subset,
        dummy,
        shuffle_spec,
        num_frames,
        audio_only,
        video_feature_type,
        vfeat_processing_order, 
        format="default",
        normalize="clean",
        spec_transform=None,
        stft_kwargs=None,
        ispecs_kwargs = None,
        return_time=False,
        spectogram_learning=False,  
        impose_batch_1 = False,            
        **ignored_kwargs,
    ):
        self.return_time = return_time

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + "/clean/*.wav"))
            # self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
        elif format == "tcd-timit":
            data_dir = "/group_storage/corpus/audio_visual/TCD-TIMIT/"
            t = "_data_NTCD"
            self.clean_files = getTIMITclean(subset, data_dir, t)

            # data_dir="/group_storage/corpus/audio_visual/NTCD-TIMIT-noisy/"
            # t='Babble/15/volunteers'
            # self.noisy_files = getTIMITnoisy(subset, data_dir, t)
        elif format == "wsj0":
            data_dir = "/group_storage/corpus/speech_recognition/wsj0_wav"
            dic = {
                "train": "**/si_tr_s/**/*.wav",
                "valid": "**/si_dt_05/**/*.wav",
                "test": "**/si_et_05/**/*.wav",
            }
            self.clean_files = sorted(glob(data_dir + dic[subset], recursive=True))
            assert audio_only==True , print(f"If format == 'wsj0' video is not available,please add --audio_only to your command")
            

        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")


        if vfeat_processing_order == "cut_extract":
            assert format in ["tcd-timit"]
            assert audio_only == False
          
            if format =="tcd-timit":
                
                ##use the mouths cropped with index (48,68) and 88*88 roi,            
                if video_feature_type in ["avhubert", "resnet", "raw_image",]:
                    self.video_size=88
                    self.video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/TCD-TIMIT/{subset}/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"                
                                
                if video_feature_type == "flow_avse":
                    self.video_size=112
                    self.video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/TCD-TIMIT/{subset}/{speaker_id}/straightcam/{filename}_mouthcrop.mp4" 

                self.fps = 25



        self.audio_only = audio_only
        self.video_feature_type = video_feature_type
        self.vfeat_processing_order = vfeat_processing_order
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.spectogram_learning = spectogram_learning
        self.impose_batch_1 = impose_batch_1
        self.format = format
        self.subset = subset
        

        assert all(
            k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]
        ), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert (
            self.stft_kwargs.get("center", None) == True
        ), "'center' must be True for current implementation"
        self.ispecs_kwargs = ispecs_kwargs


    def __getitem__(self, i):
        
        if self.vfeat_processing_order == "default": 
            assert self.audio_only==True , print(f"If self.vfeat_processing_order == 'default' video is not available")
            self.video_size = None

            x, _ = load(self.clean_files[i]) 

                                     
            if not self.impose_batch_1 : ##we no longer need to crop the audio to make all element in a batch to have same length
                # formula applies for center=True
                target_len = (self.num_frames - 1) * self.hop_length
                current_len = x.size(-1)
                pad = max(target_len - current_len, 0)
                if pad == 0:
                    # extract random part of the audio file
                    if self.shuffle_spec:
                        start = int(np.random.uniform(0, current_len - target_len))
                    else:
                        start = int((current_len - target_len) / 2)
                    x = x[..., start : start + target_len]
                else:
                    # pad audio if the length T is smaller than num_frames
                    x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

            # normalize w.r.t to the noisy or the clean signal or not at all
            # to ensure same clean signal power in x and y.
            if self.normalize == "clean":
                normfac = x.abs().max()
            elif self.normalize == "not":
                normfac = 1.0
            x = x / normfac

            if self.return_time:
                return x

            X = torch.stft(x, **self.stft_kwargs)

            X = self.spec_transform(X)  # normalises if set to transform for normalising

            if self.spectogram_learning:
                X = X.abs().pow(2)

            return X


        elif self.vfeat_processing_order == "cut_extract": 
            assert self.audio_only ==False

            x, fs = load(self.clean_files[i])    
            assert fs == 16000 

            target_duration = 2.04
            target_len = int(target_duration * fs)

            target_video_nbframe = int(target_duration*self.fps)  #51
            
            current_len = x.size(-1)            

            speaker_id_i = self.clean_files[i].split("/")[-2] 
            filename_i = self.clean_files[i].split("/")[-1].replace(".wav", "")

            video_path_i = self.video_path.format(subset=self.subset, speaker_id=speaker_id_i, filename=filename_i)
                
            x, v = get_mouthroi_audio_pair_load_raw_video(mouthroi_path = video_path_i, 
                                                          audio_x= x, 
                                                          window=target_len, 
                                                          num_of_mouthroi_frames=target_video_nbframe,
                                                          audio_sampling_rate=16000, fps=self.fps,                                                           
                                                          impose_batch_1=self.impose_batch_1,
                                                          video_feature_type = self.video_feature_type,
                                                          video_size = self.video_size)
            
                        
            # if (v is None or x is None ):
            #     return self.__getitem__(randrange(len(self.clean_files)))   

            # print(f"####### x shape : {x.shape} \n####### v shape : {v.shape} ")

            while (v is None or x is None ):
                i = randrange(len(self.clean_files))

                x, fs = load(self.clean_files[i])    
                assert fs == 16000 
                current_len = x.size(-1)  

                speaker_id_i = self.clean_files[i].split("/")[-2] 
                filename_i = self.clean_files[i].split("/")[-1].replace(".wav", "")
                video_path_i = self.video_path.format(subset=self.subset, speaker_id=speaker_id_i, filename=filename_i)
                    
                x, v = get_mouthroi_audio_pair_load_raw_video(mouthroi_path = video_path_i, 
                                                            audio_x= x, 
                                                            window=target_len, 
                                                            num_of_mouthroi_frames=target_video_nbframe,
                                                            audio_sampling_rate=16000, fps=self.fps,                                                           
                                                            impose_batch_1=self.impose_batch_1,
                                                            video_feature_type = self.video_feature_type,
                                                            video_size = self.video_size)              

            if self.normalize == "clean":
                normfac = x.abs().max()
            elif self.normalize == "not":
                normfac = 1.0
            x = x / normfac

            X = torch.stft(x, **self.stft_kwargs)

            X = self.spec_transform(X)  # normalises if set to transform for normalising

            if self.spectogram_learning:
                X = X.abs().pow(2)
                        
            if not self.audio_only : 
                return X, v
            else :
                return X
                                      

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files) / 200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--base_dir",
            type=str,
            default="dummy",
            help="The base directory of the dataset. If `default` format then should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=("default", "tcd-timit", "dns", "wsj0"),
            required=True,
            help="Read file paths according to file naming format.",
        )
        parser.add_argument(
            "--batch_size", type=int, default=8, help="The batch size. 8 by default."
        )
        parser.add_argument(
            "--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default."
        )  # to assure 256 freq bins
        parser.add_argument(
            "--hop_length",
            type=int,
            default=128,
            help="Window hop length. 128 by default.",
        )
        parser.add_argument(
            "--num_frames",
            type=int,
            default=256,
            help="Number of frames for the dataset. 256 by default.",
        )
        parser.add_argument(
            "--window",
            type=str,
            choices=("sqrthann", "hann"),
            default="hann",
            help="The window function to use for the STFT. 'hann' by default.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers to use for DataLoaders. 4 by default.",
        )
        parser.add_argument(
            "--dummy",
            action="store_true",
            help="Use reduced dummy dataset for prototyping.",
        )
        parser.add_argument(
            "--spec_factor",
            type=float,
            default=0.15,
            help="Factor to multiply complex STFT coefficients by. 0.15 by default.",
        )
        parser.add_argument(
            "--spec_abs_exponent",
            type=float,
            default=0.5,
            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.",
        )
        parser.add_argument(
            "--normalize",
            type=str,
            choices=("clean", "not"),
            default="clean",
            help="Normalize the input waveforms by the clean signal, or not at all.",
        )
        parser.add_argument(
            "--transform_type",
            type=str,
            choices=("exponent", "log", "none", "normalise"),
            default="exponent",
            help="Spectogram transformation for input representation.",
        )
        parser.add_argument(
            "--spectogram_learning",
            action="store_true",
            help="Train model on spectograms and use mixture signal for phase approximations",
        )

        parser.add_argument(
            "--impose_batch_1",
            action="store_true",
            help="Impose the batch size to be 1",
        )   

        parser.add_argument(
            "--video_size",            
            default=88,            
            help="height and width of video frame",
        )         

        return parser

    def __init__(
        self,
        base_dir,
        audio_only,        
        vfeat_processing_order,
        format="wsj0",
        batch_size=8,
        n_fft=510,
        hop_length=128,
        num_frames=256,
        window="hann",
        num_workers=4,
        dummy=False,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        gpu=True,
        normalize="clean",
        transform_type="exponent",
        spectogram_learning=False,               
        impose_batch_1 = False,
        
        **kwargs,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format

        if impose_batch_1:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.spectogram_learning = spectogram_learning

        self.audio_only = audio_only
        self.vfeat_processing_order = vfeat_processing_order
        self.impose_batch_1 = impose_batch_1

        self.kwargs = kwargs

        
    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs,
            num_frames=self.num_frames,
            spec_transform=self.spec_fwd,
            ispecs_kwargs = self.istft_kwargs,
            **self.kwargs,
        )
        
        if stage == "fit" or stage is None:
            self.train_set = Specs(
                data_dir=self.base_dir,
                subset="train",
                dummy=self.dummy,
                shuffle_spec=True,
                format=self.format,
                normalize=self.normalize,
                spectogram_learning=self.spectogram_learning,
                audio_only = self.audio_only ,
                vfeat_processing_order = self.vfeat_processing_order,
                impose_batch_1 = self.impose_batch_1,
               
                **specs_kwargs, ## specs_kwargs already contained video_feature_type
            )
            self.valid_set = Specs(
                data_dir=self.base_dir,
                subset="valid",
                dummy=self.dummy,
                shuffle_spec=False,
                format=self.format,
                normalize=self.normalize,
                spectogram_learning=self.spectogram_learning,
                audio_only = self.audio_only ,
                vfeat_processing_order = self.vfeat_processing_order ,
                impose_batch_1 = self.impose_batch_1,
                
                **specs_kwargs,
            )
        if stage == "test" or stage is None:
            self.test_set = Specs(
                data_dir=self.base_dir,
                subset="test",
                dummy=self.dummy,
                shuffle_spec=False,
                format=self.format,
                normalize=self.normalize,
                return_time=True,
                spectogram_learning=self.spectogram_learning,
                audio_only = self.audio_only ,
                vfeat_processing_order = self.vfeat_processing_order,
                impose_batch_1 = self.impose_batch_1,
                               
                **specs_kwargs,
            )


    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "normalise":
            spec = spec / spec.abs().max()
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "normalise":
            spec = spec
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(
            spec, **{**self.istft_kwargs, "window": window, "length": length}
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
        )
