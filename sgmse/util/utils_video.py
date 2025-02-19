import os
import numpy as np
import cv2
from random import randrange
import torch
import sys
from argparse import Namespace


##if fairseq folder is not in av_hubert
# sys.path.append('./sgmse/util/')
# from . fairseq import checkpoint_utils, options, tasks, utils ## this will work only if executing train.py in avdiff, but it won't work if you cd in smgse/util and execute utils_video

# from fairseq import checkpoint_utils, options, tasks, utils #### this work in both case : executing train.py in avdiff and cd in smgse/util and execute utils_video

##if fairseq folder is in av_hubert , the following
sys.path.append("./sgmse/util/av_hubert/")
from sgmse.util.av_hubert.fairseq import checkpoint_utils, options, tasks, utils

sys.path.append("./sgmse/util")
from sgmse.util.lipreading.utils import load_json
from sgmse.util.lipreading.utils import load_model
from sgmse.util.lipreading.model import Lipreading


def get_mouthroi_audio_pair(
    mouthroi,
    audio_x,
    window,
    num_of_mouthroi_frames,
    audio_sampling_rate,
    fps,
    audio_only,
    impose_batch_1,
):
    if not impose_batch_1:
        audio_start = randrange(
            0, audio_x.shape[1] - window + 1
        )  # torch.Size([1, n_samples])
        audio_x_sample = audio_x[:, audio_start : audio_start + window]
        if not audio_only:
            frame_index_start = int(
                round(audio_start / audio_sampling_rate * fps)
            )  ##FPS=30
            mouthroi = mouthroi[
                :, frame_index_start : (frame_index_start + num_of_mouthroi_frames)
            ]  # [4489,nframes]
        else:
            mouthroi = None
    else:
        audio_x_sample = audio_x
        if audio_only:
            mouthroi = None

    return mouthroi, audio_x_sample


def get_mouthroi_audio_pair_load_raw_video(
    mouthroi_path,
    audio_x,
    window,
    num_of_mouthroi_frames,
    audio_sampling_rate,
    fps,    
    impose_batch_1,
    video_feature_type,
    video_size,
):


    if not impose_batch_1:

        count_trials = 0 ; frames= np.array([]) ; upper_bound_start = audio_x.shape[1] - window + 1  #x size torch.Size([1, n_samples])
        
        ##!! Note TCD-TIMIT dataset has its all video which are 4-5 s long, so we can certainly find a cut 
        # that has 2.04s. If we don't find, we reduce the upper_bound_start a certain number of time. If still not finding, then we can skip or maybe we could have pad
        
        assert num_of_mouthroi_frames==51 ## 51 if fps=25, window = 2.04*fs, fs=16000

        while count_trials < 5 and frames.shape != (51, video_size, video_size):
        
            audio_start = randrange(0, upper_bound_start)  


            if audio_x.shape[1] > window:
                audio_x_sample = audio_x[:, audio_start : audio_start + window]
            
            else:
                audio_x_sample = None
                frames =  None
                break

                            
            vid_start = int((audio_start / audio_sampling_rate * fps))  

            cap = cv2.VideoCapture(mouthroi_path)
            if cap.isOpened():
                frames=[]
                for i in range(vid_start + num_of_mouthroi_frames):
                    ret, img = cap.read()
                    if i<vid_start:
                        continue

                    if ret:
                        # uncomment if the videos are not yet in the size (88,88) and in greyscale
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # img = cv2.resize(img, (video_size,video_size))                        
                        frames.append(img)

                frames = np.array(frames).astype(float)

                # print(f"########## frames shape {frames.shape}  ########## ")
                
                
            count_trials +=1
            upper_bound_start = upper_bound_start//2 +1
            

        #set frames to None if at the end of process we dont have the desired shape            
        if frames.shape !=  (51, video_size, video_size): 
            frames = None    

    
    else : ##we don't need to crop to make all elements in the batch to have same dimension, as  we have just 1 element in the batch
        audio_x_sample = audio_x
        
        while True:
            ret, img = cap.read()
            if not ret:
                break
            
            # uncomment if the videos are not yet in the size (88,88) and in greyscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(img, (video_size,video_size)) #img = cv2.resize(img, (88,88))
            frames.append(img)

        frames = np.array(frames).astype(float)
        assert len(frames.shape)==3  and  frames.shape[1]==frames.shape[2]==video_size


    if frames is not None :
        if video_feature_type in ["resnet", "avhubert"]:                
            frames = np.expand_dims(frames, axis=0)  #get (1, 51, 88, 88)   or  (1, 51, video_size, video_size) in general
        
        elif video_feature_type in ["flow_avse"]:
            pass              #keep (51, 88, 88) or  (1, 51, video_size, video_size) in general

        elif video_feature_type in ["raw_image"]:  
            frames = frames.reshape(frames.shape[0], frames.shape[1]*frames.shape[2])
            frames = np.swapaxes(frames,0,1) #get (88*88, 51)  or  (1, 51, video_size, video_size) in general
                      

        frames = torch.from_numpy(frames).type(torch.FloatTensor)
        

    return audio_x_sample, frames



def get_mouthroi_load_raw_video_test(
    mouthroi_path,
    audio_x,
    window,
    num_of_mouthroi_frames,
    audio_sampling_rate,
    fps,    
    impose_batch_1,
    video_feature_type,
    video_size,
):


    if not impose_batch_1:

        count_trials = 0 ; frames= np.array([]) ; upper_bound_start = audio_x.shape[1] - window + 1  #x size torch.Size([1, n_samples])
        
        ##!! Note TCD-TIMIT dataset has its all video which are 4-5 s long, so we can certainly find a cut 
        # that has 2.04s. If we don't find, we reduce the upper_bound_start a certain number of time. If still not finding, then we can skip or maybe we could have pad
        
        assert num_of_mouthroi_frames==51 ## 51 if fps=25, window = 2.04*fs, fs=16000

        while count_trials < 5 and frames.shape != (51, video_size, video_size):
        
            audio_start = randrange(0, upper_bound_start)  


            if audio_x.shape[1] > window:
                audio_x_sample = audio_x[:, audio_start : audio_start + window]
            
            else:
                audio_x_sample = None
                frames =  None
                break

                            
            vid_start = int((audio_start / audio_sampling_rate * fps))  

            cap = cv2.VideoCapture(mouthroi_path)
            if cap.isOpened():
                frames=[]
                for i in range(vid_start + num_of_mouthroi_frames):
                    ret, img = cap.read()
                    if i<vid_start:
                        continue

                    if ret:
                        # uncomment if the videos are not yet in the size (88,88) and in greyscale
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # img = cv2.resize(img, (video_size,video_size))                        
                        frames.append(img)

                frames = np.array(frames).astype(float)

                # print(f"########## frames shape {frames.shape}  ########## ")
                
                
            count_trials +=1
            upper_bound_start = upper_bound_start//2 +1
            

        #set frames to None if at the end of process we dont have the desired shape            
        if frames.shape !=  (51, video_size, video_size): 
            frames = None    

    
    else : ##we don't need to crop to make all elements in the batch to have same dimension, as  we have just 1 element in the batch
        audio_x_sample = audio_x
        
        while True:
            ret, img = cap.read()
            if not ret:
                break
            
            # uncomment if the videos are not yet in the size (88,88) and in greyscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(img, (video_size,video_size)) #img = cv2.resize(img, (88,88))
            frames.append(img)

        frames = np.array(frames).astype(float)
        assert len(frames.shape)==3  and  frames.shape[1]==frames.shape[2]==video_size


    if frames is not None :
        if video_feature_type in ["resnet", "avhubert"]:                
            frames = np.expand_dims(frames, axis=0)  #get (1, 51, 88, 88)   or  (1, 51, video_size, video_size) in general
        
        elif video_feature_type in ["flow_avse"]:
            pass              #keep (51, 88, 88) or  (1, 51, video_size, video_size) in general

        elif video_feature_type in ["raw_image"]:  
            frames = frames.reshape(frames.shape[0], frames.shape[1]*frames.shape[2])
            frames = np.swapaxes(frames,0,1) #get (88*88, 51)  or  ( 51, video_size, video_size) in general
                      

        frames = torch.from_numpy(frames).type(torch.FloatTensor)
        

    return audio_x_sample, frames



def videocap(path, video_size,  start_frame=0 ): 
    vid_start = int(start_frame/16000*25)
    cap = cv2.VideoCapture(path)
    vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fm_num= vidlength

    if cap.isOpened():
        frames=[]
        for i in range(vid_start+fm_num):
            ret, img = cap.read()
            if i<vid_start:
                continue

            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = cv2.resize(img, (video_size,video_size))
                frames.append(img)

            else:
                raise Exception(f"Empty frame in {path}, frame number {i} ") 
                            
                
        frames = np.array(frames)

        assert frames.shape == (fm_num, video_size, video_size), f"padding is set wrong {frames.shape}"        
        
        return frames # (fm_num, H, W)
    
    else:
        print(path, " is not opened…")
        return None


def prep_video(video_path, start_frame, video_size, video_feature_type):
    frames = videocap(video_path, video_size, start_frame)
    if frames is None:
        print(f"{video_path} is invalid!!! ")
        raise Exception(f"Cannot process {video_path}, it is None")
   

    if video_feature_type in ["resnet", "avhubert"]:                
        frames = np.expand_dims(frames, axis=0)  #get (1, 51, 88, 88)   or  (1, 51, video_size, video_size) in general
    
    elif video_feature_type in ["flow_avse"]:
        pass              #keep (51, 88, 88) or  (51, video_size, video_size) in general

    elif video_feature_type in ["raw_image"]:  
        frames = frames.reshape(frames.shape[0], frames.shape[1]*frames.shape[2])
        frames = np.swapaxes(frames,0,1) #get (88*88, 51)  or  (video_size*video_size, 51 ) in general
                    
    visualFeature = torch.from_numpy(frames).type(torch.FloatTensor)

    return visualFeature


def load_visual_data_for_enhancement(vfile_path, video_feature_type, vfeat_processing_order):        
                
    if vfeat_processing_order in ["cut_extract"]:            
        video_size_dict = {"avhubert":88,"resnet":88, "raw_image":88, "flow_avse":112}

        v = prep_video(video_path=vfile_path, 
                start_frame=0, video_size=video_size_dict[video_feature_type],
                video_feature_type=video_feature_type)     
    
        if video_feature_type in ["resnet", "avhubert"]: 
            nb_v_frame = v.shape[1]

        elif video_feature_type in ["flow_avse"]: 
            nb_v_frame= v.shape[0]

        elif video_feature_type in ["raw_image"]: 
            nb_v_frame= v.shape[1]

    return v, nb_v_frame  


def load_array(path=None, precise_device = False,ndarray=None, mode="avhubert"):
    """this is to treat visual data feature already provided by NTCD. In that case,
     the video of mouth roi that has the shape (4489,nb_brame)=(67*67,nb_brame). 
     we don't really use this function
    """

    if ndarray is not None and path == None:
        arr = ndarray  ##(4489,t)
    elif ndarray == None and path != None:
        arr = np.load(path)
    else:
        raise ValueError("Should not specify both path and ndarray")

    if mode == "avhubert":
        arr = np.transpose(arr.reshape(67, 67, -1), axes=(1, 0, 2))
        arr = torch.from_numpy(np.transpose(arr.reshape(67, 67, -1), axes=(2, 0, 1)))
        # arr = torch.FloatTensor(arr)

        ##set precise_device to True in test evaluation, and False in training
        if precise_device:
            arr = arr.type(torch.FloatTensor).cuda()  # TxHxW
        else :
            arr = arr.type(torch.FloatTensor)
        
        # plt.imshow(arr[0,...], cmap='gray')
        # plt.show()
        # print(arr.shape); print(arr.max())
        arr = arr.unsqueeze(dim=0)  # (1,T,H,W)

    if mode == "resnet":

        arr = arr.transpose().reshape((-1, 1, 67, 67, 1))

        arr = torch.from_numpy(arr).permute(1, -1, 0, 3, 2)  # BxCxTxHxW    

        if precise_device:
            arr = arr.type(torch.FloatTensor).cuda()
        else :
            arr = arr.type(torch.FloatTensor)
            
        arr = torch.squeeze(
            arr, dim=0
        )  ##squeeze batch dim because, the dataloader will also add the batch dim, so the dim is CxTxHxW

    return arr


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


def build_resnet_extractor(config_path, model_path):

    args_loaded = load_json(config_path)
    backbone_type = args_loaded["backbone_type"]
    width_mult = args_loaded["width_mult"]
    relu_type = args_loaded["relu_type"]
    tcn_options = {
        "num_layers": args_loaded["tcn_num_layers"],
        "kernel_size": args_loaded["tcn_kernel_size"],
        "dropout": args_loaded["tcn_dropout"],
        "dwpw": args_loaded["tcn_dwpw"],
        "width_mult": args_loaded["tcn_width_mult"],
    }

    vfeats = Lipreading(
        modality="video",
        num_classes=500,
        tcn_options=tcn_options,
        backbone_type=backbone_type,
        relu_type=relu_type,
        width_mult=width_mult,
        extract_feats=True,
    ).cuda()

    vfeats = load_model(model_path, vfeats, allow_size_mismatch=False)

    vfeats.cuda()
    vfeats.eval()

    return vfeats


def build_extractor(video_feature_type):
    """
    Args:
    video_feature_type : is expected to be resnet or avhubert
    """

    if video_feature_type == "resnet":
        model_config_path = "./sgmse/util/lipreading/data/lrw_resnet18_mstcn.json"
        ckpt_path = "./sgmse/util/lipreading/data/lrw_resnet18_mstcn_adamw_s3.pth.tar"
        feature_extractor = build_resnet_extractor(model_config_path, ckpt_path)

    elif video_feature_type == "avhubert":

        avhubert_ckpt_path = "./sgmse/util/av_hubert/avhubert/base_vox_433h.pt" #"../pretrained_model/finetune-model.pt"
        avhubert_user_dir = "./sgmse/util/av_hubert/avhubert"
        feature_extractor = build_avhubert_extractor(
            avhubert_ckpt_path, avhubert_user_dir
        )

    if video_feature_type in ["resnet", "avhubert"]:
        for param in feature_extractor.parameters():
            param.requires_grad = False

    return feature_extractor


def resample_video_numpy(video, target_num):
    ##from  https://github.com/msaadeghii/av-dkf/blob/master/dvae/utils/speech_dataset.py#L47C5-L55
    n, N = video.shape  # (4489, 129)
    ratio = N / target_num
    idx_lst = np.arange(target_num).astype(float)
    idx_lst *= ratio
    res = np.zeros((n, target_num))
    for i in range(target_num):
        res[:, i] = video[:, int(idx_lst[i])]
    return res


def resample_video(video, target_num):
    ## Adapted from  https://github.com/msaadeghii/av-dkf/blob/master/dvae/utils/speech_dataset.py#L47C5-L55

    n, N = video.shape  # for example (4489, 129)
    ratio = N / target_num
    idx_lst = torch.arange(
        target_num, dtype=torch.float64, device=video.device, requires_grad=False
    )
    idx_lst *= ratio
    res = torch.zeros((n, target_num)).to(video)
    # res = torch.zeros((n, target_num), device=video.device, requires_grad=False)
    for i in range(target_num):
        res[:, i] = video[:, int(idx_lst[i])]
    return res

