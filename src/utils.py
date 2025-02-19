import torch
import os
from src.eval_metrics import compute_pesq, compute_sisdr, compute_stoi, energy_ratios
import librosa.display
import matplotlib.pyplot as plt
import abc

import pandas as pd
from tqdm import tqdm
from six.moves import cPickle as pickle 



def count_files_with_extension(folder_path, extension):
    count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item.lower().endswith(extension):
            count += 1
    return count


def show_spec(spectogram: list, titles=None, hop_length=128, sr=16000, savefile=None):
    """
    Print only spectograms.
    """

    fig = plt.figure(figsize=(5 * len(spectogram), 3), tight_layout=True)
    for i, specto in enumerate(spectogram):
        plt.subplot(int(f"1{len(spectogram)}{i+1}"))
        X_log = librosa.amplitude_to_db(specto.squeeze().abs().cpu().numpy())
        librosa.display.specshow(
            X_log,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="log",
        )
        if titles != None:
            plt.title(titles[i])

    if savefile != None:
        fig.savefig(savefile)
        plt.close(fig)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def collect_rtf_pickle_files(save_dir_rtf,dataset):

    ## if we got rtf of all evaluation set files (ie if the SE  of all test files is done), then gather them and make a csv file
    pkl_files = [f for f in os.listdir(save_dir_rtf) if f.endswith('.pkl')] 

    if len(pkl_files)==len(dataset):

        df_metrics = pd.DataFrame(columns=['speaker_id', 'file_name', 'noise_type', 'snr', 'enhanced', 'rtf'])
        
        for this_file in tqdm(pkl_files):
            this_file_path = os.path.join(save_dir_rtf, this_file)
            this_res = load_dict(this_file_path)
            
            df_metrics = pd.concat([df_metrics, pd.DataFrame.from_dict({'speaker_id': [this_res['speaker_id']], 'file_name': [this_res['file_name']],
                                            'noise_type': [this_res['noise_type']], 'snr': [this_res['snr']], 'enhanced': [this_res['enhanced']],
                                            'rtf': [this_res['rtf']] })], ignore_index=True)   

            # remove tmp file
            os.system(f"rm {this_file_path}")


        # Save the DataFrame to a CSV file
        df_metrics.to_csv(os.path.join(save_dir_rtf, 'rtf.csv'), index=False) 



def calc_metrics(x_ref, x_hat, n, sr=16000):
    """
    Returns dictionary of metrics.
    """

    metrix = {}

    metrix["pesq"] = compute_pesq(target=x_ref, enhanced=x_hat, sr=sr)
    metrix["si_sdr"] = compute_sisdr(reference=x_ref, estimate=x_hat)
    metrix["estoi"] = compute_stoi(target=x_ref, enhanced=x_hat, sr=sr)

    metrix["si_sir"] = energy_ratios(x_hat, x_ref, n)[1]
    metrix["si_sar"] = energy_ratios(x_hat, x_ref, n)[2]
    return metrix


class Scheduler(abc.ABC):
    """
    Copied from derevp
    """

    def __init__(self, N, eps, **kwargs):
        super().__init__()
        self.N = N
        self.eps = eps

    @abc.abstractmethod
    def timesteps(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass


class LinearScheduler(Scheduler):
    def timesteps(self):
        timesteps = torch.linspace(1.0, self.eps, self.N)
        return torch.cat([timesteps, torch.Tensor([0.0])])

    def copy(self):
        return LinearScheduler(N=self.N, eps=self.eps)


class KarrasScheduler(Scheduler):
    def __init__(self, N, eps, sigma_min=3e-2, sigma_max=1.0, rho=5, **kwargs):
        super().__init__(N, eps)
        self.sigma_min = sigma_min
        self.sigma_max = 1.0
        self.rho = rho

    def timesteps(self):
        lin_timesteps = torch.linspace(self.eps, 1.0, self.N)
        timesteps = (
            self.sigma_max ** (1 / self.rho)
            + lin_timesteps
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return torch.cat([timesteps, torch.Tensor([0.0])])

    def copy(self):
        return KarrasScheduler(
            N=self.N,
            eps=self.eps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )
