import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import os
import pdb
import numpy as np

class VoiceBankDemandDataset_VAD(Dataset):
    def __init__(self, data_dir, train=True, hop_length=256):
        self.data_dir = data_dir
        self.train = train
        self.tier = 'trainset_28spk' if train else 'testset'
        self.hop_length = hop_length
        self.vadpath = './noisy-vctk-16k/'

        self.clean_root = os.path.join(
                self.data_dir, f'clean_{self.tier}_wav_16k')
        self.noisy_root = os.path.join(
                self.data_dir, f'noisy_{self.tier}_wav_16k')

        self.clean_path = self.get_path(self.clean_root)
        self.noisy_path = self.get_path(self.noisy_root)

    def get_path(self, root):
        paths = []
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.endswith('.wav'):
                    paths.append(os.path.join(r, f))
        return paths

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):

        S = self.clean_path[idx].split('/')
        vpath = os.path.join(self.vadpath, S[-2], S[-1].replace(".wav", ""))
        vad = np.loadtxt(vpath)
        noisy = torchaudio.load(self.noisy_path[idx])[0]
        noisy = self.normalize(noisy)
        length = noisy.size(-1)//256
        if self.train:
            noisy.squeeze_(0)
            start = torch.randint(0, length - 64 - 1, (1, ))
            end = start + 64
            vad = vad[start:end]
            noisy = noisy[start*256:end*256]
            return noisy.unsqueeze(0), vad

        else:
            noisy = self.padding(noisy)[:,:-512]
                        
        return noisy, vad, length
