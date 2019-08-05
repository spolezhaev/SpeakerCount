import torch
from torchaudio_contrib import Melspectrogram
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import datetime
import math

from sklearn.metrics import accuracy_score

import librosa

import pandas as pd
import numpy as np

from pathlib import Path

import json

from net_config.audio import MelspectrogramStretch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from tqdm.autonotebook import tqdm

import logging


from visualization import WriterTensorboardX

from transforms import AudioTransforms

class SoundSet(Dataset):
    def __init__(self, transform=None, mode="train"):
        # setting directories for data
        self.mode = mode
        self.all_files = list(Path("data/processed").glob('**/*.wav'))
        if self.mode is "train":
            self.files = [file for i, file in enumerate(self.all_files) if i%5 !=0]
        elif self.mode is "test":
            self.files = [file for i, file in enumerate(self.all_files) if i%5 ==0]
        with open(str(Path("data/processed") / 'labels.json')) as f:
            self.classes = json.load(f)
        # dict for mapping class names into indices. can be obtained by 
        # {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
        #self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7,'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39,  'Writing': 11}
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        data, sr = librosa.load(str(filename))
        data = data.reshape(-1, 1)   
        
        if self.transform is not None:
            data = self.transform.apply(data)

        #if self.mode is "train":
        label = self.classes[filename.stem]
        return  data, sr, label

#         elif self.mode is "test":
#             return torch.from_numpy(data).float(), sr
