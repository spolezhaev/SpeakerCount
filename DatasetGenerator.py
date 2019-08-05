from pathlib import Path
import librosa
from librosa.effects import trim
from preprocess_samples.split_active_voice import VoiceActivityDetection
import numpy
import scipy.io.wavfile as wf
from tqdm import tqdm
import os
import re
import random
from pydub import AudioSegment
import json

def split_active_voice(ms_threshold, filePath):
    wav = wf.read(filePath)
    ch = 1
    if len(wav[1].shape) > 1:
        ch = wav[1].shape[1]
    sr = wav[0]

    if len(wav[1].shape) > 1:
        c0 = wav[1][:,0]
    else:
        c0 = wav[1][:]

    # ONLY FOR MONO
    vad = VoiceActivityDetection(sr, ms_threshold, 1)
    vad.process(c0, filePath.parent / f'out_{filePath.stem}', filePath)


def generateDataset(path_to_archives = 'data/archives/'):
    p = Path(path_to_archives)
    for file in tqdm(list(p.glob("**/*.wav"))):
        if file.name.startswith('ru'):
            #print(file)
            y, sr = librosa.load(file)
            yt, _ = librosa.effects.trim(y, top_db=10)
            librosa.output.write_wav(file, yt, sr)
            split_active_voice(600, file)



def mixAudioTracks(path_to_data = 'data/archives', path_to_export = 'data/processed', n=1000):
    """
    Classes: 1, 2, 3, 3+ concurrent speakers
    """
    curr_class = 1
    counter = 0
    labels = {}
    files = list(Path(path_to_data).glob("**/*.wav"))
    max_count_per_cl = n // 4
    for _ in tqdm(range(n)):
        if curr_class < 4:
            num_of_speakers = curr_class
        else:
            num_of_speakers = random.randint(4, 9)

        mix = None
        for i in range(num_of_speakers):
            while True:
                file = random.choice(files)
                speech = AudioSegment.from_wav(file)
                if len(speech) >= 3000:
                    break
            if mix == None:
                mix = speech[:3000]
            else:
                mix = mix.overlay(speech[:3000])

        mix.export(Path(path_to_export) / f"{curr_class}_{counter+1}.wav", format='wav', parameters=["ac", "1"])
        labels[f"{curr_class}_{counter+1}"] = curr_class

        counter += 1
        if counter >= max_count_per_cl:
            if curr_class < 4:
                counter = 0
                curr_class += 1

    with open('data/processed/labels.json', 'w+') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)  

            
        
            

mixAudioTracks()

#generateDataset()
