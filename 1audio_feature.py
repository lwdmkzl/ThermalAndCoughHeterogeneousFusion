import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
import tensorflow
print(tensorflow.__version__)
import tensorflow.keras
print(tensorflow.keras.__version__)
from tensorflow.keras.applications import imagenet_utils

fenlei = 'cough nocough'.split()
file_name = 'data/pigCoughThermalv2/acoustic_data.csv'

header = 'filename chroma_stft chroma_cq rmse spectral_centroid spectral_bandwidth rolloff flatness o_env zero_crossing_rate mfcc1 mfcc2 mfcc3 mfcc4 mfcc5 mfcc6 mfcc7 mfcc8 mfcc9 mfcc10 mfcc11 mfcc12 mfcc13 mfcc14 mfcc15 mfcc16 mfcc17 mfcc18 mfcc19 mfcc20 label'
file = open(file_name, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header.split(' '))

for g in fenlei:

    for filename in os.listdir(f'./data/pigCoughThermal/audio/{g}'):

        songname = f'./data/pigCoughThermal/audio/{g}/{filename}'

        y, sr = librosa.load(songname,sr=None, mono=True, duration=30)
        


        zcr = librosa.feature.zero_crossing_rate(y)

        rmse = librosa.feature.rms(y=y)
        


        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        flatness = librosa.feature.spectral_flatness(y=y)

        o_env = librosa.onset.onset_strength(y=y, sr=sr)
        


        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        

        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(chroma_cq)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(flatness)} {np.mean(o_env)} {np.mean(zcr)}'

        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'


        file = open(file_name, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())