import os
import librosa as li
from librosa.display import waveplot, display
import pywt
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pandas as pd

audio_dir = '/Users/eric/BMC/data/audio_data'
df_ = pd.read_csv('/Users/eric/BMC/data/song_data.csv')
df = df_[df_['bpm'].notna() & ((df_['style'] == 'waltz') | (df_['style'] == 'viennese_waltz'))][['order', 'style', 'bpm']]
primary_keys = [(i, j) for i, j in zip(df['order'], df['style'])]
files = {(order, style) : os.path.join(audio_dir, style, style + '.' + '{:0>4}'.format(order)) + '.wav' for order, style in zip(df['order'], df['style'])}
for file in files.values():
    print(file)
    y, sr = li.load(file, offset=30, duration=30)
    S = li.feature.melspectrogram(y, sr=sr)
    fig, ax = plt.subplots()
    S_dB = li.power_to_db(S, ref=np.max)
    img = li.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


    