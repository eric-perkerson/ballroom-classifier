# Estimate tempo using librosa
import os
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

tempo_dict = {}

audio_dir = '/Users/eric/BMC/data/audio_data/'
styles = os.listdir(audio_dir)[1:]
for style in styles:
    songs = os.listdir(os.path.join(audio_dir, style))[1:]
    if songs:
        tempo_dict[style] = []
        for song in songs:
            try:
                y, sr = li.load(os.path.join(audio_dir, style, song), offset=30, duration=30)
                tempo = li.beat.tempo(y)[0]
                tempo_dict[style].append(tempo)
                print(song + ': ' + str(tempo))
            except:
                print('Skipping ' + song)

all_tempos = np.array([item for sublist in tempo_dict.values() for item in sublist])
a = np.min(all_tempos)
b = np.max(all_tempos)
eps = 10.
xplt = np.linspace(a - eps, b + eps, num=200)
ys = {}
kde_dict = {}
for style in tempo_dict.keys():
    kde_dict[style] = KernelDensity(kernel='gaussian', bandwidth=5.)
    kde_dict[style].fit(np.array(tempo_dict[style]).reshape(-1, 1))
    ys[style] = np.exp(kde_dict[style].score_samples(xplt.reshape(-1, 1)))
    plt.plot(xplt, ys[style])

plt.legend(tempo_dict.keys())
plt.show()


