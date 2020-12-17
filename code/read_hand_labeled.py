import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

df = pd.read_csv('/Users/eric/BMC/data/song_data.csv')
df_ = df[df['bpm'].notna()][['style', 'bpm']]
tempo_dict = {style: list(df_[df_['style'] == style]['bpm']) for style in df_['style'].unique()}

all_tempos = df_['bpm']
a = np.min(all_tempos)
b = np.max(all_tempos)
eps = 20.
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