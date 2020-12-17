import os
import librosa as li
from librosa.display import waveplot
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

# Analyze the different in the tempo_true and tempo_est from li
os.chdir('/Users/eric/BMC/')
offset = 30
duration = 15
xplt = np.linspace(50, 250, num=200)

# styles = ['viennese_waltz', 'waltz']
# viennese_waltz_files = [os.path.join(audio_dir, 'viennese_waltz', x) for x in os.listdir(os.path.join(audio_dir, 'viennese_waltz'))[1:]]
# waltz_files = [os.path.join(audio_dir, 'waltz', x) for x in os.listdir(os.path.join(audio_dir, 'waltz'))[1:]]
# files =  viennese_waltz_files + waltz_files
tempos_est = {}
for order, style in primary_keys:
    y, sr = li.load(files[(order, style)], offset=offset, duration=duration)
    tempos_est[(order, style)] = li.beat.tempo(y)[0]

kde_est = KernelDensity(bandwidth=3.)
kde_est.fit(np.array(list(tempos_est.values())).reshape(-1, 1))
y_est = np.exp(kde_est.score_samples(xplt.reshape(-1, 1)))

# Show the true tempos using the hand-labeled tempos in song_data.csv
tempos_true = {(df.iloc[i]['order'], df.iloc[i]['style']) : df.iloc[i]['bpm'] for i in range(len(df))}
kde_true = KernelDensity(bandwidth=3.)
kde_true.fit(np.array(list(tempos_true.values())).reshape(-1, 1))
y_true = np.exp(kde_true.score_samples(xplt.reshape(-1, 1)))

plt.plot(xplt, y_true)
plt.plot(xplt, y_est)
plt.legend(['true', 'est'])
plt.show()

for order, style in primary_keys:
    print(str(order) + ' ' + style)

# View the beats estimated by li
# for order, style in primary_keys:
order, style = primary_keys[0]
file = files[(order, style)]
print(tempos_true[(order, style)])
hop_length = 512
offset = 30
duration = 15
y, sr = li.load(files[(order, style)], offset=offset, duration=duration)
onset_env = li.onset.onset_strength(y, sr=sr, aggregate=np.median, hop_length=hop_length)
tempogram = li.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
times = li.times_like(onset_env, sr=sr, hop_length=hop_length)
tempo, beats = li.beat.beat_track(onset_envelope=onset_env, sr=sr)
fig, ax = plt.subplots(nrows=3, sharex=True)
waveplot(y, sr=sr, ax=ax[0])
ax[0].vlines(times[beats], -.5, .5, alpha=0.5, color='r', linestyle='--')
ax[1].plot(times, onset_env)
ax[1].vlines(times[beats], 0, 6, alpha=0.5, color='r', linestyle='--')
li.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma', ax=ax[2])
plt.show()

def argmax(M):
    m, n = M.shape
    linear_index = np.argmax(M)
    x, y = divmod(linear_index, m)
    return x, y

test = np.array([[1, 2, 3], [4,100, 6], [7,8,9]])
argmax(test)

x, y = argmax(tempogram[16:, :])
x += 16

avg_tempogram = np.mean(tempogram, axis=1)
plt.plot(avg_tempogram)
plt.show()



# Compute local onset autocorrelation
order, style = primary_keys[2]
y, sr = li.load(files[(order, style)], offset=30, duration=15)
hop_length = 2**10
oenv = li.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
tempogram = li.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
# Compute global onset autocorrelation
ac_global = li.autocorrelate(oenv, max_size=tempogram.shape[0])
ac_global = li.util.normalize(ac_global)
# Estimate the global tempo for display purposes
tempo = li.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
times = li.times_like(oenv, sr=sr, hop_length=hop_length)
ax[0].plot(times, oenv, label='Onset strength')
ax[0].label_outer()
ax[0].legend(frameon=True)
li.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma', ax=ax[1])
ax[1].axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
ax[1].legend(loc='upper right')
ax[1].set(title='Tempogram')
x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr, num=tempogram.shape[0])
ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
ax[2].set(xlabel='Lag (seconds)')
ax[2].legend(frameon=True)
freqs = li.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1), label='Mean local autocorrelation', basex=2)
ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75, label='Global autocorrelation', basex=2)
ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8, label='Estimated tempo={:g}'.format(tempo))
ax[3].legend(frameon=True)
ax[3].set(xlabel='BPM')
ax[3].grid(True)

plt.show()

def one_pole_filter(x, alpha=.99):
    n = len(x)
    y = np.zeros(n - 1)
    y[0] = x[0]
    for i in range(1, n-1):
        y[i] = (1 - alpha)*x[i] + alpha*y[i - 1]
    return y

def part(x, i):
    if i >= 0 and i <= len(x):
        return x[i]
    else:
        return np.nan

def lag_op(x, lag):
    return np.array([part(x, i - lag) for i in range(len(x))])

def autocorrelation(x, maxlag):
    return np.array([np.nanmean(x * lag_op(x, lag)) for lag in range(maxlag)])

def subsample(x, rate):
    return [x[rate*i] for i in range(len(x) // rate)]

y0 = np.abs(y)
y1 = one_pole_filter(y0)
y2 = subsample(y1, 16)
y3 = y2 - np.mean(y2)
plt.plot(y3); plt.show()

samples 1/(tempo_true/60/sr)

# Wavelet plot
# cA, cD = pywt.dwt(y, 'db4')
cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(y, 'db4', level=4)
waveplot(cD4)
waveplot(cD3)
waveplot(cD2)
waveplot(cD1)
# sf.write('cD4.wav', cD4, sr)    
plt.show()
