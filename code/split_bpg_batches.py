import numpy as np
import pandas as pd
import re
import librosa as li
import os
import soundfile as sf

df = pd.read_csv('/Users/eric/BMC/data/bpg_playlist_data.csv')

def parse_wav_file_name(wav_file):
    # wav_file should be in the format `style`.`start`.`end`.wav where the latter are padded to 4 digits
    style = wav_file[:-14]
    start = int(wav_file[-13:-9])
    end = int(wav_file[-8:-4])
    return style, start, end

def get_subdataframe(style, start, end):
    return df[(df['style'] == style) & (df['order'] >= start) & (df['order'] <= end)]

def get_duration(style, start, end):
    df_sub = get_subdataframe(style, start, end)
    return np.sum(df_sub['duration'])

def display_duration(style, start, end):
    total_duration = get_duration(style, start, end)
    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total duration is {0} hours, {1} minutes, and {2} seconds".format(hours, minutes, seconds))

# styles = list(df['style'].unique())
batch_dir = '/Users/eric/BMC/data/batch_data/'
audio_dir = '/Users/eric/BMC/data/audio_data/'
batch = os.listdir(batch_dir)[1:]
for batch_file in batch:
    style, start, end = parse_wav_file_name(batch_file)
    df_ = get_subdataframe(style, start, end)
    cs = list(np.cumsum(df_['duration']))
    cs0 = [0] + cs
    for i in range(len(cs)):
        y, sr = li.load(os.path.join(batch_dir, batch_file), offset=cs0[i], duration=cs0[i+1]-cs0[i]) # Use whole song for estimate
        # y, sr = li.load(os.path.join('/Users/eric/BMC/data/10s data', file), offset=cs0[i] + 30, duration=cs0[i] + 60)
        file_name = style + '.' + '{:0>4}'.format(i + start) + '.wav'
        full_file = os.path.join(audio_dir, style, file_name)
        sf.write(full_file, y, sr)

