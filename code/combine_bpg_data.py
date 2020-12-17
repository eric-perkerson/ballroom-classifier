import numpy as np
import pandas as pd
import re
import os

def strip_s(str):
    return int(str[:-1])

os.chdir('/Users/eric/BMC/data/BPG Spotify Playlists')
all_files = os.listdir()[1:]
styles = [file[:-8] for file in all_files if file[-4:] == '.csv']

dfs = []
for style in styles:
    print(style)
    if style != 'hustle':
        df = pd.read_csv('/Users/eric/BMC/data/BPG Spotify Playlists/{0}_bpg.csv'.format(style))
    else:
        df = pd.read_csv('/Users/eric/BMC/data/BPG Spotify Playlists/{0}_bpg.csv'.format(style), sep=';')
    df = df.drop('addedBy', 1)
    df['style'] = pd.Series(style, index=df.index)
    dfs.append(df[['style', 'title', 'artist', 'album', 'isrc', 'addedDate', 'duration', 'url']])

result = pd.concat(dfs, ignore_index=False)
result['duration'] = list(map(strip_s, result['duration']))
result = result.drop('addedDate', axis=1)

os.chdir('/Users/eric/BMC/data/')
result.to_csv('bpg_playlist_data.csv')