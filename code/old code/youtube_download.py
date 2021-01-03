# Download .wav file using youtube url example:
# youtube-dl -a "Batch file.txt" -x --audio-format "wav" https://www.youtube.com/watch?v=q7DzC-y75-c

import os

os.chdir('data')
all_files = os.listdir()
text_files = [file for file in all_files if file[-4:] == '.txt']
for file in text_files:
    style = file[:-4]
    if not os.path.isdir(os.path.join('audio_data', style)):
        os.mkdir(os.path.join('audio_data', style))
    f = open(file)
    lines = f.readlines()
for line in lines:
    os.system("googler -x -w youtube.com --np '%s' | sed -n '2'p | sed 's/^[ ]*//' > tmp" % line.strip())
    g = open('tmp')
    url = g.read()
    g.close()
    os.system("youtube-dl -x --audio-format 'wav' -o 'audio_data/{0}/%(title)s.%(ext)s' {1}".format(style, url))

# Manual for playlists
style = 'tango'
url = 'https://www.youtube.com/playlist?list=PLa02XFyoGvTDkuBsWjCsay9ISx9Aam5w3'
print("youtube-dl -x --yes-playlist --playlist-start 1 --audio-format 'wav' -o 'audio_data/{0}/%(title)s.%(ext)s' {1}".format(style, url))

# Convert downloaded .wav files to mono and undersample to librosa default
import librosa as li
import soundfile as sf

audio_dir = '/Users/eric/BMC/data/audio_data'
styles = os.listdir(audio_dir)[1:]
for style in styles:
    songs = os.listdir(os.path.join(audio_dir, style))[1:]
    for song in songs:
        if song[-7] == '5':
            song_file = os.path.join(audio_dir, style, song)
            y, sr = li.load(song_file)
            sf.write(song_file, y, sr)

for song in songs:
    song_file = os.path.join(audio_dir, style, song)
    y, sr = li.load(song_file)
    print(song + ': ' + str(li.beat.tempo(y)[0]))