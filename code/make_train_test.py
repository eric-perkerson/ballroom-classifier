# Populates the training_data and test_data directories with symlinks to the selected files from the audio_data directory

import os
from numpy.random import choice
from shutil import rmtree

data_dir = '/Users/eric/BMC/data'
train_dir = os.path.join(data_dir, 'training_data')
test_dir = os.path.join(data_dir, 'test_data')

rmtree(train_dir)
rmtree(test_dir)
os.mkdir(train_dir)
os.mkdir(test_dir)

train_fraction = 0.75

os.chdir(data_dir)
# styles = os.listdir('audio_data')[1:]
# ['american_rumba', 'bachata', 'cha_cha', 'foxtrot', 'hustle', 'international_rumba', 'jive', 'merengue', 'paso_doble', 'quickstep', 'salsa', 'samba', 'tango', 'viennese_waltz', 'waltz', 'west_coast_swing']
styles = ['viennese_waltz', 'waltz']
for style in styles:
    source_dir = os.path.join(data_dir, 'audio_data', style)
    style_files = os.listdir(source_dir)[1:]
    num_train = round(train_fraction*len(style_files))
    num_test = len(style_files) - num_train
    train_files = list(choice(style_files, num_train, replace=False))
    test_files = list(set(style_files) - set(train_files))
    for file in train_files:
        os.symlink(os.path.join(source_dir, file), os.path.join(train_dir, file))
    for file in test_files:
        os.symlink(os.path.join(source_dir, file), os.path.join(test_dir, file))