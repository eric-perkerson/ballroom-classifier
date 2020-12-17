import os
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
import librosa as li
import numpy as np
# from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"
data_dir = '/Users/eric/BMC/data'
sr = 22050 # Sample rate
duration = 10 # In seconds
input_size = sr*duration
n_epochs = 100
batch_size = 16

def read_label(file):
    return file[:-9]

def load_audio_data(directory, sr):
    files = os.listdir(directory)[1:] # To omit .DS_store
    data = []
    labels = []
    for file in files:
        y, sr = li.load(os.path.join(directory, file), sr=sr)
        data.append(y)
        labels.append(read_label(file))
    return data, labels

def sample_audio_data(data, sr, duration, offset_start=30, offset_end=30):
    sample_start = np.random.randint(offset_start*sr, len(data) - offset_end*sr)
    sample = data[sample_start:sample_start + duration*sr]
    return sample

def get_dataset(x, y):
    return TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1]))


# Load data
train_data, labels = load_audio_data(os.path.join(data_dir, 'training_data'), sr)
# sample = sample_audio_data(train_data[0], sr, duration)
# X = np.vstack([sample_audio_data(train_data[i], sr, duration) for i in range(len(train_data))])
num_random_samples = 20
X_all = np.vstack(
            [np.array([sample_audio_data(train_data[i], sr, duration) for i in range(len(train_data))]) 
                for j in range(num_random_samples)]
            )
y = np.array(list(map(lambda x: 1 if x == 'viennese_waltz' else 0, labels)))
y_all =  np.concatenate([y for j in range(num_random_samples)])
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_all).float(), torch.from_numpy(y_all).float()), batch_size=batch_size, shuffle=True)

# Define model
# m = nn.Conv1d(1, h1, k, padding=(k-1)//2, stride=1, dilation=1)
model = TCNModel(num_channels=[20]*2, kernel_size=3, dropout=0.25)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)

# Define loss
loss = nn.CrossEntropyLoss()

# Train model

for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_x, batch_y = data

        optimizer.zero_grad()
        # indices = permutation[i:i + batch_size]
        # batch_x, batch_y = X[indices, :], y[indices]
        outputs = model.forward(batch_x)
        loss = lossfunction(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0