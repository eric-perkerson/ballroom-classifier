import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import librosa as li
import numpy as np
import tcn
from torch.utils.data import DataLoader, TensorDataset

os.chdir('/Users/eric/BMC/code')
DEVICE = "cpu"
DATA_DIR = '/Users/eric/BMC/data'
SR = 22050 # Sample rate
DURATION = 10 # In seconds
N_EPOCHS = 10
BATCH_SIZE = 16
num_random_samples = 20

CLASS_NAMES = ['waltz', 'viennese_waltz']
N_CLASSES = len(CLASS_NAMES)
CLASS_DICT = {CLASS_NAMES[i] : i for i in range(0, N_CLASSES)}

index_to_label = {v: k for k, v in CLASS_DICT.items()}

def read_label(file):
    return file[:-9]

def load_audio_data(directory, sr):
    files = os.listdir(directory)[1:] # To omit .DS_store
    data = []
    labels = []
    for file in files:
        y, sr = li.load(os.path.join(directory, file), sr=SR)
        data.append(y)
        labels.append(read_label(file))
    return data, labels

def sample_audio_data(data, offset_start=30, offset_end=30):
    """Randomly samples an audio segment of length SR*DURATION from the data."""
    sample_start = np.random.randint(offset_start*SR, len(data) - offset_end*SR)
    sample = data[sample_start:sample_start + SR*DURATION]
    return sample

def build_batch_loader(num_random_samples, data, labels):
    X_all = np.vstack(
            [np.array([sample_audio_data(data[i]) 
                for i in range(len(data))]) 
                for j in range(num_random_samples)]
            )
    y = np.array(list(map(lambda x: CLASS_DICT[x], labels)))
    y_all =  np.concatenate([y for j in range(num_random_samples)])
    loader = DataLoader(TensorDataset(torch.from_numpy(X_all).float(), 
                torch.from_numpy(y_all).long()), batch_size=BATCH_SIZE, shuffle=True)
    return loader

# Load train and test data
data_train, labels_train = load_audio_data(os.path.join(DATA_DIR, 'train_data'), SR)
data_test, labels_test = load_audio_data(os.path.join(DATA_DIR, 'test_data'), SR)
y_train = np.array([CLASS_DICT[label] for label in labels_train])
y_test = np.array([CLASS_DICT[label] for label in labels_test])
train_loader = build_batch_loader(num_random_samples, data_train, labels_train)

# model = tcn.TCNModel([20]*10, N_CLASSES)
model = nn.Linear(1, N_CLASSES)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)
lossfunction = nn.CrossEntropyLoss()

test_loader = build_batch_loader(num_random_samples, data_test, labels_test)
list(test_loader)
X_test = torch.Tensor(np.vstack([data_test[i][40*SR:(40 + DURATION)*SR] for i in range(len(data_test))]))

for epoch in range(N_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_x, batch_y = data
        # batch_x_np = batch_x.detach().numpy()
        # tempos = [li.beat.tempo(audio)[0] for audio in batch_x_np]
        # batch_tempos = torch.Tensor(tempos)
        optimizer.zero_grad()
        outputs = model.forward(batch_x.unsqueeze(1))
        loss = lossfunction(outputs.squeeze(1), batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5 == 4:    # print every 10 mini-batches
            y_test_pred = np.argmax(model(X_test.unsqueeze(1)).detach().numpy(), axis=1).flatten()
            test_acc = np.sum(1 - np.abs(y_test - y_test_pred))/len(y_test)
            print("Test acc.: {0}".format(test_acc))
            print('[%3d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

torch.save(model, "trained_model")

# Test accuracy

# y_test_pred = np.argmax(model(X_test.unsqueeze(1)).detach().numpy(), axis=2).flatten()
# test_acc = np.sum(1 - np.abs(y_test - y_test_pred))/len(y_test)
# print(test_acc)

# maxlen = max(list(map(len, CLASS_NAMES)))

# for i in range(len(y_test)):
#     print("Song {0}; Actual style: {1}; Predicted style: {2}".format(i, 
#         index_to_label[y_test_pred[i]], labels_test[i]))


# Simple test from scratch
# Define model
# inp = batch_x.unsqueeze(1)
# layer2(layer1(layer0(inp))).shape
# layer0 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, dilation=10)
# layer1 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=10, dilation=50)
# layer2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=10, dilation=100)
# layer3 = nn.Linear(in_features=219060, out_features=n_classes, bias=True)
# model = nn.Sequential(layer0, layer1, layer2, layer3)
