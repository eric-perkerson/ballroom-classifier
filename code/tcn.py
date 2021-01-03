# Temporal Convolutional Network Module

class TemporalBlock(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_input_channels, n_output_channels, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_output_channels, n_output_channels, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.pad = nn.ConstantPad1d(((kernel_size-1) * dilation, 0), 0.)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_input_channels, n_output_channels, 1) if n_input_channels != n_output_channels else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        x = self.dropout(self.relu(self.conv1(self.pad(x))))
        x = self.dropout(self.relu(self.conv2(self.pad(x))))
        return self.relu(x + res)

class TemporalConvNet(nn.Module):
    def __init__(self, n_input_channels, hidden_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(hidden_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = n_input_channels if i == 0 else hidden_channels[i-1]
            layers += [TemporalBlock(in_channels, hidden_channels[i], kernel_size, stride=1, dilation=dilation, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, hidden_channels, n_classes, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(1, hidden_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_channels[-1], n_classes)

    def forward(self, x):
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1]))
