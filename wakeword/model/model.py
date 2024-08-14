import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRU(nn.Module):
    def __init__(self, num_classes):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.gru = nn.GRU(input_size=32 * 3, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)  # Swap dimensions to (batch, time, freq, channels)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # Flatten for GRU input
        x, _ = self.gru(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected layer
        x = self.fc(x)
        
        return x
