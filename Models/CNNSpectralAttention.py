# CNNSpectralAttention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Spectral Attention Module
class SpectralAttention(nn.Module):
    def __init__(self, num_bands=125):
        super(SpectralAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_bands, requires_grad=True))

    def forward(self, x):
        # Apply spectral attention weights to spectral bands
        attention = self.attention_weights.view(1, -1, 1, 1)
        x = x * attention
        return x


# CNN with Spectral Attention
class CNN_With_Spectral_Attention(nn.Module):
    def __init__(self):
        super(CNN_With_Spectral_Attention, self).__init__()
        self.spectral_attention = SpectralAttention(num_bands=125)
        self.conv1 = nn.Conv2d(125, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 classes: Healthy, Rust, Other

    def forward(self, x):
        # Apply spectral attention first
        x = self.spectral_attention(x)
        # CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



