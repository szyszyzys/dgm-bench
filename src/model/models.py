#!/usr/bin/env python3
"""
Refactored models and training utilities for federated learning.

This script provides:
  - Extendable image models (LeNet, SimpleCNN) using adaptive pooling.
  - A TextCNN model for NLP tasks.
  - A model factory `get_model` to easily instantiate models by name.
  - A flexible local training function that accepts different optimizers and loss functions.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Type


# ======================================================================================
#  1. MODEL DEFINITIONS
# ======================================================================================

class LeNet(nn.Module):
    # --- FIX: Accept in_channels and num_classes as arguments ---
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Use the in_channels argument here
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # ... your forward pass logic ...
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)  # <-- Add Dropout layer
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # <-- Apply dropout before the final layer
        x = self.fc2(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int,
                 filter_sizes: List[int], num_class: int, dropout: float = 0.5,
                 padding_idx: int = 0):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])

        # Add BatchNorm for stability
        # self.bn = nn.BatchNorm1d(num_filters * len(filter_sizes))
        self.ln = nn.LayerNorm(num_filters * len(filter_sizes))
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        squeezed = [conv.squeeze(3) for conv in conved]
        pooled = [F.max_pool1d(sq, sq.shape[2]).squeeze(2) for sq in squeezed]

        cat = torch.cat(pooled, dim=1)
        # cat = self.bn(cat)  # ✅ Add BatchNorm here
        cat = self.ln(cat)
        cat = self.dropout(cat)

        return self.fc(cat)

