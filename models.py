# Moving model definition here to modularize model
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin

class MLPNet(nn.Module):
    def __init__(self, input_dim=100, hidden_layer_sizes=(128,), dropout=0.5):
        super().__init__()
        layers = []
        prev_size = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x) 
