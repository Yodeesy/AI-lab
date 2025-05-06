# ===== model.py =====
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)