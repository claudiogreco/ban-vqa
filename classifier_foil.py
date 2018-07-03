"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import torch.nn as nn


class SimpleClassifierFoil(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SimpleClassifierFoil, self).__init__()
        layers = [
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
