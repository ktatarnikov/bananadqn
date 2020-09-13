from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size: int, seed: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=256, kernel_size=(1,3,3), stride=(1,3,3)), \
            nn.BatchNorm3d(256), \
            nn.ReLU())

        self.layer2 = nn.Sequential( \
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,3,3)), \
            nn.BatchNorm3d(512), \
            nn.ReLU())

        self.layer3 = nn.Sequential( \
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(4, 3, 3), stride=(1, 3, 3)), \
            nn.BatchNorm3d(512), \
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=4608, out_features=1024), nn.ReLU())
        self.layer6 = nn.Linear(in_features=1024, out_features=action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        x = self.layer6(x)
        return x
