from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size: int, action_size: int, seed: int):
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
            nn.Linear(in_features=state_size, out_features=128), nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128), nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128), nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=action_size))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
