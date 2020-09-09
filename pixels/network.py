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
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(1,3,3), stride=(1,3,3)), \
            nn.BatchNorm3d(512), \
            nn.ReLU())

        self.layer3 = nn.Sequential( \
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(4, 3, 3), stride=(1, 3, 3)), \
            nn.BatchNorm3d(512), \
            nn.ReLU())

        print("state_size: ", state_size)
        linear_input = self.conv_layers_shape(state_size)
        print("linear_input: ", linear_input)
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=linear_input, out_features=1024), nn.ReLU())
        # self.layer5 = nn.Dropout2d(0.25)
        self.layer6 = nn.Linear(in_features=1024, out_features=action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # print("state: ", state.size())
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = self.layer6(x)
        return x

    def conv_layers_shape(self, state_size) -> int:
        x = torch.rand(state_size)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print("x.size ", x.size())
        print("x.view(x.size(0), -1) ", x.view(x.size(0), -1).size())
        return x.view(x.size(0), -1).size(1)
        # print("x.data.view(1, -1).size(1) ", x.data.view(1, -1).size(1))
        # return x.data.view(1, -1).size(1)
