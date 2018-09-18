import torch
import torch.nn as nn

class TwoLayerFF(nn.Module):


    def __init__(self, structure, activF):
        super(TwoLayerFF, self).__init__()

        ni = structure[0]
        h_l_1 = structure[1]
        h_l_2 = structure[2]
        no = structure[-1]

        self.activF = activF

        self.fc1 = nn.Linear(ni, h_l_1)
        self.fc2 = nn.Linear(h_l_1, h_l_2)
        self.fc3 = nn.Linear(h_l_2, no)

    def forward(self, x):

        a1 = self.activF(self.fc1(x))
        a2 = self.activF(self.fc2(a1))
        x = self.fc3(a2)

        return x, [a1, a2]
