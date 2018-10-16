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




class ThreeLayerFF(nn.Module):


    def __init__(self, structure, activF):
        super(ThreeLayerFF, self).__init__()

        ni = structure[0]
        h_l_1 = structure[1]
        h_l_2 = structure[2]
        h_l_3 = structure[3]
        no = structure[-1]

        self.activF = activF

        self.fc1 = nn.Linear(ni, h_l_1)
        self.fc2 = nn.Linear(h_l_1, h_l_2)
        self.fc3 = nn.Linear(h_l_2, h_l_3)
        self.fc4 = nn.Linear(h_l_3, no)

    def forward(self, x):

        a1 = self.activF(self.fc1(x))
        a2 = self.activF(self.fc2(a1))
        a3 = self.activF(self.fc3(a2))
        x = self.fc4(a3)

        return x, [a1, a2, a3]


class FourLayerFF(nn.Module):


    def __init__(self, structure, activF):
        super(FourLayerFF, self).__init__()

        ni = structure[0]
        h_l_1 = structure[1]
        h_l_2 = structure[2]
        h_l_3 = structure[3]
        h_l_4 = structure[4]
        no = structure[-1]

        self.activF = activF

        self.fc1 = nn.Linear(ni, h_l_1)
        self.fc2 = nn.Linear(h_l_1, h_l_2)
        self.fc3 = nn.Linear(h_l_2, h_l_3)
        self.fc4 = nn.Linear(h_l_3, h_l_4)
        self.fc5 = nn.Linear(h_l_4, no)

    def forward(self, x):

        a1 = self.activF(self.fc1(x))
        a2 = self.activF(self.fc2(a1))
        a3 = self.activF(self.fc3(a2))
        a4 = self.activF(self.fc4(a3))
        x = self.fc5(a4)

        return x, [a1, a2, a3, a4]


class FiveLayerFF(nn.Module):


    def __init__(self, structure, activF):
        super(FiveLayerFF, self).__init__()

        ni = structure[0]
        h_l_1 = structure[1]
        h_l_2 = structure[2]
        h_l_3 = structure[3]
        h_l_4 = structure[4]
        h_l_5 = structure[5]
        no = structure[-1]

        self.activF = activF

        self.fc1 = nn.Linear(ni, h_l_1)
        self.fc2 = nn.Linear(h_l_1, h_l_2)
        self.fc3 = nn.Linear(h_l_2, h_l_3)
        self.fc4 = nn.Linear(h_l_3, h_l_4)
        self.fc5 = nn.Linear(h_l_4, h_l_5)
        self.fc6 = nn.Linear(h_l_5, no)

    def forward(self, x):

        a1 = self.activF(self.fc1(x))
        a2 = self.activF(self.fc2(a1))
        a3 = self.activF(self.fc3(a2))
        a4 = self.activF(self.fc4(a3))
        a5 = self.activF(self.fc5(a4))
        x = self.fc6(a5)

        return x, [a1, a2, a3, a4]
