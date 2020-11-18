import torch.nn as nn
import torch.nn.functional as F


class LinearG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearG, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))


class LinearD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearD, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))