import torch.nn as nn
import torch.nn.functional as F
class Drebin(nn.Module):
    def __init__(self):
        super(Drebin, self).__init__()
        self.dens1 = nn.Linear(545334, 200)
        self.relu1 = nn.ReLU()
        self.dens2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.dens3 = nn.Linear(200, 200)
        self.relu3 = nn.ReLU()
        self.dens4 = nn.Linear(200, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dens1(x)
        x = self.relu1(x)
        x = self.dens2(x)
        x = self.relu2(x)
        x = self.dens3(x)
        x = self.relu3(x)
        x = self.dens4(x)
        return x
        
    def forward_act(self, x):
        act_list = []
        x = x.view(x.size(0), -1)
        x = self.dens1(x)
        act_list.append(x)
        x = self.relu1(x)
        act_list.append(x)
        x = self.dens2(x)
        act_list.append(x)
        x = self.relu2(x)
        act_list.append(x)
        x = self.dens3(x)
        act_list.append(x)
        x = self.relu3(x)
        act_list.append(x)
        x = self.dens4(x)
        act_list.append(x)
        return x,act_list