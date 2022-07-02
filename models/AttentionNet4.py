
import __init__
from __init__ import *

import Meta

shapes = [
    [64, 85, 1], # x2
    [128,28, 1], # x3
    [128,8, 1],  # x4
]

class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()

        self.linear4 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[2])),
            nn.PReLU(),
        )
        self.a4_ones = torch.ones(1, *shapes[2]).to(Meta.device)

        self.linear3 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[1])),
            nn.PReLU(),
        )
        self.a3_ones = torch.ones(1, *shapes[1]).to(Meta.device)

        self.linear2 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[0])),
            nn.PReLU(),
        )
        self.a2_ones = torch.ones(1, *shapes[0]).to(Meta.device)

    def forward(self, x): # mod 19.12.5
        attend_rates = Meta.model_meta['layer_attentions']

        a4 = self.linear4(x).view(-1, *shapes[2])
        a4_ones = torch.ones(a4.shape)
        a4 = a4*attend_rates[2] + self.a4_ones * (1-attend_rates[2])

        a3 = self.linear3(x).view(-1, *shapes[1])
        a3_ones = torch.ones(a3.shape)
        a3 = a3*attend_rates[1] + self.a3_ones * (1-attend_rates[1])

        a2 = self.linear2(x).view(-1, *shapes[0])
        a2_ones = torch.ones(a2.shape)
        a2 = a2*attend_rates[0] + self.a2_ones * (1-attend_rates[0])

        return [a2, a3, a4]
