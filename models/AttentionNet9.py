
import __init__
from __init__ import *

import Meta

shapes = [
    [12, 166,1],
    [16, 108,1],
    [24, 70, 1],
    [32, 46, 1],
    [48, 30, 1],
    [64, 20, 1],
    [128,13, 1],
    [128, 8, 1],
]

class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()

        self.linear9 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[7])),
            nn.PReLU(),
        )
        self.a9_ones = torch.ones(1, *shapes[7]).to(Meta.device)

        self.linear8 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[6])),
            nn.PReLU(),
        )
        self.a8_ones = torch.ones(1, *shapes[6]).to(Meta.device)

        self.linear7 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[5])),
            nn.PReLU(),
        )
        self.a7_ones = torch.ones(1, *shapes[5]).to(Meta.device)

        self.linear6 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[4])),
            nn.PReLU(),
        )
        self.a6_ones = torch.ones(1, *shapes[4]).to(Meta.device)

        self.linear5 = nn.Sequential(
            nn.Linear(256, np.prod(shapes[3])),
            nn.PReLU(),
        )
        self.a5_ones = torch.ones(1, *shapes[3]).to(Meta.device)

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

        a9 = self.linear9(x).view(-1, *shapes[7])
        a9_ones = torch.ones(a9.shape)
        a9 = a9*attend_rates[7] + self.a9_ones * (1-attend_rates[7])

        a8 = self.linear8(x).view(-1, *shapes[6])
        a8_ones = torch.ones(a8.shape)
        a8 = a8*attend_rates[6] + self.a8_ones * (1-attend_rates[6])

        a7 = self.linear7(x).view(-1, *shapes[5])
        a7_ones = torch.ones(a7.shape)
        a7 = a7*attend_rates[5] + self.a7_ones * (1-attend_rates[5])

        a6 = self.linear6(x).view(-1, *shapes[4])
        a6_ones = torch.ones(a6.shape)
        a6 = a6*attend_rates[4] + self.a6_ones * (1-attend_rates[4])

        a5 = self.linear5(x).view(-1, *shapes[3])
        a5_ones = torch.ones(a5.shape)
        a5 = a5*attend_rates[3] + self.a5_ones * (1-attend_rates[3])

        a4 = self.linear4(x).view(-1, *shapes[2])
        a4_ones = torch.ones(a4.shape)
        a4 = a4*attend_rates[2] + self.a4_ones * (1-attend_rates[2])

        a3 = self.linear3(x).view(-1, *shapes[1])
        a3_ones = torch.ones(a3.shape)
        a3 = a3*attend_rates[1] + self.a3_ones * (1-attend_rates[1])

        a2 = self.linear2(x).view(-1, *shapes[0])
        a2_ones = torch.ones(a2.shape)
        a2 = a2*attend_rates[0] + self.a2_ones * (1-attend_rates[0])

        return [a2, a3, a4, a5, a6, a7, a8, a9]
