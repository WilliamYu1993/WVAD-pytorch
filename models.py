import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FCN_WVAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcnwvad = nn.Sequential(
            nn.Conv1d(1, 30, 55, stride=1, padding=27),            
            nn.LeakyReLU(0.3),
            nn.Conv1d(30, 15, 512, stride=1, padding=256),     
            nn.LeakyReLU(0.3),
            nn.Conv1d(15, 2, 512, stride=1, padding=256),
            nn.BatchNorm1d(num_features=2, momentum=0.01, affine= False, track_running_stats=False),
            nn.LeakyReLU(0.3),
            nn.Conv1d(2, 2, 512, stride=256, padding=128),
            nn.Sigmoid(),
            nn.Conv1d(2, 2, 55, stride=1, padding=27),
            nn.Sigmoid(),
            nn.Conv1d(2, 2, 15, stride=1, padding=7),
            nn.Sigmoid(),
            nn.Conv1d(2, 2, 5, stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x, layer="all"):
        if layer=="conv_feat":
            return self.fcnwvad[:5](x)
        else:
            return self.fcnwvad(x)

if __name__ == '__main__':
    inp = torch.randn(4, 1234)
    net = EncoderSupervised()
    net.train()
    out = net(inp)
    loss = F.mse_loss(out, torch.randn_like(out))
    loss.backward()
    for p in net.parameters():
        if p.requires_grad:
            print(p.grad.data)

