import torch
import torch.nn as nn


class PINN(nn.Module):
    # Simple MLP. tanh because we need smooth derivatives for autograd.
    def __init__(self, in_dim=2, out_dim=1, hidden=50, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        # Xavier init. Default pytorch init was giving me bad convergence.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=-1)
        return self.net(x)
