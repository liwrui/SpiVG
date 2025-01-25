import torch_geometric
import torch_geometric.nn as gnn
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class Block(gnn.MessagePassing):
    def __init__(self, in_channel, out_channel, k=2):
        super(Block, self).__init__()
        self.A = nn.Parameter(torch.randn(in_channel, k))
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_channel, out_channel)
        )

    def forward(self, x, e):
        return self.fc(self.propagate(e, x=x) + x)

    def message(self, x_i, x_j):
        sim = x_j[:, None, :] @ self.A @ self.A.transpose(0, 1) @ x_i[:, :, None]
        return sim[:, :, 0] * x_j  # 1, 1


class ThresBlock(gnn.MessagePassing):
    def __init__(self, in_channel, out_channel):
        super(ThresBlock, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1.]))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=False),
            nn.GELU(),
            nn.Linear(out_channel, out_channel, bias=False)
        )
        self.shortcut = nn.Identity() if in_channel == out_channel else nn.Linear(in_channel, out_channel, bias=False)

    def forward(self, x, e):
        return self.fc(x + self.propagate(e, x=x)) + self.shortcut(x)
        # return self.fc(torch.max(torch.stack([x, self.propagate(e, x=x)]), 0)[0]) + self.shortcut(x)

    def message(self, x_i, x_j):
        with torch.no_grad():
            sim = (x_i[:, None, :] @ x_j[:, :, None])[:, 0, 0] / (
                        torch.sum(x_i * x_i, 1) * torch.sum(x_j * x_j, 1)) ** 0.5
            sim = torch.abs(sim)
            # sim[sim < -0.2] = 0.2    # 0.2  0
            sim[sim > 0.1] = 0.1    # 0.5  0.8
            sim[sim < 0.1] = 0.
        return x_j * sim[:, None] * self.alpha

class SpikeFFN(nn.Module):
    def __init__(self, in_ch, out_ch, T):
        super(SpikeFFN, self).__init__()
        self.T = T
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
            nn.Linear(out_ch, out_ch, bias=False)
        )

    def forward(self, x):
        return self.fc(x).float()


class SpikeThresBlock(gnn.MessagePassing):
    def __init__(self, in_channel, out_channel, T=5):
        super(SpikeThresBlock, self).__init__()
        self.T = T
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.fc = SpikeFFN(in_channel, out_channel, T)
        self.shortcut = nn.Identity() if in_channel == out_channel else nn.Linear(in_channel, out_channel, bias=False)


    def forward(self, x, e):

        return self.fc(x + self.propagate(e, x=x)) + self.shortcut(x)

    def message(self, x_i, x_j):
        with torch.no_grad():
            sim = (x_i[:, None, :] @ x_j[:, :, None])[:, 0, 0] / (torch.sum(x_i * x_i, 1) * torch.sum(x_j * x_j, 1)) ** 0.5
            sim[sim < 0.2] = 0      # 0.2  0
            sim[sim > 0.2] = 0.5    # 0.2  0.5
            sim[sim > 0.5] = 0.8    # 0.5  0.8
        return x_j * sim[:, None] * self.alpha


if __name__ == '__main__':

    x = torch.tensor([
        [1, 2, 3],
        [0, 0, 0.]
    ])
    e = torch.tensor([
        [0, 1],
        [1, 1]
    ])

    model = Block(3, 2)

    print(model(x, e))
