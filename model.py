import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, GELU, LeakyReLU
from torch_geometric.nn import Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm, LayerNorm, GraphNorm, GCNConv, TransformerConv
from block import Block, ThresBlock, SpikeThresBlock
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class DilatedResidualLayer(Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = ReLU()
        self.dropout = Dropout()

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# This is for the iterative refinement (we refer to MSTCN++: https://github.com/sj-li/MS-TCN2)
class Refinement(Module):
    def __init__(self, final_dim, num_layers=10, interm_dim=64):
        super(Refinement, self).__init__()
        self.conv_1x1 = Conv1d(final_dim, interm_dim, kernel_size=1)
        self.layers = ModuleList([DilatedResidualLayer(2**i, interm_dim, interm_dim) for i in range(num_layers)])
        self.conv_out = Conv1d(interm_dim, final_dim, kernel_size=1)

    def forward(self, x):
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f)
        out = self.conv_out(f)
        return out


class SPELL(Module):
    def __init__(self, cfg, t_emb=True):
        super(SPELL, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = GraphNorm(channels[0])
        # self.batch01 = BatchNorm(channels[0])
        self.relu = GELU()
        # self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = ThresBlock(channels[0], channels[0])
        # self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = GraphNorm(channels[0])
        # self.batch11 = BatchNorm(channels[0])
        # self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.layer12 = ThresBlock(channels[0], channels[0])
        self.batch12 = GraphNorm(channels[0])
        # self.batch12 = BatchNorm(channels[0])
        # self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.layer13 = ThresBlock(channels[0], channels[0])
        self.batch13 = GraphNorm(channels[0])
        # self.batch13 = BatchNorm(channels[0])

        self.norm = cfg['norm']

        if num_att_heads > 0:
            self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
        else:
            # self.layer21 = SAGEConv(channels[0], channels[1])
            self.layer21 = ThresBlock(channels[0], channels[1])
            num_att_heads = 1
        self.batch21 = GraphNorm(channels[1]*num_att_heads)
        # self.batch21 = BatchNorm(channels[1] * num_att_heads)

        # self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        # self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        # self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        self.layer31 = ThresBlock(channels[1] * num_att_heads, final_dim)
        self.layer32 = ThresBlock(channels[1] * num_att_heads, final_dim)
        self.layer33 = ThresBlock(channels[1] * num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)

        self.t_emb = torch.nn.Parameter(torch.zeros(2000, channels[0])) if t_emb else None
        # self.t_emb = None


    def forward(self, x, edge_index, edge_attr, c=None):
        feature_dim = x.shape[1]

        if self.use_spf:
            x_visual = self.layer011(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf(c)), dim=1))
        else:
            x_visual = self.layer011(x[:, :feature_dim//self.num_modality])

        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
            x_audio = self.layer012(x[:, feature_dim//self.num_modality:])
            x = x_visual + x_audio

        if self.norm:
            x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        # Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        if self.norm:
            x1 = self.batch11(x1)
        x1 = self.relu(x1)
        if self.t_emb is not None:
            x1 = x1 + self.t_emb[:x1.shape[0]]
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        if self.norm:
            x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        if self.norm:
            x2 = self.batch12(x2)
        x2 = self.relu(x2)
        if self.t_emb is not None:
            x2 = x2 + self.t_emb[:x2.shape[0]]
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        if self.norm:
            x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        if self.norm:
            x3 = self.batch13(x3)
        x3 = self.relu(x3)
        if self.t_emb is not None:
            x3 = x3 + self.t_emb[:x3.shape[0]]
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        if self.norm:
            x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        # if self.use_ref:
        #     xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
        #     xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
        #     xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
        #     xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
        #     out = torch.stack((xr0, xr1, xr2, xr3), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return x1, x2, x3

class SGNN(Module):
    def __init__(self, cfg, t_emb=True):
        super(SGNN, self).__init__()
        self.spell = SPELL(cfg, t_emb)
        self.q = nn.Identity()
        # self.k = Linear(-1, 32)
        self.snn = nn.Sequential(
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
            # neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
        )
        # self.sigmas = nn.Parameter(torch.ones(5))
        self.sigma_fc = nn.Sequential(
            nn.Linear(3, 1, bias=False)
        )

        self.bias = nn.Parameter(torch.tensor([1., 1., 1., 0.]))

    def forward(self, x, edge_index, edge_attr, c=None, return_channels=False):
        o1, o2, o3 = self.spell(x, edge_index, edge_attr, c)
        q = self.q(x[1:])
        k = self.q(x[:-1])
        # sim = q[:, None, :] @ k[:, :, None]
        # sim = sim[:, 0]
        sim = torch.mean((q - k) ** 2, -1, keepdim=True)
        sim = self.snn(sim).float()
        # print(sim.shape)
        sim = nn.functional.pad(sim, (0, 0, 1, 0), 'constant', 0)
        # print(sim.shape)

        os = torch.concatenate([o1, o2, o3, sim], dim=-1)    # n, 4

        with torch.no_grad():
            do = os[1:] - os[:-1]
            ddo = do[1:] - do[:-1]
            mean_o = torch.mean(torch.abs(os), dim=0)  # 4
            mean_do = torch.mean(torch.abs(do), dim=0)
            mean_ddo = torch.mean(torch.abs(ddo), dim=0)
            fea = torch.stack([mean_o, mean_do, mean_ddo]).transpose(0, 1).detach()  # 4, 3

        sigmas = self.sigma_fc(fea)[:, 0] + self.bias   # 4
        sigmas = torch.exp(sigmas)
        sigmas = sigmas / (torch.sum(sigmas) + 1.)
        o = (os @ sigmas[:, None])

        reg = torch.sum((sigmas - torch.tensor([1., 1., 1., 0.]).to(x.device)) ** 2)

        if return_channels:
            return o, reg, o1.detach().cpu().numpy().tolist(), o2.detach().cpu().numpy().tolist(), o3.detach().cpu().numpy().tolist(), sim.detach().cpu().numpy().tolist()
        else:
            return o, reg






