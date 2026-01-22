import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Net(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        heads: int,
        dropout: float,
        use_residual: bool,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            residual=use_residual,
        )
        self.norm1 = torch.nn.LayerNorm(hidden_channels * heads)
        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            1,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            residual=False,
        )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.elu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x.view(-1)
