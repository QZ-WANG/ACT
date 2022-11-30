import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, SGConv, GATConv
from torch_geometric.nn import inits

from torch_geometric.utils import dropout_adj

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, output_type="logit", adj_dropout=0.0):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.drop_out = dropout
        # self.device = device
        self.output_type = output_type

        self.convs = torch.nn.ModuleList()

        self.normalise = False

        self.ebd_dim = out_channels

        self.adj_dropout = adj_dropout

        if self.num_layers == 1:
            print("Building single-layer GraphSage")
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            print("Building multi-layer GraphSage")
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=self.normalise))

            for _ in range(num_layers - 2):
                print("One hidden layer added.")
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=self.normalise))

            self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=self.normalise))

    def reset_parameters(self):
        print("Initialising")
        for conv in self.convs:
            inits.kaiming_uniform(conv.lin_l.weight, conv.lin_l.in_channels, a=math.sqrt(5))
            inits.kaiming_uniform(conv.lin_r.weight, conv.lin_r.in_channels, a=math.sqrt(5))
            inits.zeros(conv.lin_l.in_channels)
            inits.zeros(conv.lin_r.in_channels)
            # conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.

            if self.adj_dropout > 0:
                edge_index = dropout_adj(edge_index, p=self.adj_dropout, force_undirected=True, training=self.training)[0]

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_out, training=self.training)

        if self.output_type == "ebd":
            # x = F.relu(x)
            # x = F.dropout(x, p=self.drop_out, training=self.training)
            return x

        return x.log_softmax(dim=-1).float()


    # def inference(self, x_all, test_loader):
    #     total_edges = 0
    #     for i in range(self.num_layers):
    #         xs = []
    #         for batch_size, n_id, adj in test_loader:
    #             edge_index, _, size = adj.to(self.device)
    #             total_edges += edge_index.size(1)
    #             x = x_all[n_id].to(self.device)
    #             x_target = x[:size[1]]
    #             x = self.convs[i]((x, x_target), edge_index)
    #             if i != self.num_layers - 1:
    #                 x = F.relu(x)
    #             xs.append(x.cpu())
    #
    #         x_all = torch.cat(xs, dim=0)
    #
    #     return x_all


# class SGC(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SGC, self).__init__()
#         self.con1 = SGConv(in_channels, out_channels, K=1, cached=True)
#         self.num_layers = 1
#
#     def forward(self, x, adj, output_type="logit"):
#         if output_type not in ['ebd', 'logit']:
#             raise RuntimeError
#
#         x = self.con1(x, adj)
#         return F.log_softmax(x, dim=1) if output_type == "logit" else x

class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)

# A gat implementation using Pytorch Geometric
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        self.skip = None

        # First Layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, heads))

        # Final Layer
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads, concat=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.skip is not None:
            for skip in self.skips:
                skip.reset_parameters()


    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x
