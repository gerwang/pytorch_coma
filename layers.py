import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops
import torch.nn as nn

from utils import normal


class ChebConv_Coma(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Coma, self).__init__(in_channels, out_channels, K, normalization, bias)

    def reset_parameters(self):
        normal(self.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm, edge_weight=None):
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        x = x.transpose(0, 1)
        Tx_0 = x
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            Tx_1_transpose = Tx_1.transpose(0, 1)
            out = out + torch.matmul(Tx_1_transpose, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_2_transpose = Tx_2.transpose(0, 1)
            out = out + torch.matmul(Tx_2_transpose, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j


class Pool(MessagePassing):
    def __init__(self):
        super(Pool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat, dtype=None):
        x = x.transpose(0, 1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out.transpose(0, 1)

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j


class DenseChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cheb_order = K

        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal(self.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)

    def forward(self, x, adj):
        """
        siamese mean aggregation
        :param x: batch x V x in_C
        :param adj: V x V, may be sparse, float32!!
        :return: batch x V x out_C
        """
        d_vec = torch.sum(adj, dim=1)
        # D = torch.diag(d_vec)  # column wise add
        inv_sqrt_d = d_vec.pow(-1 / 2)
        inv_sqrt_D = torch.diag(inv_sqrt_d)
        L = inv_sqrt_D @ -adj @ inv_sqrt_D  # fixme: low efficiency TODO: not real D_sym?

        Tx_0 = x
        out = torch.matmul(x, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = L @ Tx_0
            out += torch.matmul(Tx_1, self.weight[1])

        for i in range(2, self.weight.size(0)):
            Tx_2 = 2 * L @ Tx_1 - Tx_0
            out += torch.matmul(Tx_2, self.weight[i])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out


# class DiffPool(nn.Module):
#     def __init__(self, in_vertices, out_vertices, channels, K, normalization=None, bias=True):
#         super(DiffPool, self).__init__()
#         self.in_vertices = in_vertices
#         self.out_vertices = out_vertices
#         self.channels = channels
#         self.K = K
#         self.data_conv = ChebConv(channels, channels, K, normalization=normalization, bias=bias)
#         self.pool_conv = ChebConv(channels, out_vertices, K, normalization=normalization, bias=bias)
#
#     def forward(self, x, adj):
#         z = self.data_conv()

from torch_cluster import graclus_cluster
