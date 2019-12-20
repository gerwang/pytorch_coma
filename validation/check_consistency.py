"""
Aimed the check the dense chebconv is consistent with sparse chebconv
"""

import torch
from layers import ChebConv_Coma, DenseChebConv
from psbody.mesh import Mesh
import mesh_operations
from main import scipy_to_torch_sparse
import numpy as np

in_channels = 16
out_channels = 32
batch_size = 16
K = 6

m = Mesh(filename='../template/template.obj')
sparse_adj = mesh_operations.get_vert_connectivity(m.v, m.f).tocoo()
sparse_adj = scipy_to_torch_sparse(sparse_adj)
dense_adj = sparse_adj.to_dense()
dense_adj[dense_adj != 0] = 1

n_vertex = m.v.shape[0]

x = torch.randn(batch_size, n_vertex, in_channels)

sparse_conv = ChebConv_Coma(in_channels, out_channels, K)
dense_conv = DenseChebConv(in_channels, out_channels, K)

# sparse_conv.weight = torch.nn.Parameter(torch.ones(K, in_channels, out_channels))
# sparse_conv.bias = torch.nn.Parameter(torch.zeros(out_channels))
dense_conv.weight = sparse_conv.weight
dense_conv.bias = sparse_conv.bias

edge_index, edge_norm = ChebConv_Coma.norm(sparse_adj._indices(), n_vertex)

y_sparse = sparse_conv(x, edge_index, edge_norm)
y_dense = dense_conv(x, dense_adj)

idx = torch.isclose(y_sparse, y_dense, rtol=1e-3, atol=1e-5).bitwise_not()

res = torch.cat([y_sparse[idx].unsqueeze(1), y_dense[idx].unsqueeze(1)], dim=1)
print(res.shape)
print(res)
