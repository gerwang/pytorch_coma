import torch
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling
# from topkpool import TopKPooling

from layers import ChebConv_Coma, Pool

class Coma(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(Coma, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        self.filters.insert(0, dataset.num_features)  # To get initial features per node
        self.downsampling_factors = config['downsampling_factors']
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.cheb = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i-1], self.filters[-i-2], self.K[i])
                                             for i in range(len(self.filters)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.down_pool = torch.nn.ModuleList([TopKPooling(self.filters[i+1], 1/self.downsampling_factors[i])
                                         for i in range(len(self.downsampling_factors))])
        self.up_pool = Pool()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters[0])
        x, x_list, edge_index_list, perm_list = self.encoder(x, edge_index, data.batch)
        x = self.decoder(x)
        x = x.reshape(-1, self.filters[0])
        return x, x_list, edge_index_list, perm_list

    def encoder(self, x, edge_index, batch):
        self.new_edge_index = self.A_edge_index[0]
        self.new_norm = self.A_norm[0]
        self.new_batch = batch
        x_list = []
        edge_index_list = []
        perm_list = []
        for i in range(self.n_layers):
            # old coma
            # x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            # x = self.pool(x, self.downsample_matrices[i])

            x = F.relu(self.cheb[i](x, self.new_edge_index, self.new_norm))
            batch_size = x.shape[0]
            x = x.view(-1, x.shape[-1])
            x, new_edge_index, _, self.new_batch, perm, _ = self.down_pool[i](x, self.new_edge_index, batch=self.new_batch)
            x = x.view(batch_size, -1, x.shape[-1])
            self.new_edge_index, self.new_norm = ChebConv_Coma.norm(new_edge_index, x.shape[1])
            x_list.append(x.detach().cpu().numpy())
            edge_index_list.append(self.new_edge_index.detach().cpu().numpy())
            perm_list.append(perm.detach().cpu().numpy())
        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x, x_list, edge_index_list, perm_list

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.up_pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[0], self.A_norm[0])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

