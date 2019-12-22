import torch
import torch.nn.functional as F

from layers import DenseChebConv, Pool, ChebConv_Coma


class Coma(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(Coma, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        self.filters.insert(0, dataset.num_features)  # To get initial features per node
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.cheb = []
        for i in range(len(self.filters) - 2):
            con = DenseChebConv
            if i == 0:
                con = ChebConv_Coma
            self.cheb.append(con(self.filters[i], self.filters[i + 1], self.K[i]))
        self.cheb = torch.nn.ModuleList(self.cheb)

        self.cheb_dec = []
        for i in range(len(self.filters) - 1):
            con = DenseChebConv
            if i >= len(self.filters) - 3:
                con = ChebConv_Coma
            self.cheb_dec.append(con(self.filters[-i - 1], self.filters[-i - 2], self.K[i]))
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.cheb_dec = torch.nn.ModuleList(self.cheb_dec)

        self.pool = Pool()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0] * self.filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1] * self.upsample_matrices[-1].shape[1])
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters[0])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, self.filters[0])
        return x

    def encoder(self, x):
        for i in range(self.n_layers):
            if i == 0:
                x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            else:
                adj = self.adjacency_matrices[i].to_dense()
                x = F.relu(self.cheb[i](x, adj))
            x = self.pool(x, self.downsample_matrices[i])
        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i - 1])
            if i == self.n_layers - 1:
                x = F.relu(
                    self.cheb_dec[i](x, self.A_edge_index[self.n_layers - i - 1], self.A_norm[self.n_layers - i - 1]))
            else:
                adj = self.adjacency_matrices[self.n_layers - i - 1].to_dense()
                x = F.relu(self.cheb_dec[i](x, adj))
        x = self.cheb_dec[-1](x, self.A_edge_index[0], self.A_norm[0])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)
