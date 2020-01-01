import torch
import torch.nn.functional as F

from layers import ChebConv_Coma, Pool


class Coma(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(Coma, self).__init__()
        self.n_layers = config['n_layers']
        self.enc_filters = config['enc_filters']
        self.dec_filters = config['dec_filters']
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.cheb = torch.nn.ModuleList([ChebConv_Coma(self.enc_filters[i], self.enc_filters[i + 1], self.K[i])
                                         for i in range(len(self.enc_filters) - 1)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.dec_filters[i], self.dec_filters[i + 1], self.K[i])
                                             for i in range(len(self.dec_filters) - 1)])
        # self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0] * self.enc_filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.dec_filters[0] * self.upsample_matrices[-1].shape[1])
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.enc_filters[0])
        z_1 = self.encoder(x)
        x = self.decoder(z_1)
        if self.training:
            z_2 = self.encoder(x)
            x = x.reshape(-1, self.dec_filters[-1])
            return x, z_1, z_2
        else:
            x = x.reshape(-1, self.dec_filters[-1])
            return x

    def encoder(self, x):
        for i in range(self.n_layers):
            x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])
        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = self.enc_lin(x)
        return x

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.dec_filters[0])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i - 1])
            x = self.cheb_dec[i](x, self.A_edge_index[self.n_layers - i - 1], self.A_norm[self.n_layers - i - 1])
            if i < self.n_layers - 1:
                x = F.relu(x)
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)
