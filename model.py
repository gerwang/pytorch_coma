import torch
import torch.nn.functional as F

from layers import ChebConv_Coma, Pool, DenseChebConv, SparseDiffPool, DenseDiffPool

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
        self.cheb = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i-1], self.filters[-i-2], self.K[i])
                                             for i in range(len(self.filters)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer

        self.cheb = torch.nn.ModuleList(self.cheb)
        self.cheb_dec = torch.nn.ModuleList(self.cheb_dec)
        self.pools = torch.nn.ModuleList(self.pools)
        self.unpools = torch.nn.ModuleList(self.unpools)

        self.enc_lin = torch.nn.Linear(num_nodes[-1] * self.enc_filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.dec_filters[0] * num_nodes[-1])
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.enc_filters[0])
        x, prev_adj, link_losses_1, ent_losses_1 = self.encoder(x)
        x, prev_adj, link_losses_2, ent_losses_2 = self.decoder(x, prev_adj)
        link_losses_1.extend(link_losses_2)
        ent_losses_1.extend(ent_losses_2)
        link_loss = torch.sum(torch.cat(link_losses_1))
        ent_loss = torch.sum(torch.cat(ent_losses_1))
        x = x.reshape(-1, self.filters[0])
        return x, link_loss, ent_loss

    def encoder(self, x):
        prev_adj = None
        link_losses = []
        ent_losses = []
        for i in range(self.n_layers):
            if prev_adj is None:
                x = F.relu(self.cheb[i](x, self.top_edge_index, self.top_norm))
                x, prev_adj, link_loss, ent_loss = self.pools[i](x, self.top_edge_index, self.top_norm,
                                                                 self.top_dense_adj)
            else:
                x = F.relu(self.cheb[i](x, prev_adj))
                x, prev_adj, link_loss, ent_loss = self.pools[i](x, prev_adj)
            link_losses.append(link_loss)
            ent_losses.append(ent_loss)

        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x, prev_adj, link_losses, ent_losses

    def decoder(self, x, prev_adj):
        link_losses = []
        ent_losses = []
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

