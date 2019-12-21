import torch
import torch.nn.functional as F

from layers import ChebConv_Coma, Pool, DenseChebConv, SparseDiffPool, DenseDiffPool


class Coma(torch.nn.Module):

    def __init__(self, dataset, config, template_adj, num_nodes):
        super(Coma, self).__init__()
        self.n_layers = config['n_layers']
        self.enc_filters = config['enc_filters']
        self.dec_filters = config['dec_filters']
        self.K = config['polygon_order']
        self.z = config['z']
        top_adj = template_adj
        top_edge_index, top_norm = ChebConv_Coma.norm(top_adj._indices(), num_nodes[0])
        top_dense_adj = top_adj.to_dense()

        self.register_buffer('top_edge_index', top_edge_index)
        self.register_buffer('top_norm', top_norm)
        self.register_buffer('top_dense_adj', top_dense_adj)

        self.cheb = []
        self.cheb_dec = []
        self.pools = []
        self.unpools = []
        self.num_nodes = num_nodes
        for i in range(len(self.enc_filters) - 1):
            if i == 0:
                self.cheb.append(ChebConv_Coma(self.enc_filters[i], self.enc_filters[i + 1], self.K[i]))
                self.pools.append(
                    SparseDiffPool(self.num_nodes[i], self.num_nodes[i + 1], self.enc_filters[i + 1], self.K[i]))
            else:  # use dense
                self.cheb.append(DenseChebConv(self.enc_filters[i], self.enc_filters[i + 1], self.K[i]))
                self.pools.append(
                    DenseDiffPool(self.num_nodes[i], self.num_nodes[i + 1], self.enc_filters[i + 1], self.K[i]))

        for i in range(len(self.dec_filters) - 1):
            self.unpools.append(
                DenseDiffPool(self.num_nodes[-1 - i], self.num_nodes[-2 - i], self.dec_filters[i], self.K[i]))
            if i == len(self.dec_filters) - 2:
                self.cheb_dec.append(
                    ChebConv_Coma(self.dec_filters[i], self.dec_filters[i + 1], self.K[i]))  # note: K must be the same!
            else:
                self.cheb_dec.append(DenseChebConv(self.dec_filters[i], self.dec_filters[i + 1], self.K[i]))

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
        x, prev_adj, link_loss_1, ent_loss_1 = self.encoder(x)
        x, prev_adj, link_loss_2, ent_loss_2 = self.decoder(x, prev_adj)
        x = x.reshape(-1, self.dec_filters[-1])
        return x, link_loss_1 + link_loss_2, ent_loss_1 + ent_loss_2

    def encoder(self, x):
        prev_adj = None
        link_loss_sum = 0
        ent_loss_sum = 0
        for i in range(self.n_layers):
            if prev_adj is None:
                x = F.relu(self.cheb[i](x, self.top_edge_index, self.top_norm))
                x, prev_adj, link_loss, ent_loss = self.pools[i](x, self.top_edge_index, self.top_norm,
                                                                 self.top_dense_adj)
            else:
                x = F.relu(self.cheb[i](x, prev_adj))
                x, prev_adj, link_loss, ent_loss = self.pools[i](x, prev_adj)
            link_loss_sum += link_loss
            ent_loss_sum += ent_loss

        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x, prev_adj, link_loss_sum, ent_loss_sum

    def decoder(self, x, prev_adj):
        link_loss_sum = 0
        ent_loss_sum = 0
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.dec_filters[0])
        for i in range(self.n_layers):
            x, prev_adj, link_loss, ent_loss = self.unpools[i](x, prev_adj)
            link_loss_sum += link_loss
            ent_loss_sum += ent_loss
            if i == self.n_layers - 1:
                x = F.relu(self.cheb_dec[i](x, self.top_edge_index, self.top_norm))
            else:
                x = F.relu(self.cheb_dec[i](x, prev_adj))
        return x, prev_adj, link_loss_sum, ent_loss_sum

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)
