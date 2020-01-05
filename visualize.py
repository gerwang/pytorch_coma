import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model import Coma
from transform import Normalize
from tqdm import tqdm


def main():

    config = read_config('./topk.cfg')
    print('Loading Dataset')
    data_dir = config['data_dir']

    normalize_transform = Normalize()
    dataset_test = ComaDataset(data_dir, dtype='test', split='sliced', split_term='sliced',
                               pre_transform=normalize_transform)
    
    mean = dataset_test.mean.numpy()
    std = dataset_test.std.numpy()
    for i, data_file in tqdm(enumerate(dataset_test.data_file[:5])):
        mesh = Mesh(filename=data_file)
        mesh.v = mesh.v * std + mean
        for layer in range(4):
            npz_file = np.load('./mesh_output/{}_{}.npz'.format(i, layer))
            # x = npz_file['x']
            edge_index = npz_file['edge_index']
            perm = npz_file['perm']
            mask = np.zeros(mesh.v.shape[0])
            for idx in perm:
                mask[idx] = 1
            mask = (mask==1)
            # new vertices
            new_v = mesh.v[mask]
            print('point num:', new_v.shape[0])
            # new faces from edge_index
            new_f = []
            adjacency = np.zeros((new_v.shape[0], new_v.shape[0]))
            for j in range(edge_index.shape[1]):
                start = edge_index[0][j]
                end = edge_index[1][j]
                adjacency[start][end] = 1
                adjacency[end][start] = 1
            for j in range(edge_index.shape[1]):
                start = edge_index[0][j]
                end = edge_index[1][j]
                if start == end:
                    continue
                if start > end:
                    start, end = end, start
                for k in range(end+1, new_v.shape[0]):
                    if adjacency[start][k] == 1 and adjacency[end][k] == 1:
                        # find a face
                        new_f.append([start, end, k])
            new_f = np.array(new_f)
            print('face num:', new_f.shape[0])
            new_mesh = Mesh(v=new_v, f=new_f)
            new_mesh.write_obj('./mesh_output/{}_{}.obj'.format(i, layer))
            mesh.v = new_v

if __name__ == "__main__":
    main()
    