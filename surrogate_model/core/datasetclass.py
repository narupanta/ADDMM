import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
from .utils import *
import numpy as np
class LinearElasticityDataset(Dataset):
    def __init__(self, data_dir):
        """
        Generates synthetic dataset for material deformation use case.

        Args:
            num_graphs (int): Number of graphs in the dataset.
            num_nodes (int): Number of nodes per graph.
            num_features (int): Number of features per node.
            num_material_params (int): Number of material parameters.
        """
        super(LinearElasticityDataset, self).__init__()
        self.data_dir = data_dir
        self.file_name_list = [filename for filename in sorted(os.listdir(data_dir)) if not os.path.isdir(os.path.join(data_dir, filename))]
        with open(os.path.join(data_dir, "params_desc/G_K_indices.json"), "r") as file:
            meta = json.load(file)
        # Print the content
        self.meta = meta
    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        # Randomly generate node features
        file_name = self.file_name_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))
        i_g = int(file_name.rstrip(".npz").split("_")[2])
        i_k = int(file_name.rstrip(".npz").split("_")[3])

        decomposed_connectivity = triangles_to_edges(torch.tensor(data['node_connectivity']))['two_way_connectivity']
        nodal_material_params = [self.meta["G_list"][i_g], self.meta["G_list"][i_k]]

        u = data["u"]
        mesh_pos = torch.tensor(data["mesh_pos"], dtype=torch.float)
        triangles = torch.tensor(torch.tensor(data['node_connectivity']))
        # edge_index = torch.cat((decomposed_connectivity[0].reshape(-1, 1), decomposed_connectivity[1].reshape(-1, 1)), dim=1)
        edge_index = torch.cat((decomposed_connectivity[0].reshape(1, -1), decomposed_connectivity[1].reshape(1, -1)), dim=0)
        material_params = torch.tensor(nodal_material_params).repeat(mesh_pos.shape[0], 1)  # Material parameter
        gt_displacement = torch.tensor(u, dtype=torch.float)  # Target node values

        return Data(mesh_pos = mesh_pos, edge_index = edge_index, triangles = triangles, material_params = material_params, gt_displacement = gt_displacement)
    def get_name(self, idx) :
        return self.file_name_list[idx]