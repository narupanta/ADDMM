import json
import torch
from torch_geometric.data import Data, Dataset
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
    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        # Randomly generate node features
        file_name = self.file_name_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))

        decomposed_connectivity = triangles_to_edges(torch.tensor(data['node_connectivity']))['two_way_connectivity']
        sender, receiver = decomposed_connectivity
        max_params = torch.tensor([2 * 6e+10, 2 * 8e+10])
        min_params = torch.tensor([6e+10, 8e+10])
        u = data["u"]
        mesh_pos = torch.tensor(data["mesh_pos"], dtype=torch.float)
        triangles = torch.tensor(data['node_connectivity'])
        edge_index = torch.cat((decomposed_connectivity[0].reshape(1, -1), decomposed_connectivity[1].reshape(1, -1)), dim=0)
        # build edge_features
        edge_features = torch.norm(mesh_pos[sender] - mesh_pos[receiver], p = 2, dim = 1).reshape(-1, 1)
        # material_params = (torch.tensor(data["params"], dtype = torch.float32) * torch.ones(mesh_pos.shape[0], 2) - min_params)/(max_params - min_params)
        material_params = torch.tensor(data["params"], dtype = torch.float32) * torch.ones(mesh_pos.shape[0], 2)
        node_features = torch.cat((mesh_pos, material_params), dim = 1)
        gt_displacement = torch.tensor(u, dtype=torch.float)  # Target node values

        return Data(x = node_features, edge_index = edge_index, edge_features = edge_features, y = gt_displacement, triangles = triangles)
    def get_name(self, idx) :
        return self.file_name_list[idx]
    
class LinearElasticityVAEDataset(Dataset):
    def __init__(self, data_dir):
        """
        Generates synthetic dataset for material deformation use case.

        Args:
            num_graphs (int): Number of graphs in the dataset.
            num_nodes (int): Number of nodes per graph.
            num_features (int): Number of features per node.
            num_material_params (int): Number of material parameters.
        """
        super(LinearElasticityVAEDataset, self).__init__()
        self.data_dir = data_dir
        self.file_name_list = [filename for filename in sorted(os.listdir(data_dir)) if not os.path.isdir(os.path.join(data_dir, filename))]
    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        # Randomly generate node features
        file_name = self.file_name_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))

        decomposed_connectivity = triangles_to_edges(torch.tensor(data['node_connectivity']))['two_way_connectivity']
        sender, receiver = decomposed_connectivity

        u = data["u"]
        mesh_pos = torch.tensor(data["mesh_pos"], dtype=torch.float)
        triangles = torch.tensor(data['node_connectivity'])
        edge_index = torch.cat((decomposed_connectivity[0].reshape(1, -1), decomposed_connectivity[1].reshape(1, -1)), dim=0)
        # build edge_features
        edge_features = torch.norm(mesh_pos[sender] - mesh_pos[receiver], p = 2, dim = 1).reshape(-1, 1)
        material_params = torch.tensor(data["params"], dtype = torch.float32) * torch.ones(mesh_pos.shape[0], 2)
        gt_displacement = torch.tensor(u, dtype=torch.float)  # Target node values

        return Data(x = gt_displacement, material_params = material_params, edge_index = edge_index, edge_features = edge_features, triangles = triangles, pos = mesh_pos)
    def get_name(self, idx) :
        return self.file_name_list[idx]