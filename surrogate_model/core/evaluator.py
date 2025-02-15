import torch
from torch_geometric.data import Data, Dataset
from core.utils import *
import numpy as np

def build_graph(mesh_pos, node_connectivity, material_params) :
    material_params = torch.tensor(material_params, dtype = torch.float32) * torch.ones(mesh_pos.shape[0], 2)
    node_features = torch.cat((mesh_pos, material_params), dim = 1)
    decomposed_connectivity = triangles_to_edges(torch.tensor(node_connectivity))['two_way_connectivity']
    sender, receiver = decomposed_connectivity
    edge_index = torch.cat((sender.reshape(1, -1), receiver.reshape(1, -1)), dim=0)
    edge_features = torch.norm(mesh_pos[sender] - mesh_pos[receiver], p = 2, dim = 1).reshape(-1, 1)
    return Data(x = node_features, edge_index = edge_index, edge_features = edge_features, triangles = decomposed_connectivity)

def predictor(model, graph) :
    model.eval()
    normalize_deformation = model(graph)
    predicted_deformation = model._output_normalizer.inverse(normalize_deformation)
    return graph, predicted_deformation
