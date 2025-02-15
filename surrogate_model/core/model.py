import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import VGAE
import numpy as np
import json
import os
from .utils import * 
from .normalization import Normalizer
# Encoder definition with conditional material parameters
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, material_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim + material_dim, hidden_dim)  # Conditional input with material parameters
        self.conv_mu = GCNConv(hidden_dim, latent_dim)  # Mean of latent distribution
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)  # Log-variance of latent distribution

    def forward(self, mesh_node, edge_index, material_params):
        # Concatenate material parameters to node features
        x = torch.cat([mesh_node, material_params], dim=1)
        x = F.relu(self.conv1(x, edge_index))  # Apply GCNConv with material parameters
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


# Decoder definition using latent space and material parameters
class GraphDecoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, material_dim):
        super(GraphDecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim + material_dim, latent_dim)  # Conditional input with material parameters
        self.conv2 = GCNConv(latent_dim, output_dim)

    def forward(self, z, edge_index, material_params):
        # Concatenate material parameters to the latent space z
        z = torch.cat([z, material_params], dim=1)
        x = F.relu(self.conv1(z, edge_index))  # Apply GCNConv with material parameters
        x = self.conv2(x, edge_index)
        return x


# Complete CVAE model using VGAE with conditional decoder
class CVGAEWithDeformationDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(CVGAEWithDeformationDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._output_normalizer = Normalizer(2, "output_normalizer")
        self._meshpos_normalizer = Normalizer(2, "meshpos_normalizer")
    def encode(self, x, edge_index, material_params):
        mu, logvar = self.encoder(x, edge_index, material_params)
        self.mu = mu
        self.logvar = logvar
        return mu, logvar

    def decode(self, z, edge_index, material_params):
        return self.decoder(z, edge_index, material_params)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, data, is_training) :
        node_features = self._meshpos_normalizer(data.mesh_pos, None, is_training)
        edge_index = data.edge_index
        material_params = data.material_params
        mu, logvar = self.encode(node_features, edge_index, material_params)
        z = self.reparameterize(mu, logvar)
        # Decode to predict nodal values
        self.latent_space = z
        self.edge_index = data.edge_index
        self.mesh_pos = data.mesh_pos
        self.triangles = data.triangles
        return self.decode(z, edge_index, material_params).squeeze()

    def loss(self, output, data, beta) :
        loss = F.mse_loss(output, self._output_normalizer(data.gt_displacement, None, True))
        loss += -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * beta
        return loss

