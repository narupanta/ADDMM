import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.nn.models import VGAE
import numpy as np
import json
import os
from .utils import * 
from .normalization import Normalizer
import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import numpy as np
from tqdm import tqdm
# from core.datasetclass import LinearElasticityVAEDataset
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:55:42 2022

@author: marcomau
"""
# GNN for displacement/stress field predictions

import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing
import functools
import collections
from torch_geometric.data import Data
from .normalization import Normalizer

class GraphConvolutionCVAE(torch.nn.Module) :
    def __init__(self, latent_dim, input_size, hidden_size, skip,
                 node_feat_size,
                 edge_feat_size,
                 output_size,
                 params_size,                
                 name='GCVAE', act=F.elu, conv='GMMConv'):
        super(GraphConvolutionCVAE, self).__init__()   
        self.act = act
        self.hidden_size = hidden_size
        self.skip = skip
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.conv = conv
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.params_size = params_size

        self._output_normalizer = Normalizer(size=output_size, name='output_normalizer')
        self._node_features_normalizer = Normalizer(size=node_feat_size, name='node_features_normalizer')
        self._edge_features_normalizer = Normalizer(size=edge_feat_size, name='edge_features_normalizer')
        self._params_normalizer = Normalizer(size=params_size, name='params_normalizer')

        self.down_convs = Sequential(gnn.GMMConv(self.node_feat_size + self.params_size, 2, dim=1, kernel_size=5),
                                     gnn.GMMConv(2, 2, dim=1, kernel_size=5))
        self.mlp_mu = Sequential(Linear(self.input_size * node_feat_size, self.hidden_size),
                                   ReLU(),
                                   Linear(self.hidden_size, self.hidden_size),
                                   ReLU(),
                                   LayerNorm(self.hidden_size),
                                   Linear(self.hidden_size, self.latent_dim))
        self.mlp_logvar = Sequential(Linear(self.input_size * node_feat_size, self.hidden_size),
                                   ReLU(),
                                   Linear(self.hidden_size, self.hidden_size),
                                   ReLU(),
                                   LayerNorm(self.hidden_size),
                                   Linear(self.hidden_size, self.latent_dim))

        self.up_convs = torch.nn.ModuleList()
        self.up_convs = Sequential(gnn.GMMConv(2, 2, dim=1, kernel_size=5),
                                   gnn.GMMConv(2, self.node_feat_size, dim=1, kernel_size=5))
        
        self.up_latent = Sequential(Linear(self.latent_dim + self.params_size, self.hidden_size),
                                   ReLU(),
                                   Linear(self.hidden_size, self.hidden_size),
                                   ReLU(),
                                   LayerNorm(self.hidden_size),
                                   Linear(self.hidden_size, self.input_size * self.node_feat_size))

    def encode(self, input_graph) :
        normalized_node_features = self._node_features_normalizer(input_graph.x)
        normalized_mat = self._params_normalizer(input_graph.material_params)
        normalized_edge_features = self._edge_features_normalizer(input_graph.edge_features)
        input_x = torch.cat((normalized_node_features, normalized_mat), dim = 1)
        x = input_x
        idx = 0
        for layer in self.down_convs:
            x = self.act(layer(x, input_graph.edge_index, normalized_edge_features))
            if self.skip:
                x = x + normalized_node_features
            idx += 1

        x = x.reshape(1, -1)
        mu = self.mlp_mu(x)
        logvar = self.mlp_logvar(x)
        self.mu = mu
        self.logvar = logvar
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, input_graph, is_training) :
        normalized_mat = self._params_normalizer(input_graph.material_params, None, is_training)
        normalized_edge_features = self._edge_features_normalizer(input_graph.edge_features, None, is_training)
        mat = normalized_mat[0, :].reshape(1, -1)
        conditioned_latent_space = torch.cat((z, mat), dim = 1)
        x = self.up_latent(conditioned_latent_space)
        h = x.reshape(-1, self.node_feat_size)
        x = h
        idx = 0
        for layer in self.up_convs:
            x = layer(x, input_graph.edge_index, normalized_edge_features)
            if (idx != 1):
                x = self.act(x)
            if self.skip:
                x = x + h
            idx += 1
        return x
    def forward(self, input_graph, is_training) :
        mu, logvar = self.encode(input_graph)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, input_graph, is_training)
    
    def predict(self, input_graph) :
        z = self.reparameterize(self.mu, self.logvar)
        prediction = self.decode(z, input_graph, is_training = False)
        return self._output_normalizer.inverse(prediction)

    def loss_function(self, predictions, input_graph, beta) :
        loss = F.mse_loss(predictions, self._output_normalizer(input_graph.x))
        loss += -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * beta
        return loss
    
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.mu, os.path.join(path, "mu.pth"))
        torch.save(self.logvar, os.path.join(path, "logvar.pth"))
        torch.save(self._output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self._node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self._edge_features_normalizer, os.path.join(path, "edge_features_normalizer.pth"))
        torch.save(self._params_normalizer, os.path.join(path, "params_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.mu = torch.load(os.path.join(path, "mu.pth"))
        self.logvar = torch.load(os.path.join(path, "logvar.pth"))
        self._output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self._node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self._edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
        self._params_normalizer = torch.load(os.path.join(path, "params_normalizer.pth"))

if __name__ == "__main__" :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LinearElasticityVAEDataset(data_dir = "/home/narupanta/ADDMM/surrogate_model/dataset")
    print("number of trajectories: ", len(dataset))
    print(dataset[1])
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle = False)
    num_nodes = dataset[0].x.shape[0]
    model = GraphConvolutionCVAE(params_size = 2,
                                 hidden_size = 128,
                                 latent_dim = 2,
                                 input_size = num_nodes, skip = True, 
                                 act = F.elu, conv = 'GMMConv', node_feat_size=2,
                                 edge_feat_size= 1, output_size=2, latent_size=128)

    print("a")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)

    # Training loop
    num_epochs = 5
    loss_values = []
    pass_count = len(train_dataset)
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for idx_traj, batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss = model.loss_function(predictions, batch, beta = 0)

            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                # print(f"Trajectory {idx_traj}/{len(train_loader)}, Loss: {loss:.4f}")
            loop.set_description(f"trajectory {idx_traj + 1}/{len(train_loader)}")
            loop.set_postfix(loss = loss.item())
        
        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")