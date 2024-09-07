import torch
from torch import nn
import numpy as np
import math

from modules.graph_conv import TimeHANConv, TimeHGTConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.conv import HeteroConv, GATConv

class EmbeddingModule(nn.Module):
  def __init__(self, memory, time_encoder, n_layers,
            n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    # self.memory = memory
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    pass


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]

class TimeEmbedding(EmbeddingModule):
  def __init__(self, memory, time_encoder, n_layers,
               n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(memory,
                                        time_encoder, n_layers,
                                        n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.embedding_dimension)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings

class GraphEmbedding(EmbeddingModule):
    def __init__(self, memory, time_encoder, n_layers,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, metadata = None):
        super(GraphEmbedding, self).__init__(memory,
                                             time_encoder, n_layers,
                                             n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device

        # self.lin_dict = torch.nn.ModuleDict()
        # for node_type in metadata[0]:
        #     self.lin_dict[node_type] = Linear(embedding_dimension, embedding_dimension)

        self.lin = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin[node_type] = torch.nn.LayerNorm(embedding_dimension)

    def compute_embedding(self, x_dict, edge_index_dict, x_time_dict, edge_feature_dict, edge_time_dict=None):
        # for node_type, x in x_dict.items():
        #     x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            # x_dict = conv(x_dict, edge_index_dict, x_time_dict, edge_feature_dict, edge_time_dict)
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict)

            # x_dict = conv(x_dict, edge_index_dict)

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin[node_type](x)
        out = x_dict
        return out

class HANGraphEmbedding(GraphEmbedding):
  def __init__(self, memory, time_encoder, n_layers, n_edge_features,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, metadata = None):
    super(GraphEmbedding, self).__init__(self, memory, time_encoder, n_layers,
                 n_time_features, embedding_dimension, device)
    inchannels = {}
    for node_type in metadata[0]:
        inchannels[node_type] = embedding_dimension

    self.convs = torch.nn.ModuleList()
    for _ in range(n_layers):
        conv = TimeHANConv(inchannels, embedding_dimension, n_edge_features, metadata, time_encoder,
                           n_heads, dropout=dropout)
        # conv = HeteroConv({
        #     ('user', '0', 'item'): GATConv((-1,-1), embedding_dimension),
        #     ('item', '1', 'user'): GATConv((-1,-1), embedding_dimension),
        # }, aggr='sum')
        self.convs.append(conv)
    self.lin = torch.nn.ModuleDict()
    for node_type in metadata[0]:
        self.lin[node_type] = torch.nn.LayerNorm(embedding_dimension)


class HGTGraphEmbedding(GraphEmbedding):
  def __init__(self, memory, time_encoder, n_layers,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, metadata = None):
    super(GraphEmbedding, self).__init__(self, memory, time_encoder, n_layers,
                 n_time_features, embedding_dimension, device)
    inchannels = {}
    for node_type in metadata[0]:
        inchannels[node_type] = embedding_dimension

    self.convs = torch.nn.ModuleList()
    for _ in range(n_layers):
        conv = TimeHGTConv(inchannels, embedding_dimension, metadata, time_encoder,
                           n_heads)
        self.convs.append(conv)

    self.lin = torch.nn.ModuleDict()
    for node_type in metadata[0]:
        self.lin[node_type] = torch.nn.LayerNorm(embedding_dimension)

def get_embedding_module(module_type, memory,
                         time_encoder, n_layers, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, use_memory=True, metadata = None):
  if module_type == "hgt":
    return HGTGraphEmbedding(memory=memory,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory, metadata = metadata)
  elif module_type == "han":
    return HANGraphEmbedding(memory=memory,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory, metadata = metadata)

  elif module_type == "identity":
    return IdentityEmbedding(
                             memory=memory,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(
                         memory=memory,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))