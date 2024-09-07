import logging
import random

import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.mailbox import Mailbox
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.message_aggregator import get_message_aggregator
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode

from torch_geometric.nn import Linear, HGTConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter_max
import time

class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
      super().__init__()
      self.fc_1 = torch.nn.Linear(dim, dim)
      self.fc_2 = torch.nn.Linear(dim, dim)
      self.fc_3 = torch.nn.Linear(dim, 1)
      self.act = torch.nn.Sigmoid()
      self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def reset_parameters(self):
      self.fc_1.reset_parameters()
      self.fc_2.reset_parameters()
      self.fc_3.reset_parameters()

  def forward(self, x):
      x = torch.relu(self.fc_1(x))
      x = self.dropout(x)
      x = torch.relu(self.fc_2(x))
      x = self.dropout(x)
      return self.fc_3(x).squeeze(dim=1)

class Decoder(torch.nn.Module):
  def __init__(self, dim, drop=0.1):
      super().__init__()
      self.fc_1 = torch.nn.Linear(dim, dim)
      self.fc_2 = torch.nn.Linear(dim, dim)
      self.fc_3 = torch.nn.Linear(dim, 2)
      self.act = torch.nn.Sigmoid()
      self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def reset_parameters(self):
      self.fc_1.reset_parameters()
      self.fc_2.reset_parameters()
      self.fc_3.reset_parameters()

  def forward(self, x):
      x = torch.relu(self.fc_1(x))
      x = self.dropout(x)
      x = torch.relu(self.fc_2(x))
      x = self.dropout(x)
      return self.fc_3(x)#.squeeze(dim=1)

class decoder(torch.nn.Module):
  def __init__(self, dim, out_dim, drop=0.1):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, dim)
    self.fc_2 = torch.nn.Linear(dim//2, dim//4)
    self.fc_3 = torch.nn.Linear(dim, out_dim)
    self.act = torch.nn.Tanh()
    self.norm = torch.nn.LayerNorm(dim)
    self.norm1 = torch.nn.LayerNorm(dim//2)
    self.norm2 = torch.nn.LayerNorm(dim//4)
    # self.act = torch.nn.functional.leaky_relu(0.1)
    self.dropout = torch.nn.Dropout(p=0, inplace=False)

    torch.nn.init.xavier_normal_(self.fc_1.weight)
    torch.nn.init.xavier_normal_(self.fc_2.weight)
    torch.nn.init.xavier_normal_(self.fc_3.weight)

  def forward(self, x):
    x = self.act((self.fc_1(x)))
    x = self.dropout(x)
    # x1 = x = self.act(self.fc_2(x))
    # x = self.dropout(x)
    return self.fc_3(x), x#.squeeze(dim=1)

  def reset_parameters(self):
    self.fc_1.reset_parameters()
    self.fc_2.reset_parameters()
    self.fc_3.reset_parameters()

class TGN(torch.nn.Module):
  def __init__(self, n_nodes, device, edge_feature, n_layers=4,
               n_heads=2, dropout=0.1, use_memory=True,
               message_dimension=32,embedding_dimension = 32, num_labels = 2,
               memory_dimension=500, embedding_module_type="hgt",
               message_function="mlp",
               log_time=False, aggregator_type="last",
               memory_updater_type="rnn", deliver_to = 'all',
               metadata=None):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.device = device
    self.logger = logging.getLogger(__name__)

    # self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    # self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    # self.n_node_features = self.node_raw_features.shape[1]
    if not isinstance(n_nodes, dict):
        n_nodes = {node_type: n_nodes for node_type in metadata[0]}

    self.n_nodes = n_nodes
    # self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = embedding_dimension
    self.embedding_module_type = embedding_module_type
    self.metadata = metadata

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.embedding_dimension, log_time=log_time)
    self.memory = None
    self.deliver_to = deliver_to

    if self.use_memory:
      self.memory_dimension = embedding_dimension
      # self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = self.embedding_dimension + \
                              self.time_encoder.dimension
      message_dimension = embedding_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Mailbox(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device, metadata=metadata)

      self.message_function = get_message_function(module_type='mlp',
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension = embedding_dimension,
                                                   metadata=metadata)

      # self.message_constructor = get_embedding_module(module_type=embedding_module_type,
      #                                            memory=self.memory,
      #                                            time_encoder=self.time_encoder,
      #                                            n_layers=2,
      #                                            n_time_features=self.embedding_dimension,
      #                                            embedding_dimension=self.embedding_dimension,
      #                                            device=self.device,
      #                                            n_heads=n_heads, dropout=dropout,
      #                                            use_memory=use_memory,
      #                                            metadata=metadata)

      self.message_aggregator = get_message_aggregator('mean', device=self.device)

      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=embedding_dimension,
                                               memory_dimension=self.embedding_dimension,
                                               device=device, metadata=metadata)

    self.embedding_module_type = embedding_module_type

    self.edge_feature = torch.from_numpy(edge_feature).float().to(self.device)

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 memory=self.memory,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=2, n_edge_features = self.edge_feature.shape[-1],
                                                 n_time_features=self.embedding_dimension,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 metadata=metadata)

    self.embedding_module2 = get_embedding_module(module_type=embedding_module_type,
                                                 memory=self.memory,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=2, n_edge_features = self.edge_feature.shape[-1],
                                                 n_time_features=self.embedding_dimension,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 metadata=metadata)

    self.lin = torch.nn.ModuleDict()
    for node_type in metadata[0]:
        self.lin[node_type] = torch.nn.Linear(embedding_dimension, embedding_dimension, bias=False)

    self.edge_feature_linear = torch.nn.ModuleDict()
    for edge_type in metadata[1]:
        edge_type_ = '__'.join(edge_type)
        self.edge_feature_linear[edge_type_] = torch.nn.Linear(edge_feature.shape[1], embedding_dimension)

    self.distance = decoder(2*self.embedding_dimension, 1, drop=0.0)#torch.nn.Linear(2*embedding_dimension, 1)#torch.nn.ModuleDict()

    self.negative_weight = 1.0

    self.last_neg = []

    self.decoder = Decoder(self.embedding_dimension, drop=0.0)

    self.author_decoder = torch.nn.Linear(self.embedding_dimension, self.embedding_dimension)
    self.paper_decoder = torch.nn.Linear(self.embedding_dimension, self.embedding_dimension)
    self.keyword_decoder = torch.nn.Linear(self.embedding_dimension, self.embedding_dimension)
    self.venue_decoder = torch.nn.Linear(self.embedding_dimension, self.embedding_dimension)


  def compute_temporal_embeddings(self, node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    # node memory
    node_memory = {}
    neg_node_memory = {} # for negative sample
    t1 = time.time()
    # get_updated node memory
    for node_type in self.metadata[0]:
        # node_memory[node_type], _ = self.get_updated_memory(node_x[node_type], node_type)
        # neg_node_memory[node_type], _ = self.get_updated_neg_memory(node_x[node_type], node_type)

        node_memory[node_type] = self.memory.get_memory(node_x[node_type], node_type)
        # print(node_memory[node_type][:5,:5])
        # neg_node_memory[node_type] = self.memory.get_neg_memory(node_x[node_type], node_type)
        # shuffle_index = torch.randperm(node_memory[node_type].shape[0])
        # neg_node_memory[node_type] = node_memory[node_type][shuffle_index]

        # get neighbor edges & nodes for graph embedding
    # neighbor_node_memory_dict = {}
    # neg_neighbor_node_memory_dict = {}
    # neighbor_node_timestamp_dict = {}

    neighbor_edge_index = {}
    neighbor_edge_idx = {}
    neighbor_edge_rel_times = {}

    # for node_type in self.metadata[0]:
    #     neighbor_node_memory_dict[node_type] = node_memory[node_type][node_mask[node_type]==1]
    #     neg_neighbor_node_memory_dict[node_type] = neg_node_memory[node_type][node_mask[node_type]==1]
    #     neighbor_node_timestamp_dict[node_type] = node_timestamp[node_type][node_mask[node_type]==1]

    for edge_type in self.metadata[1]:
        neighbor_edge_index[edge_type] = edge_index[edge_type][:, (edge_mask[edge_type] != 0).view(-1)].view(2,-1)
        neighbor_edge_idx[edge_type] = self.edge_feature[edge_idx[edge_type][edge_mask[edge_type]!= 0]]#.view(-1, 1)
        edge_idx[edge_type] = self.edge_feature[edge_idx[edge_type]]#.view(-1, 1)
        neighbor_edge_rel_times[edge_type] = edge_rel_times[edge_type][edge_mask[edge_type]!= 0].view(-1, 1)
    t2 = time.time()
    # graph embedding
    node_embedding = self.embedding_module.compute_embedding(node_memory, neighbor_edge_index,
                                                             node_timestamp, neighbor_edge_idx,
                                                             neighbor_edge_rel_times)
    # neg_node_embedding = self.embedding_module.compute_embedding(neg_node_memory, neighbor_edge_index,
    #                                                              node_timestamp, neighbor_edge_idx,
    #                                                              neighbor_edge_rel_times)

    # node_embedding = node_memory
    # neg_node_embedding = neg_node_memory


    t3 = time.time()
    # message computing (updated event node embedding)
    event_edge_index = {}
    event_edge_idx = {}
    event_edge_rel_times = {}
    with torch.no_grad():
        for edge_type in self.metadata[1]:
            event_edge_index[edge_type] = edge_index[edge_type][:, (edge_mask[edge_type] != 2).view(-1)].view(2,-1)
            event_edge_idx[edge_type] = edge_idx[edge_type][edge_mask[edge_type] != 2]
            event_edge_rel_times[edge_type] = edge_rel_times[edge_type][edge_mask[edge_type] != 2].view(-1, 1)

        updated_node_embedding = self.embedding_module.compute_embedding(node_embedding, event_edge_index,
                                                                 node_timestamp, event_edge_idx,
                                                                 event_edge_rel_times)

        # pos_pool, updated_pos_pool = 0, 0
        # for node_type in self.metadata[0]:
        #
        #     if torch.sum(node_mask[node_type] == 0) > 0:
        #         pos_pool += global_mean_pool(node_embedding[node_type][node_mask[node_type] == 0],
        #                                      batch[node_type][node_mask[node_type] == 0])
        #     if torch.sum(node_mask[node_type] == 1) > 0:
        #         updated_pos_pool += global_mean_pool(node_embedding[node_type][
        #                                                  torch.logical_and(node_mask[node_type] == 1,
        #                                                                    node_negative[node_type] == 0)],
        #                                              batch[node_type][torch.logical_and(node_mask[node_type] == 1,
        #                                                                                 node_negative[node_type] == 0)])
        #     message = pos_pool+updated_pos_pool

        # neg_updated_node_embedding = self.embedding_module.compute_embedding(neg_node_embedding, edge_index,
        #                                                          node_timestamp, edge_idx,
        #                                                          edge_rel_times)

    # updated_node_embedding = self.embedding_module2.compute_embedding(node_memory, event_edge_index,
    #                                                                  node_timestamp, event_edge_idx,
    #                                                                  event_edge_rel_times)
    # neg_updated_node_embedding = self.embedding_module2.compute_embedding(neg_node_embedding, event_edge_index,
    #                                                                      node_timestamp, event_edge_idx,
    #                                                                      event_edge_rel_times)
    t4 = time.time()
    # message transform
    with torch.no_grad():
        # node_message = self.embedding_module.compute_embedding(updated_node_embedding, neighbor_edge_index,
        #                                                          node_timestamp, neighbor_edge_idx,
        #                                                          neighbor_edge_rel_times)
        # neg_node_message = self.embedding_module.compute_embedding(neg_updated_node_embedding, neighbor_edge_index,
        #                                                              node_timestamp, neighbor_edge_idx,
        #                                                              neighbor_edge_rel_times)

        # message delivering
        # for node_type in self.metadata[0]:
        #     self.update_memory(node_x[node_type], node_type)
        #     self.update_neg_memory(node_x[node_type], node_type)
            # self.memory.clear_messages(node_x[node_type].cpu().numpy().tolist(), node_type)

        # node_id_to_messages = self.get_raw_messages(node_x, node_mask, node_negative, node_timestamp, updated_node_embedding, deliver_to=self.deliver_to)
        # node_id_to_neg_messages = self.get_raw_messages(node_x, node_mask, node_negative, node_timestamp, neg_updated_node_embedding, deliver_to='all', negative=True)
        t5 = time.time()

        # for node_type in self.metadata[0]:
            # self.memory.store_raw_messages(node_id_to_messages[node_type].keys(), node_id_to_messages[node_type], node_type)
            # self.memory.store_neg_messages(node_id_to_neg_messages[node_type].keys(), node_id_to_neg_messages[node_type],
            #                                node_type)

    return node_embedding, node_memory, node_memory, node_memory

  def compute_temporal_embeddings_noupdating(self, node_x, node_mask, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    # node memory
    node_memory = {}
    t1 = time.time()
    # get_updated node memory
    for node_type in self.metadata[0]:
        node_memory[node_type] = self.memory.get_memory(node_x[node_type], node_type)

    neighbor_edge_index = {}
    neighbor_edge_idx = {}
    neighbor_edge_rel_times = {}

    for edge_type in self.metadata[1]:
        neighbor_edge_index[edge_type] = edge_index[edge_type][:, (edge_mask[edge_type] == 1).view(-1)].view(2,-1)
        neighbor_edge_idx[edge_type] = edge_idx[edge_type][edge_mask[edge_type]==1].view(-1, 1)
        neighbor_edge_rel_times[edge_type] = edge_rel_times[edge_type][edge_mask[edge_type]==1].view(-1, 1)
    t2 = time.time()
    # graph embedding
    node_embedding = self.embedding_module.compute_embedding(node_memory, neighbor_edge_index,
                                                             node_timestamp, neighbor_edge_idx,
                                                             neighbor_edge_rel_times)


    return node_embedding

  def predict(self, node_x, node_mask, node_negative, node_timestamp, edge_index,
                                                           edge_rel_times, edge_idx, edge_mask, batch):
      node_embedding, updated_node_embedding, negative_node_embedding, updated_negative_node_embedding = self.compute_temporal_embeddings(
          node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)

      event_rep = 0
      for node_type in self.metadata[0]:
          if torch.sum(node_mask[node_type] == 0) > 0:
              # pos_pool += global_mean_pool(node_embedding[node_type][node_mask[node_type] == 0], batch[node_type][node_mask[node_type] == 0])
              pos_pool = node_embedding[node_type][node_mask[node_type] == 0]

      # pred_class = self.decoder(pos_pool)
      # print(pred_class)
      return pos_pool

  def get_nega_embedding(self, nega_graph):
      node_x = nega_graph.collect('x')  # for feature & memory
      node_mask = nega_graph.collect(
          'node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
      node_timestamp = nega_graph.collect('timestamp')

      # edge information
      edge_index = nega_graph.collect('edge_index')
      edge_rel_times = nega_graph.collect('edge_rel_times')
      edge_idx = nega_graph.collect('edge_idxs')  # for feature
      edge_mask = nega_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

      batch = nega_graph.collect('batch')

      node_embedding = self.compute_temporal_embeddings_noupdating(
          node_x, node_mask, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask)

      pool = 0
      for node_type in self.metadata[0]:
          if torch.sum(node_mask[node_type] == 1) > 0:
              pool += global_mean_pool(node_embedding[node_type][node_mask[node_type] == 1],
                                                   batch[node_type][node_mask[node_type] == 1])

      return pool

  def compute_event_probabilities(self, node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    # batch = event_graph.collect('batch')
    # mask = event_graph.collect('node_mask')
    #
    # node_x = event_graph.collect('x')
    # node_memory = {}
    # neg_node_memory = {}
    # for node_type in self.metadata[0]:
    #     node_memory[node_type], _ = self.get_updated_memory(node_x[node_type], node_type)
    #     neg_node_memory[node_type], _ = self.get_updated_neg_memory(node_x[node_type], node_type)
    # updated_neg_pool = self.get_nega_embedding(nega_graph)
    node_embedding, updated_node_embedding, negative_node_embedding, updated_negative_node_embedding = self.compute_temporal_embeddings(
      node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)
    # pos_score = 0
    # neg_score = 0

    pos_pool, updated_pos_pool = 0,0
    pos_reg, neg_reg = 0,0
    updated_pos_reg, updated_neg_reg = 0,0
    neg_pool, updated_neg_pool = 0, 0

    # updated_neg_pool = self.get_nega_embedding(nega_graph)

    for node_type in self.metadata[0]:

        if torch.sum(node_mask[node_type]==0) > 0:
            # pos_pool += global_mean_pool(node_embedding[node_type][node_mask[node_type] == 0], batch[node_type][node_mask[node_type] == 0])
            pos_pool = node_embedding[node_type][node_mask[node_type] == 0]#torch.logical_and(node_mask[node_type] == 0, node_negative[node_type]==0)]
            # neg_pool += global_mean_pool(negative_node_embedding[node_type][node_mask[node_type] == 0], batch[node_type][node_mask[node_type] == 0])
        if torch.sum(node_mask[node_type]==1) > 0:
            # updated_pos_pool += global_mean_pool(node_embedding[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]!=1)], batch[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]!=1)])
            updated_pos_pool = node_embedding[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]==0)]
            # updated_neg_reg += global_mean_pool(negative_node_embedding[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]>0)], batch[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]>0)])
            shuffle_index = torch.roll(torch.arange(updated_pos_pool.shape[0]), 4, 0)#torch.randperm(torch.sum(node_mask[node_type]==1))

            # updated_neg_pool += global_mean_pool(node_embedding[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]>0)], batch[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]>0)])#[shuffle_index]#
            updated_neg_pool = node_embedding[node_type][torch.logical_and(node_mask[node_type] == 1, node_negative[node_type]==1)]#[shuffle_index]#
            # print(global_mean_pool(node_embedding[node_type][node_mask[node_type] == 1], batch[node_type][node_mask[node_type] == 1]).shape,updated_neg_pool.shape)
            # updated_neg_pool += global_mean_pool(node_embedding[node_type][node_mask[node_type] == 1], batch[node_type][node_mask[node_type] == 1])[shuffle_index]#[:torch.sum(node_mask[node_type]==1)]
        # pos_pool += global_mean_pool(torch.tanh(self.lin[node_type](event_node_embedding[node_type].detach())), batch[node_type])
        # pos_pool += global_mean_pool(node_embedding[node_type], batch[node_type])
        # updated_pos_pool += global_mean_pool(updated_node_embedding[node_type], batch[node_type])
        # # neg_pool += global_mean_pool(torch.tanh(self.lin[node_type](negative_node_embedding[node_type].detach())), batch[node_type])
        # neg_pool += global_mean_pool(negative_node_embedding[node_type], batch[node_type])
        # # shuffle_index = torch.randperm(event_node_embedding[node_type][mask[node_type]==1].shape[0])
        # # negative_event_memory_list[_][node_type] = negative_event_memory_list[_][node_type][shuffle_index]
        # updated_neg_pool += global_mean_pool(updated_negative_node_embedding[node_type], batch[node_type])

        # pos_reg += global_mean_pool(node_memory[node_type], batch[node_type])
        # updated_pos_reg += global_mean_pool(node_embedding[node_type], batch[node_type])
        # neg_reg += global_mean_pool(neg_node_memory[node_type], batch[node_type])
        # updated_neg_reg += global_mean_pool(negative_node_embedding[node_type], batch[node_type])
    shuffle_index = torch.randperm(updated_pos_pool.shape[0])
    # print(pos_pool[:5,:5])
    # print(updated_pos_pool[:5,:5])
    # print(updated_neg_pool[:5,:5])
    pos_score,_ = self.distance(torch.cat([pos_pool, updated_pos_pool],dim=-1))#self.distance(self.hadamad_distance(pos_pool, updated_pos_pool))
    neg_score,_ = self.distance(torch.cat([pos_pool, updated_neg_pool],dim=-1))#self.distance(self.hadamad_distance(neg_pool, updated_neg_pool))
    # pos_score = torch.sum(torch.cat([pos_pool, updated_pos_pool, pos_pool-updated_pos_pool, pos_pool*updated_pos_pool],dim=-1), dim=-1).view(-1,1)
    # neg_score = torch.sum(torch.cat([pos_pool, updated_neg_pool, pos_pool-updated_neg_pool, pos_pool*updated_neg_pool],dim=-1), dim=-1).view(-1,1)
    # pos_reg, _ = self.distance(torch.cat([pos_pool, updated_pos_pool],
    #                                        dim=-1))  # self.distance(self.hadamad_distance(pos_pool, updated_pos_pool))
    # neg_reg, _ = self.distance(torch.cat([pos_pool, updated_neg_reg],
    #                                        dim=-1))  # self.distance(self.hadamad_distance(neg_pool, updated_neg_pool))
    # if len(self.last_neg) < 5:
    #     updated_neg_pool = updated_pos_pool
    # else:
    #     updated_neg_pool = self.last_neg[random.randint(0, len(self.last_neg)-1)]
    #     while updated_neg_pool.shape[0] != updated_pos_pool.shape[0]:
    #         updated_neg_pool = self.last_neg[random.randint(0, len(self.last_neg) - 1)]
    # self.last_neg.append(updated_pos_pool.detach())


    # shuffle_index = torch.randperm(updated_pos_pool.shape[0])
    # pos_score = torch.sum(
    #     pos_pool * updated_pos_pool,
    #     dim=-1).view(-1, 1)
    # neg_score = torch.sum(
    #     pos_pool * updated_neg_pool,
    #     dim=-1).view(-1, 1)

    pos_reg = torch.sum(
        pos_pool * updated_pos_pool,
        dim=-1).view(-1, 1)
    neg_reg,_ = self.distance(torch.cat([pos_pool, updated_pos_pool[shuffle_index]],dim=-1))
        # torch.sum(
        # pos_pool * updated_neg_reg,
        # dim=-1).view(-1, 1)

    # pos_score += global_mean_pool((self.distance[node_type](self.hadamad_distance(event_node_embedding[node_type],updated_event_node_embedding[node_type]))), batch[node_type])
    # neg_score += global_mean_pool((self.distance[node_type](self.hadamad_distance(negative_node_embedding[node_type], updated_negative_node_embedding[node_type]))), batch[node_type])

    return pos_score, neg_score, pos_reg, neg_reg

  def pass_link(self, node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """

    node_embedding, updated_node_embedding, negative_node_embedding, updated_negative_node_embedding = self.compute_temporal_embeddings(
        node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)

    return None, None

  def compute_heteroevent_probabilities(self, node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    # batch = event_graph.collect('batch')
    # mask = event_graph.collect('node_mask')
    #
    # node_x = event_graph.collect('x')
    # node_memory = {}
    # neg_node_memory = {}
    # for node_type in self.metadata[0]:
    #     node_memory[node_type], _ = self.get_updated_memory(node_x[node_type], node_type)
    #     neg_node_memory[node_type], _ = self.get_updated_neg_memory(node_x[node_type], node_type)
    # updated_neg_pool = self.get_nega_embedding(nega_graph)
    node_embedding, updated_node_embedding, negative_node_embedding, updated_negative_node_embedding = self.compute_temporal_embeddings(
      node_x, node_mask, node_negative, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)
    # pos_score = 0
    # neg_score = 0

    # pos_pool, updated_pos_pool = 0,0
    # pos_reg, neg_reg = 0,0
    # updated_pos_reg, updated_neg_reg = 0,0
    # neg_pool, updated_neg_pool = 0, 0

    # updated_neg_pool = self.get_nega_embedding(nega_graph)
    # paper_pos = node_embedding['paper'][node_mask['paper']==0][::2]
    paper_pos = self.paper_decoder(node_embedding['paper'][node_mask['paper']==0]) #+ self.keyword_decoder(global_mean_pool(node_embedding['keyword'][node_negative['keyword']==0], batch['keyword'][node_negative['keyword']==0]))
    # paper_pos = node_embedding['paper'][node_mask['paper'] == 0]#self.author_decoder(node_embedding['paper'][node_mask['paper'] == 0])#global_mean_pool(node_embedding['author'][node_mask['author'] != 2], batch['author'][node_mask['author'] != 2])) + self.keyword_decoder(global_mean_pool(node_embedding['keyword'][node_mask['keyword'] != 2], batch['keyword'][node_mask['keyword'] != 2])) + self.venue_decoder(global_mean_pool(node_embedding['venue'], batch['venue']))

    # author_pos = self.author_decoder(node_embedding['author'][torch.logical_and(node_mask['author'] == 1, node_negative['author']==0)])
    # author_neg = self.author_decoder(node_embedding['author'][torch.logical_and(node_mask['author'] == 1, node_negative['author']==1)])

    cite_pos = self.author_decoder(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==0)])#self.paper_decoder(global_mean_pool(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==0)], batch['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==0)]))
    # venue_pos = self.venue_decoder(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==0)])
    cite_neg = self.author_decoder(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==1)])#self.paper_decoder(global_mean_pool(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==1)], batch['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==1)]))
    # venue_neg = self.venue_decoder(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==1)])
    # venue_neg = self.venue_decoder(node_embedding['paper'][torch.logical_and(node_mask['paper'] == 1, node_negative['paper']==1)])
    shuffle_index = torch.roll(torch.arange(cite_neg.shape[0]), 4,
                               0)  # torch.randperm(torch.sum(node_mask[node_type]==1))
    if paper_pos.shape[0] != cite_pos.shape[0] or paper_pos.shape[0] != cite_neg.shape[0]:
        paper_pos = paper_pos[:cite_neg.shape[0]]
        cite_pos = cite_pos[:cite_neg.shape[0]]

    pos_score, _ = self.distance(torch.cat([paper_pos, cite_pos],
                                           dim=-1))  # self.distance(self.hadamad_distance(pos_pool, updated_pos_pool))
    neg_score, _ = self.distance(torch.cat([paper_pos, cite_neg],
                                           dim=-1))  # self.distance(self.hadamad_distance(neg_pool, updated_neg_pool))

    return pos_score, neg_score

  def cos_similarity(self, x1, x2):
      return x1*x2

  def hadamad_distance(self, x1, x2):
      return torch.pow(x1-x2, 2)

  # def node_classification(self, event_graph, neighbor_graph):
  #   batch = event_graph.collect('batch')
  #     # mask = event_graph.collect('node_mask')
  #     # n_samples = len(source_nodes)
  #   event_node_embedding, updated_event_node_embedding, _ = self.compute_temporal_embeddings(
  #         event_graph, neighbor_graph)
  #
  #   event_rep = 0
  #   for node_type in self.metadata[0]:
  #       event_rep += global_mean_pool(event_node_embedding[node_type], batch[node_type]).detach()
  #
  #   out = self.classification_layer(event_rep)
  #
  #   return out
  #
  # def get_neg_memory(self, nodes, node_type = None):
  #     to_update_node_ids, unique_messages, unique_timestamps = \
  #         self.message_aggregator.aggregate(
  #             nodes,
  #             self.memory.neg_messages[node_type],
  #             node_type)
  #     # unique_aug_messages = self.memory.generate_aug_messages(unique_nodes)
  #     np.random.shuffle(to_update_node_ids)
  #     if len(to_update_node_ids) > 0:
  #         unique_messages = self.message_function.compute_message(unique_messages, node_type)
  #         # unique_aug_messages = self.message_function.compute_message(unique_aug_messages)
  #
  #     updated_memory, updated_last_update = self.memory_updater.get_updated_memory(nodes, to_update_node_ids,
  #                                                                                  unique_messages, node_type,
  #                                                                                  unique_timestamps)
  #
  #     return updated_memory, updated_last_update

  def update_neg_memory(self, nodes, node_type = None):
    # Aggregate messages for the same nodes
    to_update_node_ids, unique_messages, unique_timestamps = \
        self.message_aggregator.aggregate(
            nodes,
            self.memory.neg_messages[node_type],
            node_type)

    # unique_aug_messages = self.memory.generate_aug_messages(unique_nodes)

    if len(to_update_node_ids) > 0:
      unique_messages = self.message_function.compute_message(unique_messages, node_type)
      # unique_aug_messages = self.message_function.compute_message(unique_aug_messages)

    # Update the memory with the aggregated messages

    self.memory_updater.update_neg_memory(nodes, to_update_node_ids, unique_messages, node_type,
                                      unique_timestamps)

    # self.memory_updater.update_aug_memory(unique_nodes, unique_aug_messages)

  def get_updated_neg_memory(self, nodes, node_type = None):
    # Aggregate messages for the same nodes
    to_update_node_ids, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        self.memory.neg_messages[node_type],
        node_type)
    # unique_aug_messages = self.memory.generate_aug_messages(unique_nodes)
    if len(to_update_node_ids) > 0:

      unique_messages = self.message_function.compute_message(unique_messages, node_type)
      # unique_aug_messages = self.message_function.compute_message(unique_aug_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_neg_memory(nodes, to_update_node_ids,
                                                                                 unique_messages, node_type,
                                                                                 unique_timestamps)

    return updated_memory, updated_last_update

  def update_memory(self, nodes, node_type = None):
    # Aggregate messages for the same nodes
    to_update_node_ids, unique_messages, unique_timestamps = \
        self.message_aggregator.aggregate(
            nodes,
            self.memory.messages[node_type],
            node_type)

    # unique_aug_messages = self.memory.generate_aug_messages(unique_nodes)

    if len(to_update_node_ids) > 0:
      unique_messages = self.message_function.compute_message(unique_messages, node_type)
      # unique_aug_messages = self.message_function.compute_message(unique_aug_messages)

    # Update the memory with the aggregated messages

    self.memory_updater.update_memory(nodes, to_update_node_ids, unique_messages, node_type,
                                      unique_timestamps)

    # self.memory_updater.update_aug_memory(unique_nodes, unique_aug_messages)

  def get_updated_memory(self, nodes, node_type = None):
    # Aggregate messages for the same nodes
    to_update_node_ids, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        self.memory.messages[node_type],
        node_type)
    # unique_aug_messages = self.memory.generate_aug_messages(unique_nodes)
    if len(to_update_node_ids) > 0:

      unique_messages = self.message_function.compute_message(unique_messages, node_type)
      # unique_aug_messages = self.message_function.compute_message(unique_aug_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(nodes, to_update_node_ids,
                                                                                 unique_messages, node_type,
                                                                                 unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, node_x, node_mask, node_negative, node_timestamp, event_local_message, negative = False, deliver_to = 'all'):
    # event_times = torch.from_numpy(event_times).float().to(self.device)
    raw_message_list = {}
    # negative_message_list = {}
    # unique_node_list = {}

    masks = {}
    negatives = {}
    nodes = {}
    messages = event_local_message.copy()
    node_to_time_delta = node_timestamp.copy()
    node_to_time_encoding = {}

    neg = 1
    # if negative:
    #     neg = 1

    for node_type in self.metadata[0]:
        masks[node_type] = node_mask[node_type].data.to('cpu').numpy()
        negatives[node_type] = node_negative[node_type].data.to('cpu').numpy()
        nodes[node_type] = node_x[node_type]
        if deliver_to == 'all':
            masks[node_type] = np.logical_and(masks[node_type]>=0, negatives[node_type]!=neg)
        elif deliver_to == 'self':
            masks[node_type] = np.logical_and(masks[node_type]<2, negatives[node_type]!=neg)
        else:
            masks[node_type] = np.logical_and(masks[node_type]==2, negatives[node_type]!=neg)

    for node_type in self.metadata[0]:
        if not negative:
            # node_to_time_delta[node_type] = node_to_time_delta[node_type][masks[node_type]] - self.memory.last_update[node_type][nodes[node_type][masks[node_type]]]
            # node_to_time_encoding[node_type] = self.time_encoder(node_to_time_delta[node_type]).view(len(
            #     nodes[node_type][masks[node_type]]), -1)
            # messages[node_type] = torch.cat([messages[node_type][masks[node_type]], node_to_time_encoding[node_type]],
            #                                 dim=1)
            messages[node_type] = messages[node_type][masks[node_type]]#+node_to_time_encoding[node_type])

        else:
            node_to_time_delta[node_type] = node_to_time_delta[node_type][masks[node_type]] - \
                                            self.memory.last_update[node_type][nodes[node_type][masks[node_type]]]
            # node_to_time_delta[node_type] = node_to_time_delta[node_type][torch.randperm(len(node_to_time_delta[node_type]))]
            node_to_time_encoding[node_type] = self.time_encoder(node_to_time_delta[node_type]).view(len(
                nodes[node_type][masks[node_type]]), -1)#[torch.randperm(len(node_to_time_delta[node_type]))]
            shuffle_index = torch.randperm(messages[node_type][masks[node_type]].shape[1])
            # messages[node_type] = torch.cat([messages[node_type][masks[node_type]][:, shuffle_index], node_to_time_encoding[node_type]],
            #                                 dim=1)
            messages[node_type] = (messages[node_type][masks[node_type]]+node_to_time_encoding[node_type])[:, shuffle_index]#messages[node_type][masks[node_type]][:, shuffle_index]

        # raw_message_list[node_type] = defaultdict(list)
        type_node = nodes[node_type][masks[node_type]]#.data.to('cpu').numpy().tolist()
        # if len(type_node) == 400:
        #     type_node = type_node[::2]
        t1 = time.time()
        # for idx, node in enumerate(type_node):
        if not negative:
            self.memory.set_memory(type_node, messages[node_type].detach(), node_type)
            # self.memory.set_memory(node, messages[node_type][idx], node_type)
            # self.memory.memory[node_type][node, :] = messages[node_type][idx]
        else:
            self.memory.set_neg_memory(type_node, messages[node_type].detach(), node_type)
                # self.memory.neg_memory[node_type][node, :] = messages[node_type][idx].detach()
            # raw_message_list[node_type][node].append([messages[node_type][idx], neighbor_graph.collect('timestamp')[node_type][idx]])
        t2 = time.time()

    del node_to_time_encoding
    del messages
        # print(node_type,t2-t1)
    return None

