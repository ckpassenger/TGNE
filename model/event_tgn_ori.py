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


class Decoder(torch.nn.Module):
    # def __init__(self, dim, out_dim, drop=0.1):
    #   super().__init__()
    #   self.fc_1 = torch.nn.Linear(dim, 50)
    #   self.fc_2 = torch.nn.Linear(50, out_dim)
    #   self.layer_norm = torch.nn.BatchNorm1d(100)
    #   self.act = torch.nn.ReLU()
    #   self.dropout = torch.nn.Dropout(p=drop, inplace=False)
    #
    # def forward(self, x):
    #   x1 = x = self.dropout(self.act(self.fc_1(x)))
    #   return self.fc_2(x), x1#.squeeze(dim=1)
    #
    # def reset_parameters(self):
    #   self.fc_1.reset_parameters()
    #   self.fc_2.reset_parameters()
    #   self.fc_3.reset_parameters()
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, dim)
        self.fc_2 = torch.nn.Linear(dim, dim)
        self.fc_3 = torch.nn.Linear(dim, 5)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x)  # .squeeze(dim=1)


class decoder(torch.nn.Module):
    def __init__(self, dim, out_dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 64)
        self.fc_2 = torch.nn.Linear(64, 16)
        self.fc_3 = torch.nn.Linear(16, out_dim)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x1 = x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x), x1  # .squeeze(dim=1)

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()


class TGN(torch.nn.Module):
    def __init__(self, n_nodes, device, edge_feature, n_layers=4,
                 n_heads=2, dropout=0.0, use_memory=True,
                 message_dimension=32, embedding_dimension=32, num_labels=2,
                 memory_dimension=500, embedding_module_type="han",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, aggregator_type="last",
                 memory_updater_type="rnn",
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
        self.time_encoder = TimeEncode(dimension=self.embedding_dimension)
        self.memory = None
        #
        # self.mean_time_shift_src = mean_time_shift_src
        # self.std_time_shift_src = std_time_shift_src
        # self.mean_time_shift_dst = mean_time_shift_dst
        # self.std_time_shift_dst = std_time_shift_dst

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
                                                         message_dimension=embedding_dimension,
                                                         metadata=metadata)

            self.message_constructor = get_embedding_module(module_type=embedding_module_type,
                                                            memory=self.memory,
                                                            time_encoder=self.time_encoder,
                                                            n_layers=2,
                                                            n_time_features=self.embedding_dimension,
                                                            embedding_dimension=self.embedding_dimension,
                                                            device=self.device,
                                                            n_heads=n_heads, dropout=dropout,
                                                            use_memory=use_memory,
                                                            metadata=metadata)

            self.message_aggregator = get_message_aggregator('last', device=self.device)

            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=embedding_dimension,
                                                     memory_dimension=self.embedding_dimension,
                                                     device=device, metadata=metadata)

        self.embedding_module_type = embedding_module_type

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     memory=self.memory,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=1,
                                                     n_time_features=self.embedding_dimension,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     metadata=metadata)

        self.lin = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin[node_type] = torch.nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        self.edge_feature = torch.from_numpy(edge_feature).float().to(self.device)

        self.edge_feature_linear = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type_ = '__'.join(edge_type)
            self.edge_feature_linear[edge_type_] = torch.nn.Linear(edge_feature.shape[1], embedding_dimension)

        self.distance = decoder(2 * self.embedding_dimension, 1,
                                drop=0)  # torch.nn.Linear(2*embedding_dimension, 1)#torch.nn.ModuleDict()

        self.negative_weight = 1.0

        self.last_neg = []

    def compute_temporal_embeddings(self, event_graph):
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

        # extract information from input graph
        # node information
        node_x = event_graph.collect('x')  # for feature & memory
        node_mask = event_graph.collect(
            'node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
        node_timestamp = event_graph.collect('timestamp')

        # edge information
        edge_index = event_graph.collect('edge_index_dict')
        edge_rel_times = event_graph.collect('edge_rel_times')
        edge_idx = event_graph.collect('edge_idxs')  # for feature
        edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

        # node memory
        node_memory = {}
        neg_node_memory = {}  # for negative sample

        # get_updated node memory
        for node_type in self.metadata[0]:
            node_memory[node_type], _ = self.get_updated_memory(node_x[node_type], node_type)
            neg_node_memory[node_type], _ = self.get_updated_neg_memory(node_x[node_type], node_type)

        # get neighbor edges & nodes for graph embedding
        neighbor_node_memory_dict = {}
        neg_neighbor_node_memory_dict = {}
        neighbor_node_timestamp_dict = {}

        neighbor_edge_index = {}
        neighbor_edge_idx = {}
        neighbor_edge_rel_times = {}

        for node_type in self.metadata[0]:
            neighbor_node_memory_dict[node_type] = node_memory[node_type][node_mask[node_type] == 1]
            neg_neighbor_node_memory_dict[node_type] = neg_node_memory[node_type][node_mask[node_type] == 1]
            neighbor_node_timestamp_dict[node_type] = node_timestamp[node_type][node_mask[node_type] != 1]

        for edge_type in self.metadata[1]:
            neighbor_edge_index[edge_type] = edge_index[edge_type][edge_mask[node_type] == 1]
            neighbor_edge_idx[edge_type] = edge_idx[edge_type][edge_mask[node_type] == 1]
            neighbor_edge_rel_times[edge_type] = edge_rel_times[edge_type][edge_mask[node_type] == 1]

        # graph embedding
        node_embedding = self.embedding_module.compute_embedding(node_memory, neighbor_edge_index,
                                                                 node_timestamp, neighbor_edge_idx,
                                                                 neighbor_edge_rel_times)
        neg_node_embedding = self.embedding_module.compute_embedding(neg_node_memory, neighbor_edge_index,
                                                                     node_timestamp, neighbor_edge_idx,
                                                                     neighbor_edge_rel_times)

        # message computing (updated event node embedding)
        event_edge_index = {}
        event_edge_idx = {}
        event_edge_rel_times = {}

        for edge_type in self.metadata[1]:
            event_edge_index[edge_type] = edge_index[edge_type][edge_mask[node_type] == 0]
            event_edge_idx[edge_type] = edge_idx[edge_type][edge_mask[node_type] == 0]
            event_edge_rel_times[edge_type] = edge_rel_times[edge_type][edge_mask[node_type] == 0]

        updated_event_node_embedding = self.embedding_module.compute_embedding(node_embedding, event_edge_index,
                                                                               node_timestamp, event_edge_idx,
                                                                               event_edge_rel_times)
        neg_updated_event_node_embedding = self.embedding_module.compute_embedding(neg_node_embedding, event_edge_index,
                                                                                   node_timestamp, event_edge_idx,
                                                                                   event_edge_rel_times)

        # message transform
        node_message = self.embedding_module.compute_embedding(updated_event_node_embedding, neighbor_edge_index,
                                                               node_timestamp, neighbor_edge_idx,
                                                               neighbor_edge_rel_times)
        neg_node_message = self.embedding_module.compute_embedding(neg_updated_event_node_embedding,
                                                                   neighbor_edge_index,
                                                                   node_timestamp, neighbor_edge_idx,
                                                                   neighbor_edge_rel_times)

        # message delivering
        for node_type in self.metadata[0]:
            self.update_memory(node_x[node_type], node_type)
            self.update_neg_memory(node_x[node_type], node_type)
            self.memory.clear_messages(node_x[node_type].cpu().numpy().tolist(), node_type)

        node_id_to_messages = self.get_raw_messages(event_graph, node_message)
        node_id_to_neg_messages = self.get_raw_messages(event_graph, neg_node_message, negative=True)

        for node_type in self.metadata[0]:
            self.memory.store_raw_messages(node_id_to_messages[node_type].keys(), node_id_to_messages[node_type],
                                           node_type)
            self.memory.store_neg_messages(node_id_to_neg_messages[node_type].keys(),
                                           node_id_to_neg_messages[node_type],
                                           node_type)

        event_edge_dict = {}

        batch = event_graph.collect('batch')

        for edge_type in self.metadata[1]:
            edge_type_ = '__'.join(edge_type)
            # print(edge_type)
            neighbor_edge_dict[
                edge_type] = 0  # self.edge_feature_linear[edge_type_](self.edge_feature[neighbor_edge_idx_dict[edge_type]])
            # shuffle_index = torch.randperm(
            #     neighbor_graph.collect('edge_rel_times')[edge_type].shape[0])
            # neg_neighbor_edge_dict[edge_type] = 0#neighbor_edge_dict[edge_type][shuffle_index]
            # neg_neighbor_edge_index_dict[edge_type] = neighbor_graph.edge_index_dict[edge_type][:, shuffle_index]
            # neg_neighbor_edge_rel_time_dict[edge_type] = neighbor_graph.collect('edge_rel_times')[edge_type][shuffle_index]#+torch.randint_like(neighbor_graph.collect('edge_rel_times')[edge_type], high=10)#[shuffle_index]

            event_edge_dict[
                edge_type] = 0  # self.edge_feature_linear[edge_type_](self.edge_feature[event_edge_idx_dict[edge_type]])
        event_node_timestamp = {}
        if self.use_memory:
            for node_type in self.metadata[0]:
                neighbor_memory_dict[node_type], _ = self.get_updated_memory(event_x_dict[node_type], node_type)
                neg_neighbor_memory_dict[node_type], _ = self.get_updated_neg_memory(event_x_dict[node_type], node_type)
                shuffle_index = torch.randperm(
                    event_graph.collect('timestamp')[node_type].shape[0])
                event_node_timestamp[node_type] = event_graph.collect('timestamp')[node_type][
                    shuffle_index]  # +torch.randint_like(neighbor_graph.collect('timestamp')[node_type], high=10)#[shuffle_index]
                # print(neighbor_graph.collect('timestamp')[node_type])
                # print(torch.mean(neighbor_memory_dict[node_type]), torch.mean(neg_neighbor_memory_dict[node_type]))
                # if torch.sum(event_graph.collect('node_mask')[node_type] == 1) > 0:
                # neg_neighbor_memory_dict[node_type][neighbor_graph.collect('node_mask')[node_type]==0] = neighbor_memory_dict[node_type][neighbor_graph.collect('node_mask')[node_type]==0]
                # event_memory_dict[node_type], _ = self.get_updated_memory(event_x_dict[node_type], node_type)
        # Compute the embeddings using the embedding module

        node_embedding = self.embedding_module.compute_embedding(neighbor_memory_dict, event_graph.edge_index_dict,
                                                                 event_graph.collect('timestamp'), neighbor_edge_dict,
                                                                 event_graph.collect('edge_rel_times'))
        neg_node_embedding = self.embedding_module.compute_embedding(neg_neighbor_memory_dict,
                                                                     event_graph.edge_index_dict,
                                                                     event_graph.collect('timestamp'),
                                                                     neighbor_edge_dict,
                                                                     event_graph.collect('edge_rel_times'))
        # neg_node_embedding = self.embedding_module.compute_embedding(neg_neighbor_memory_dict, neg_neighbor_edge_index_dict, event_node_timestamp, neg_neighbor_edge_dict, neg_neighbor_edge_rel_time_dict)

        event_memory_dict = {}
        event_node_embedding = {}
        negative_event_memory_list = [{} for _ in range(1)]
        negative_event_node_embedding_list = [{} for _ in range(1)]

        for node_type in self.metadata[0]:
            event_node_embedding[node_type] = node_embedding[node_type][
                neighbor_graph.collect('node_mask')[node_type] == 1]  # .clone() # for self-supervise, event_nodes
            event_memory_dict[node_type] = node_embedding[node_type][neighbor_graph.collect('node_mask')[
                                                                         node_type] == 1].clone()  # torch.tanh(self.lin[node_type]())#.clone() # for self-supervise, event_nodes

            for _ in range(1):
                negative_event_node_embedding_list[_][node_type] = node_embedding[node_type][
                    neighbor_graph.collect('node_mask')[node_type] == 1].clone()
                # negative_event_memory_list[_][node_type] = event_node_embedding[node_type].clone()#+0.5*torch.randn_like(event_memory_dict[node_type]).detach() # for self-supervise
                negative_event_memory_list[_][node_type] = node_embedding[node_type][
                    neighbor_graph.collect('node_mask')[
                        node_type] == 1].clone()  # +0.5*torch.randn_like(event_memory_dict[node_type]).detach() # for self-supervise
                if torch.sum(event_graph.collect('node_mask')[node_type] == 1) > 0:
                    shuffle_index = torch.randperm(negative_event_memory_list[_][node_type][
                                                       event_graph.collect('node_mask')[node_type] == 1].shape[0])
                    # negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1] = negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][shuffle_index]
                    negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type] == 1] = \
                    node_embedding[node_type][neighbor_graph.collect('node_mask')[node_type] == 1][
                        event_graph.collect('node_mask')[node_type] == 1][shuffle_index]
                # negative_event_node_embedding_list[_][node_type] = negative_event_memory_list[_][node_type]
                # with torch.no_grad():
                #     random_score = torch.rand_like(negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][:,0])
                #     # print(scatter_max(random_score, batch[node_type][event_graph.collect('node_mask')[node_type]==1]))
                #     score_max = scatter_max(random_score, batch[node_type][event_graph.collect('node_mask')[node_type]==1])[0].index_select(0, batch[node_type][event_graph.collect('node_mask')[node_type]==1])
                #     perm = (random_score == score_max)
                #     shuffle_index = torch.roll(torch.arange(negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][perm].shape[0]), 4, 0)#torch.randperm(negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][perm].shape[0])
                #     # shuffle_index2 = torch.roll(torch.arange(negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1].shape[0]), 4, 0)#torch.randperm(negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][perm].shape[0])
                #     # shuffle_index = torch.randperm(negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][perm].shape[0])
                # negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][perm] = negative_event_memory_list[_][node_type][event_graph.collect('node_mask')[node_type]==1][perm][shuffle_index]#+0.1*torch.randn_like(negative_event_memory_list[_][node_type][perm])#.unsqueeze(1)

            event_node_embedding[node_type] = node_embedding[node_type][
                neighbor_graph.collect('node_mask')[node_type] == 1]  # for self-supervise

        updated_event_node_embedding = self.embedding_module.compute_embedding(event_node_embedding,
                                                                               event_graph.edge_index_dict,
                                                                               event_graph.collect('timestamp'),
                                                                               event_edge_dict)
        # updated_event_node_embedding = self.embedding_module.compute_embedding(updated_event_node_embedding, event_graph.edge_index_dict, event_graph.collect('timestamp'), event_edge_dict)

        negative_node_embedding_ = [{} for _ in range(1)]
        negative_updated_node_embedding_ = [{} for _ in range(1)]
        negative_node_embedding_list = defaultdict(list)
        updated_negative_node_embedding_list = defaultdict(list)
        negative_updated_node_embedding_list = defaultdict(list)  # for negative msg

        negative_node_embedding = {}
        updated_negative_node_embedding = {}
        negative_updated_node_embedding = {}  # for neg msg
        for _ in range(1):
            negative_node_embedding_[_] = self.embedding_module.compute_embedding(negative_event_node_embedding_list[_],
                                                                                  event_graph.edge_index_dict,
                                                                                  event_graph.collect('timestamp'),
                                                                                  event_edge_dict)

            negative_updated_node_embedding_[_] = self.embedding_module.compute_embedding(negative_event_memory_list[_],
                                                                                          event_graph.edge_index_dict,
                                                                                          event_graph.collect(
                                                                                              'timestamp'),
                                                                                          event_edge_dict)
            # negative_node_embedding_[_] = self.embedding_module.compute_embedding(negative_node_embedding_[_], event_graph.edge_index_dict, event_graph.collect('timestamp'), event_edge_dict)
            for node_type in self.metadata[0]:
                negative_node_embedding_list[node_type].append(
                    negative_event_node_embedding_list[_][node_type])  # torch.tanh(self.lin[node_type]())
                updated_negative_node_embedding_list[node_type].append(negative_node_embedding_[_][node_type])
                negative_updated_node_embedding_list[node_type].append(negative_updated_node_embedding_[_][node_type])

        for node_type in self.metadata[0]:
            negative_node_embedding[node_type] = torch.cat(negative_node_embedding_list[node_type], dim=1)
            updated_negative_node_embedding[node_type] = torch.cat(updated_negative_node_embedding_list[node_type],
                                                                   dim=1)
            negative_updated_node_embedding[node_type] = torch.cat(negative_updated_node_embedding_list[node_type],
                                                                   dim=1)

        with torch.no_grad():
            message_to_deliver = {}  # neighbor_memory_dict.copy()
            neg_message_to_deliver = {}  # neighbor_memory_dict.copy()
            for node_type in self.metadata[0]:
                message_to_deliver[node_type] = neighbor_memory_dict[node_type].data.clone()  # set neighbor to 0
                message_to_deliver[node_type][neighbor_graph.collect('node_mask')[node_type] == 1] = \
                updated_event_node_embedding[node_type].data.clone()
                # neg_message_to_deliver[node_type] = neighbor_memory_dict[node_type].data.clone()
                neg_message_to_deliver[node_type] = neg_neighbor_memory_dict[
                    node_type].data.clone()  # neighbor_memory_dict[node_type].data.clone()
                # neg_message_to_deliver[node_type][neighbor_graph.collect('node_mask')[node_type]==1] = negative_updated_node_embedding[node_type].data.clone()
                #
                # if torch.sum(event_graph.collect('node_mask')[node_type] == 1) > 0:
                # #     # random_score = torch.randn_like(negative_updated_node_embedding[node_type][:, 0]).view(-1)
                # #     # score_max = scatter_max(random_score, batch[node_type])[0].index_select(0, batch[node_type])
                # #     # perm = (random_score == score_max).view(-1)
                # #     # shuffle_index = torch.randperm(negative_updated_node_embedding[node_type][perm].shape[0])
                # #     # neg_message_to_deliver[node_type][neighbor_graph.collect('node_mask')[node_type]==1][perm] = negative_updated_node_embedding[node_type][perm][
                # #     #     shuffle_index].clone()
                # # #
                # # #
                #     shuffle_index = torch.randperm(updated_event_node_embedding[node_type].shape[0])
                neg_message_to_deliver[node_type][neighbor_graph.collect('node_mask')[node_type] == 1] = \
                negative_updated_node_embedding[node_type].data.clone()

            neighbor_delivered_message = self.embedding_module.compute_embedding(message_to_deliver,
                                                                                 neighbor_graph.edge_index_dict,
                                                                                 neighbor_graph.collect('timestamp'),
                                                                                 neighbor_edge_dict,
                                                                                 neighbor_graph.collect(
                                                                                     'edge_rel_times'))
            neg_neighbor_delivered_message = self.embedding_module.compute_embedding(neg_message_to_deliver,
                                                                                     neg_neighbor_edge_index_dict,
                                                                                     neighbor_graph.collect(
                                                                                         'timestamp'),
                                                                                     neighbor_edge_dict,
                                                                                     neighbor_graph.collect(
                                                                                         'edge_rel_times'))
            # neg_neighbor_delivered_message = self.embedding_module.compute_embedding(neg_message_to_deliver, neg_neighbor_edge_index_dict, event_node_timestamp, neg_neighbor_edge_dict, neg_neighbor_edge_rel_time_dict)

            # for node_type in self.metadata[0]:
            #     neighbor_delivered_message[node_type][neighbor_graph.collect('node_mask')[node_type]==1] = updated_event_node_embedding[node_type].data.clone()  # event node message
            #     # shuffle_index = torch.randperm(neg_neighbor_delivered_message[node_type].shape[0])
            #     # neg_neighbor_delivered_message[node_type] = neg_neighbor_delivered_message[node_type][shuffle_index]
            #     neg_neighbor_delivered_message[node_type][neighbor_graph.collect('node_mask')[node_type]==1] = neg_message_to_deliver[node_type][neighbor_graph.collect('node_mask')[node_type]==1].data.clone()  # event node message

            # TODO: negative sample

            if self.use_memory:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                for node_type in self.metadata[0]:
                    self.update_memory(neighbor_x_dict[node_type], node_type)
                    self.update_neg_memory(neighbor_x_dict[node_type], node_type)
                    self.memory.clear_messages(neighbor_x_dict[node_type].cpu().numpy().tolist(), node_type)
                    # assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-2), \
                    #   str(torch.mean(memory[positives])) + str(torch.mean(self.memory.get_memory(positives))) + "Something wrong in how the memory was updated"

                    # Remove messages for the positives since we have already updated the memory using them
                    self.update_memory(event_x_dict[node_type], node_type)
                    self.update_neg_memory(event_x_dict[node_type], node_type)
                    self.memory.clear_messages(event_x_dict[node_type].cpu().numpy().tolist(), node_type)
                node_id_to_messages = self.get_raw_messages(neighbor_graph, neighbor_delivered_message)
                node_id_to_neg_messages = self.get_raw_messages(neighbor_graph, neg_neighbor_delivered_message,
                                                                negative=True)

                for node_type in self.metadata[0]:
                    self.memory.store_raw_messages(node_id_to_messages[node_type].keys(),
                                                   node_id_to_messages[node_type], node_type)
                    self.memory.store_neg_messages(node_id_to_neg_messages[node_type].keys(),
                                                   node_id_to_neg_messages[node_type], node_type)

        return event_memory_dict, updated_event_node_embedding, negative_node_embedding, negative_updated_node_embedding

    def compute_event_probabilities(self, event_graph, neighbor_graph):
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
        batch = event_graph.collect('batch')
        mask = event_graph.collect('node_mask')

        event_x_dict = event_graph.collect('x')
        event_memory_dict = {}
        neg_event_memory_dict = {}
        for node_type in self.metadata[0]:
            event_memory_dict[node_type], _ = self.get_updated_memory(event_x_dict[node_type], node_type)
            neg_event_memory_dict[node_type], _ = self.get_updated_neg_memory(event_x_dict[node_type], node_type)

        event_node_embedding, updated_event_node_embedding, negative_node_embedding, updated_negative_node_embedding = self.compute_temporal_embeddings(
            event_graph, neighbor_graph)
        # pos_score = 0
        # neg_score = 0

        pos_pool, updated_pos_pool = 0, 0
        pos_reg, neg_reg = 0, 0
        updated_pos_reg, updated_neg_reg = 0, 0
        neg_pool, updated_neg_pool = 0, 0
        for node_type in self.metadata[0]:
            # if torch.sum(mask[node_type]==0) > 0:
            #     pos_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 0], batch[node_type][mask[node_type] == 0])
            #     neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 0], batch[node_type][mask[node_type] == 0])
            #
            # if torch.sum(mask[node_type]==1) > 0:
            #     updated_pos_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])
            #     shuffle_index = torch.randperm(updated_pos_pool.shape[0])
            #     # updated_neg_pool += global_mean_pool(negative_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])[shuffle_index]#
            #     updated_neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])[shuffle_index]
            # pos_pool += global_mean_pool(torch.tanh(self.lin[node_type](event_node_embedding[node_type].detach())), batch[node_type])
            pos_pool += global_mean_pool(event_node_embedding[node_type], batch[node_type])
            updated_pos_pool += global_mean_pool(updated_event_node_embedding[node_type], batch[node_type])
            # neg_pool += global_mean_pool(torch.tanh(self.lin[node_type](negative_node_embedding[node_type].detach())), batch[node_type])
            neg_pool += global_mean_pool(negative_node_embedding[node_type], batch[node_type])
            # shuffle_index = torch.randperm(event_node_embedding[node_type][mask[node_type]==1].shape[0])
            # negative_event_memory_list[_][node_type] = negative_event_memory_list[_][node_type][shuffle_index]
            updated_neg_pool += global_mean_pool(updated_negative_node_embedding[node_type], batch[node_type])

            pos_reg += global_mean_pool(event_memory_dict[node_type], batch[node_type])
            updated_pos_reg += global_mean_pool(event_node_embedding[node_type], batch[node_type])
            neg_reg += global_mean_pool(neg_event_memory_dict[node_type], batch[node_type])
            updated_neg_reg += global_mean_pool(negative_node_embedding[node_type], batch[node_type])

        pos_score, _ = self.distance(torch.cat([pos_pool, updated_pos_pool],
                                               dim=-1))  # self.distance(self.hadamad_distance(pos_pool, updated_pos_pool))
        neg_score, _ = self.distance(torch.cat([neg_pool, updated_neg_pool],
                                               dim=-1))  # self.distance(self.hadamad_distance(neg_pool, updated_neg_pool))
        # pos_score = torch.sum(torch.cat([pos_pool, updated_pos_pool, pos_pool-updated_pos_pool, pos_pool*updated_pos_pool],dim=-1), dim=-1).view(-1,1)
        # neg_score = torch.sum(torch.cat([neg_pool, updated_neg_pool, neg_pool-updated_neg_pool, neg_pool*updated_neg_pool],dim=-1), dim=-1).view(-1,1)

        # if len(self.last_neg) < 5:
        #     updated_neg_pool = updated_pos_pool
        # else:
        #     updated_neg_pool = self.last_neg[random.randint(0, len(self.last_neg)-1)]
        #     while updated_neg_pool.shape[0] != updated_pos_pool.shape[0]:
        #         updated_neg_pool = self.last_neg[random.randint(0, len(self.last_neg) - 1)]
        # self.last_neg.append(updated_pos_pool.detach())

        shuffle_index = torch.randperm(updated_pos_pool.shape[0])
        # pos_score = torch.sum(
        #     pos_pool * updated_pos_pool,
        #     dim=-1).view(-1, 1)
        # neg_score = torch.sum(
        #     neg_pool * updated_neg_pool,
        #     dim=-1).view(-1, 1)

        pos_score_reg = torch.sum(
            pos_reg * updated_pos_reg,
            dim=-1).view(-1, 1)
        neg_score_reg = torch.sum(
            neg_reg * updated_neg_reg,
            dim=-1).view(-1, 1)

        # pos_score += global_mean_pool((self.distance[node_type](self.hadamad_distance(event_node_embedding[node_type],updated_event_node_embedding[node_type]))), batch[node_type])
        # neg_score += global_mean_pool((self.distance[node_type](self.hadamad_distance(negative_node_embedding[node_type], updated_negative_node_embedding[node_type]))), batch[node_type])

        return pos_score, neg_score, pos_score_reg, neg_score_reg

    def cos_similarity(self, x1, x2):
        return x1 * x2

    def hadamad_distance(self, x1, x2):
        return torch.pow(x1 - x2, 2)

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

    def update_neg_memory(self, nodes, node_type=None):
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

    def get_updated_neg_memory(self, nodes, node_type=None):
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

    def update_memory(self, nodes, node_type=None):
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

    def get_updated_memory(self, nodes, node_type=None):
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

    def get_raw_messages(self, neighbor_graph, event_local_message, negative=False, deliver_to='all'):
        # event_times = torch.from_numpy(event_times).float().to(self.device)
        raw_message_list = {}
        # negative_message_list = {}
        # unique_node_list = {}

        masks = neighbor_graph.collect('node_mask')
        nodes = neighbor_graph.collect('x')
        messages = event_local_message.copy()
        node_to_time_delta = neighbor_graph.collect('timestamp').copy()
        node_to_time_encoding = {}

        for node_type in self.metadata[0]:
            masks[node_type] = masks[node_type].data.to('cpu').numpy()
            nodes[node_type] = nodes[node_type].data.to('cpu').numpy()

            if deliver_to == 'all':
                masks[node_type] = (masks[node_type] >= 0)
            elif deliver_to == 'self':
                masks[node_type] = (masks[node_type] == 1)
            else:
                masks[node_type] = (masks[node_type] == 0)

            # print(np.sum(masks[node_type]))

        for node_type in self.metadata[0]:
            node_to_time_delta[node_type] = node_to_time_delta[node_type][masks[node_type]] - \
                                            self.memory.last_update[node_type][nodes[node_type][masks[node_type]]]
            if negative:
                node_to_time_delta[node_type] = node_to_time_delta[node_type][
                    torch.randperm(len(node_to_time_delta[node_type]))]
            node_to_time_encoding[node_type] = self.time_encoder(node_to_time_delta[node_type]).view(len(
                nodes[node_type][masks[node_type]]), -1)
            messages[node_type] = torch.cat([messages[node_type][masks[node_type]], node_to_time_encoding[node_type]],
                                            dim=1)

            raw_message_list[node_type] = defaultdict(list)
            for idx, node in enumerate(nodes[node_type][masks[node_type]].tolist()):
                raw_message_list[node_type][node].append(
                    [messages[node_type][idx], neighbor_graph.collect('timestamp')[node_type][idx]])
        return raw_message_list

