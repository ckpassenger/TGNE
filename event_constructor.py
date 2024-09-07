import random

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
import gc
# class Graph_Constructor():
#     def __init__(self, meta_data, without_types = -1):
#         self.meta_data = meta_data
#         self.without_types = without_types
#
#         self.node_type_batch_num =
#
#         for node_type in self.meta_data['node_types'].values():
#
#
#     def set_node_remap_batch_id(self):
#
#         for node_type in self.meta_data['node_types'].values():
#             if node_type == self.meta_data['edge_types'][self.without_types][1]:
#                 continue
#
#
#
#     def initial_graph_dict(self):
#         self.node_dict = {}  # store nodes
#         self.node_mask_dict = {}  # store nodes
#         self.edge_index_dict = {}
#         self.edge_rel_timestamp_dict = {}
#         self.edge_idxs_dict = {}
#         self.node_remap_dict = {}  # store remap result
#         self.edge_type_map_dict = {}
#
#         for node_type in self.meta_data['node_types'].values():
#             if node_type == self.meta_data['edge_types'][self.without_types][1]:
#                 continue
#             self.node_remap_dict[node_type] = []
#             self.node_dict[node_type] = []
#             self.node_mask_dict[node_type] = []
#
#         for type_id, edge_types in self.meta_data['edge_types'].items():
#
#             if edge_types == self.meta_data['edge_types'][self.without_types]:
#                 continue
#
#             if edge_types[0] is None:
#                 break
#
#             edge_type0 = edge_types[0] + '_0_' + edge_types[1]  # direction 0 hyperedge - > node
#             edge_type1 = edge_types[1] + '_1_' + edge_types[0]  # direction 1 node - > hyperedge
#
#             self.edge_type_map_dict[2 * type_id] = [edge_type0, edge_types]
#             self.edge_type_map_dict[2 * type_id + 1] = [edge_type1, [edge_types[1], edge_types[0]]]
#
#             self.edge_index_dict[edge_type0] = []
#             self.edge_index_dict[edge_type1] = []
#             self.edge_rel_timestamp_dict[edge_type0] = []
#             self.edge_rel_timestamp_dict[edge_type1] = []
#             self.edge_idxs_dict[edge_type0] = []
#             self.edge_idxs_dict[edge_type1] = []
#
#     def remap_node_id(self, Heterograph, event_timestamp, edges, nodes):
#         self.initial_graph_dict()
#
#         for edge in edges:
#             if len(edge) == 2:  # meta graph edges
#
#                 source, node_type = edge[0], edge[1]
#
#                 if node_type == self.meta_data['edge_types'][self.without_types][1]:
#                     continue
#
#                 if source not in self.node_remap_dict[node_type].keys():
#                     self.node_remap_dict[node_type][source] = len(self.node_dict[node_type])
#                     self.node_dict[node_type].append(source.split('_')[-1])
#
#                     self.node_mask_dict[node_type].append(1)
#
#             else:
#                 if len(edge) == 6:  # neighbor graph edges
#                     source, destination, timestamp, edge_idx, edge_type, edge_direction = edge[0], edge[1], edge[2], \
#                                                                                           edge[3], edge[4], edge[5]
#                     rel_timestamp = (event_timestamp - timestamp) / np.timedelta64(60 * 24, 'm')
#                 else:  # meta graph edges
#                     source, destination, timestamp, edge_idx, edge_type = edge[0], edge[1], edge[2], edge[3], edge[
#                         4]  # event/hyperedge nodes
#                     edge_direction = 0
#                     rel_timestamp = 0
#
#                 if edge_type == self.without_types:
#                     continue
#
#                 src_node_type, dst_node_type = self.meta_data['edge_types'][edge_type][edge_direction], \
#                                                self.meta_data['edge_types'][edge_type][1 - edge_direction]
#                 if source not in self.node_remap_dict[src_node_type].keys():
#                     self.node_remap_dict[src_node_type][source] = len(self.node_dict[src_node_type])
#                     self.node_dict[src_node_type].append(source.split('_')[-1])
#                     if source in nodes:
#                         self.node_mask_dict[src_node_type].append(1)
#                     else:
#                         self.node_mask_dict[src_node_type].append(0)
#
#                 if destination not in self.node_remap_dict[dst_node_type].keys():
#                     self.node_remap_dict[dst_node_type][destination] = len(self.node_dict[dst_node_type])
#                     self.node_dict[dst_node_type].append(destination.split('_')[-1])
#                     if destination in nodes:
#                         self.node_mask_dict[dst_node_type].append(1)
#                     else:
#                         self.node_mask_dict[dst_node_type].append(0)
#
#                 # add both directions
#                 edge_type0 = self.edge_type_map_dict[2 * edge_type + edge_direction][0]
#
#                 self.edge_index_dict[edge_type0].append(
#                     [self.node_remap_dict[src_node_type][source], self.node_remap_dict[dst_node_type][destination]])
#                 self.edge_rel_timestamp_dict[edge_type0].append(rel_timestamp)
#                 self.edge_idxs_dict[edge_type0].append(edge_idx)
#
#                 edge_type1 = self.edge_type_map_dict[2 * edge_type + (1 - edge_direction)][0]
#
#                 self.edge_index_dict[edge_type1].append(
#                     [self.node_remap_dict[dst_node_type][destination], self.node_remap_dict[src_node_type][source]])
#                 self.edge_rel_timestamp_dict[edge_type1].append(rel_timestamp)
#                 self.edge_idxs_dict[edge_type1].append(edge_idx)
#
#         for edge_types in self.edge_type_map_dict.values():
#             src_node_type, dst_node_type = edge_types[1][0], edge_types[1][1]  # [edge_type, [src_type, dst_type]]
#
#             Heterograph[src_node_type].x = torch.LongTensor(np.array(self.node_dict[src_node_type]).astype('int64'))
#             Heterograph[src_node_type].num_nodes = len(self.node_dict[src_node_type])
#             Heterograph[src_node_type].node_mask = torch.LongTensor(
#                 np.array(self.node_mask_dict[src_node_type]).astype('int64'))
#             Heterograph[src_node_type].timestamp = (
#                         torch.zeros_like(Heterograph[src_node_type].node_mask, dtype=torch.float) + (
#                             event_timestamp - np.datetime64('2003-01-01T00:00')) / np.timedelta64(60 * 24, 'm'))
#             Heterograph[dst_node_type].x = torch.LongTensor(np.array(self.node_dict[dst_node_type]).astype('int64'))
#             Heterograph[dst_node_type].num_nodes = len(self.node_dict[dst_node_type])
#             Heterograph[dst_node_type].node_mask = torch.LongTensor(
#                 np.array(self.node_mask_dict[dst_node_type]).astype('int64'))
#             Heterograph[dst_node_type].timestamp = (
#                         torch.zeros_like(Heterograph[dst_node_type].node_mask, dtype=torch.float) + (
#                             event_timestamp - np.datetime64('2003-01-01T00:00')) / np.timedelta64(60 * 24, 'm'))
#
#             edge_types_contain = edge_types[0].split('_')
#             edge_type = (edge_types_contain[0], edge_types_contain[1], edge_types_contain[2])
#
#             Heterograph[edge_type].edge_index = torch.LongTensor(
#                 np.array(self.edge_index_dict[edge_types[0]]).reshape([-1, 2]).transpose(1, 0).astype('int64'))
#             Heterograph[edge_type].edge_rel_times = torch.FloatTensor(
#                 np.array(self.edge_rel_timestamp_dict[edge_types[0]]).reshape(-1, 1).astype('float64'))
#             Heterograph[edge_type].edge_idxs = torch.LongTensor(
#                 np.array(self.edge_idxs_dict[edge_types[0]]).reshape(-1, 1).astype('int64'))
#         return Heterograph
import time
import threading
from multiprocessing import  Process
from queue import Queue
from collections import defaultdict



def get_neighbor_tree(neighbor_finder, event_center_nodes, timestamp, num_hops=1, without_types = -1):
    hop_center_nodes = [[event_center_nodes]]

    neighbor_sources, neighbor_destinations, neighbor_edge_types, neighbor_edge_idxs, \
    neighbor_edge_times, neighbor_edge_directions = [], [], [], [], [], []

    for hop in range(num_hops):
        next_hop_center_nodes = []
        center_nodes = hop_center_nodes[hop]

        node_neighbor_sources, node_neighbor_destinations, node_neighbor_edge_types, node_neighbor_edge_idxs, \
        node_neighbor_edge_times, node_neighbor_edge_directions = neighbor_finder.get_temporal_neighbor(center_nodes, timestamp, 20, without_types=without_types)
        neighbor_nodes = node_neighbor_destinations.tolist()

        next_hop_center_nodes.extend(neighbor_nodes)

        neighbor_sources.append(node_neighbor_sources)
        neighbor_destinations.append(node_neighbor_destinations)
        neighbor_edge_types.append(node_neighbor_edge_types)
        neighbor_edge_idxs.append(node_neighbor_edge_idxs)
        neighbor_edge_times.append(node_neighbor_edge_times)
        neighbor_edge_directions.append(node_neighbor_edge_directions)

        hop_center_nodes.append(next_hop_center_nodes.copy())

    neighbor_sources = np.concatenate(neighbor_sources, axis=-1)
    neighbor_destinations = np.concatenate(neighbor_destinations, axis=-1)
    neighbor_edge_types = np.concatenate(neighbor_edge_types, axis=-1)
    neighbor_edge_idxs = np.concatenate(neighbor_edge_idxs, axis=-1)
    neighbor_edge_times = np.concatenate(neighbor_edge_times, axis=-1)
    neighbor_edge_directions = np.concatenate(neighbor_edge_directions, axis=-1)

    if len(neighbor_sources) == 0:
        neighbor_sources = event_center_nodes#np.array([event_center_nodes]).astype('str')

    neighbor_tree = (hop_center_nodes, neighbor_sources, neighbor_destinations, neighbor_edge_types, neighbor_edge_idxs, neighbor_edge_times, neighbor_edge_directions)

    return neighbor_tree

def remap_node_id(meta_data, event_timestamp, edges, nodes, without_types = -1):
    node_remap_dict = defaultdict(dict)  # store remap result
    node_dict = defaultdict(list)  # store nodes
    node_mask_dict = defaultdict(list)  # store nodes
    node_negative_dict = defaultdict(list)  # store nodes
    edge_index_dict = defaultdict(list)
    edge_mask_dict = defaultdict(list)
    edge_rel_timestamp_dict = defaultdict(list)
    edge_idxs_dict = defaultdict(list)
    edge_type_map_dict = defaultdict(list)
    # for node_type in meta_data['node_types'].values():
    #     if node_type == meta_data['edge_types'][without_types][1]:
    #         continue
    #
    #     node_remap_dict[node_type] = {}
    #     node_dict[node_type] = []
    #     node_mask_dict[node_type] = []
    #
    for type_id, edge_types in meta_data['edge_types'].items():

        if edge_types == meta_data['edge_types'][without_types]:
            continue

        if edge_types[0] is None:
            break

        edge_type0 = edge_types[0] + '_0_' + edge_types[1] # direction 0 hyperedge - > node
        edge_type1 = edge_types[1] + '_1_' + edge_types[0] # direction 1 node - > hyperedge

        edge_type_map_dict[2*type_id] = [edge_type0, edge_types]
        edge_type_map_dict[2*type_id+1] = [edge_type1, [edge_types[1], edge_types[0]]]

        # edge_index_dict[edge_type0] = []
        # edge_index_dict[edge_type1] = []
        # edge_rel_timestamp_dict[edge_type0] = []
        # edge_rel_timestamp_dict[edge_type1] = []
        # edge_idxs_dict[edge_type0] = []
        # edge_idxs_dict[edge_type1] = []
    t1 = time.time()

    observed_edges = []

    for edge in edges:
        t4 = time.time()
        if edge in observed_edges:
            continue
        else:
            observed_edges.append(edge)

        if len(edge) == 3: # meta graph edges

            ori_source, node_type, negative = edge[0], edge[1], edge[2]

            if negative == 1:
                source = 'neg_' + ori_source
            else:
                source = ori_source

            if node_type == meta_data['edge_types'][without_types][1]:
                continue

            if source not in node_remap_dict[node_type].keys():
                node_remap_dict[node_type][source] = len(node_dict[node_type])
                node_dict[node_type].append(source.split('_')[-1])
                # if source != nodes[-1]:
                if ori_source in nodes:
                    node_mask_dict[node_type].append(1)
                else:
                    node_mask_dict[node_type].append(2)
                node_negative_dict[node_type].append(negative)
            # elif source in node_remap_dict[node_type].keys() and negative == 1:# and source == nodes[-1] and node_negative_dict[node_type][node_remap_dict[node_type][source]] == 0:
            #     # node_mask_dict[node_type][node_remap_dict[node_type][source]] = 1
            #     node_negative_dict[node_type][node_remap_dict[node_type][source]] = 2

        else:
            if len(edge) == 7:# neighbor graph edges
                ori_source, ori_destination, timestamp, edge_idx, edge_type, edge_direction, negative = edge[0], edge[1], edge[2], edge[3], edge[4], edge[5], edge[6]
                # print(event_timestamp, timestamp)
                rel_timestamp = event_timestamp - timestamp #(event_timestamp - timestamp)/np.timedelta64(60 * 24, 'm')
                # rel_timestamp = timestamp #(event_timestamp - timestamp)/np.timedelta64(60 * 24, 'm')

                if negative == 1:
                    source ='neg_'+ori_source
                    destination ='neg_'+ori_destination
                else:
                    source = ori_source
                    destination = ori_destination

            else: # meta graph edges
                ori_source, ori_destination, timestamp, edge_idx, edge_type = edge[0], edge[1], edge[2], edge[3], edge[
                    4]  # event/hyperedge nodes
                edge_direction = 0
                rel_timestamp = 1#timestamp
                negative = 0
                source = ori_source
                destination = ori_destination
            # print(event_timestamp, rel_timestamp)
            if edge_type == without_types:
                continue

            src_node_type, dst_node_type = meta_data['edge_types'][edge_type][edge_direction], meta_data['edge_types'][edge_type][1-edge_direction]
            if source not in node_remap_dict[src_node_type].keys():
                node_remap_dict[src_node_type][source] = len(node_dict[src_node_type])
                node_dict[src_node_type].append(source.split('_')[-1])
                node_negative_dict[src_node_type].append(negative)
                if ori_source in nodes:
                    if ori_source == nodes[0] and len(edge) == 5:
                        node_mask_dict[src_node_type].append(0)
                    else:
                        node_mask_dict[src_node_type].append(1)
                else:
                    node_mask_dict[src_node_type].append(2)

            # elif source in node_remap_dict[src_node_type].keys() and negative == 1 and source == nodes[-1] and node_negative_dict[src_node_type][node_remap_dict[src_node_type][source]] == 0:
            #     node_mask_dict[src_node_type][node_remap_dict[src_node_type][source]] = 1
            #     node_negative_dict[src_node_type][node_remap_dict[src_node_type][source]] = 2

            if destination not in node_remap_dict[dst_node_type].keys():
                node_remap_dict[dst_node_type][destination] = len(node_dict[dst_node_type])
                node_dict[dst_node_type].append(destination.split('_')[-1])
                node_negative_dict[dst_node_type].append(negative)
                if ori_destination in nodes and len(edge) == 5:
                    node_mask_dict[dst_node_type].append(1)
                else:
                    node_mask_dict[dst_node_type].append(2)

            # elif destination in node_remap_dict[dst_node_type].keys() and negative == 1:# and destination == nodes[-1] and node_negative_dict[dst_node_type][node_remap_dict[dst_node_type][destination]]==0:
            #     # node_mask_dict[dst_node_type][node_remap_dict[dst_node_type][destination]] = 1
            #     node_negative_dict[dst_node_type][node_remap_dict[dst_node_type][destination]] = 1
            #     node_dict[dst_node_type].append(destination.split('_')[-1])
            #     node_remap_dict[dst_node_type][destination] = len(node_dict[dst_node_type])

            # add both directions

            edge_type0 = edge_type_map_dict[2 * edge_type+edge_direction][0]
            edge_index_dict[edge_type0].append(
                [node_remap_dict[src_node_type][source], node_remap_dict[dst_node_type][destination]])
            edge_rel_timestamp_dict[edge_type0].append(rel_timestamp)
            edge_idxs_dict[edge_type0].append(edge_idx)

            edge_type1 = edge_type_map_dict[2 * edge_type+(1-edge_direction)][0]
            edge_index_dict[edge_type1].append(
                [node_remap_dict[dst_node_type][destination], node_remap_dict[src_node_type][source]])
            edge_rel_timestamp_dict[edge_type1].append(rel_timestamp)
            edge_idxs_dict[edge_type1].append(edge_idx)

            if len(edge) == 5:
                edge_mask_dict[edge_type0].append(0)
                edge_mask_dict[edge_type1].append(0)
            elif negative!=1:
                edge_mask_dict[edge_type0].append(1)
                edge_mask_dict[edge_type1].append(1)
            else:
                edge_mask_dict[edge_type0].append(2)
                edge_mask_dict[edge_type1].append(2)
        t5 = time.time()
    # for edge_types in edge_type_map_dict.values():
    #     src_node_type, dst_node_type = edge_types[1][0], edge_types[1][1]
    #     if len(node_dict[dst_node_type]) == 0:
    #         node_dict[dst_node_type].append(0)
    #         node_mask_dict[dst_node_type].append(0)
    #
    #         edge_index_dict[edge_types[0]].append(
    #             [0,0])
    #         edge_rel_timestamp_dict[edge_types[0]].append(0)
    #         edge_idxs_dict[edge_types[0]].append(0)
    #
    #     if len(node_dict[src_node_type]) == 0:
    #         node_dict[src_node_type].append(0)
    #         node_mask_dict[src_node_type].append(0)

            # edge_index_dict[edge_types[0]].append(
            #     [0,0])
            # edge_rel_timestamp_dict[edge_types[0]].append(0)
            # edge_idxs_dict[edge_types[0]].append(0)
    t2 = time.time()

    Heterograph = HeteroData()
    for edge_types in edge_type_map_dict.values():
        src_node_type, dst_node_type = edge_types[1][0], edge_types[1][1]  # [edge_type, [src_type, dst_type]]

        if src_node_type not in Heterograph.keys:
            Heterograph[src_node_type].x = torch.LongTensor(np.array(node_dict[src_node_type]).astype('int64'))
            # Heterograph[src_node_type].num_nodes = len(node_dict[src_node_type])
            Heterograph[src_node_type].node_mask = torch.LongTensor(np.array(node_mask_dict[src_node_type]).astype('int64'))
            Heterograph[src_node_type].negative = torch.LongTensor(np.array(node_negative_dict[src_node_type]).astype('int64'))
            Heterograph[src_node_type].timestamp = (torch.zeros_like(Heterograph[src_node_type].node_mask, dtype = torch.float) + event_timestamp).view(-1,1)#(event_timestamp-np.datetime64('2003-01-01T00:00'))/np.timedelta64(60 * 24, 'm'))
        if dst_node_type not in Heterograph.keys:
            Heterograph[dst_node_type].x = torch.LongTensor(np.array(node_dict[dst_node_type]).astype('int64'))
            # Heterograph[dst_node_type].num_nodes = len(node_dict[dst_node_type])
            Heterograph[dst_node_type].node_mask = torch.LongTensor(np.array(node_mask_dict[dst_node_type]).astype('int64'))
            Heterograph[dst_node_type].negative = torch.LongTensor(np.array(node_negative_dict[dst_node_type]).astype('int64'))
            Heterograph[dst_node_type].timestamp = (torch.zeros_like(Heterograph[dst_node_type].node_mask, dtype = torch.float) + event_timestamp).view(-1,1)#(event_timestamp-np.datetime64('2003-01-01T00:00'))/np.timedelta64(60 * 24, 'm'))

        edge_types_contain = edge_types[0].split('_')
        edge_type = (edge_types_contain[0], edge_types_contain[1], edge_types_contain[2])

        Heterograph[edge_type].edge_index = torch.LongTensor(
            np.array(edge_index_dict[edge_types[0]]).reshape([-1, 2]).transpose(1, 0).astype('int64'))
        Heterograph[edge_type].edge_rel_times = torch.FloatTensor(
            np.array(edge_rel_timestamp_dict[edge_types[0]]).reshape(-1, 1).astype('float64'))
        Heterograph[edge_type].edge_mask = torch.FloatTensor(
            np.array(edge_mask_dict[edge_types[0]]).reshape(-1, 1).astype('int64'))
        Heterograph[edge_type].edge_idxs = torch.LongTensor(
            np.array(edge_idxs_dict[edge_types[0]]).reshape(-1, 1).astype('int64'))
    t3 = time.time()
    # if t2-t1 >0.1:
    #     print('remap error')
    #
    # if t3-t2 >0.1:
    #     print('encoding error')

    # del node_remap_dict
    # del node_dict
    # del node_mask_dict
    # del node_negative_dict
    # del edge_index_dict
    # del edge_mask_dict
    # del edge_rel_timestamp_dict
    # del edge_idxs_dict
    # del edge_type_map_dict

    return Heterograph


def construct_event_Data(neighbor_finder, rand_sampler, event_edges, event_nodes, event_timestamp, meta_data, num_hops=1):
    # construct event_graph
    # contains : event_HeteroData, event_nodes, event_label (classification)

    # event_edges, event_nodes, event_timestamp = event_data
    if len(event_nodes) > 2:
        event_edges = event_edges[:1]
        event_nodes = event_nodes[:2]

    destination, _ = event_nodes[-1]
    _, neg = rand_sampler.sample(1)#'item_' + str(random.randint(1, 1000))
    while neg == destination:
        _, neg = rand_sampler.sample(1)#'item_'+str(random.randint(1, 1000))
    # construct event_neighbor_graph
    # contains : neighbor_HeteroData, neighbor_tree (for mail delivering)
    # event_nodes = [event_edges[0][0]]
    t1 = time.time()
    event_neighbor_edges = event_edges
    event_neighbor_edges.extend([[neg[0], 'node', 1]]) # node_id, node_type, node_negative
    event_nodes.append((neg[0], 'node'))
    negative = 0

    for node_idx, (node, node_type) in enumerate(event_nodes):
        if num_hops == 0:
            break
        # destination, timestamp, edge_type = edge[1], edge[2], edge[4]  # event/hyperedge nodes
        # event_nodes.append(destination)
        t1 = time.time()
        # source, destination, timestamp, edge_idx, edge_type = edge[0], edge[1], edge[2], edge[3], edge[4]  # event/hyperedge nodes
        hop_center_nodes, neighbor_sources, neighbor_destinations, neighbor_edge_types, neighbor_edge_idxs, neighbor_edge_times, neighbor_edge_directions\
            = get_neighbor_tree(neighbor_finder, node, event_timestamp, num_hops=num_hops, without_types = meta_data['label'])
        t2 = time.time()

        if node_idx == len(event_nodes)-1:
            negative = 1

        if len(neighbor_edge_types) == 0:
            event_neighbor_edges.extend([[neighbor_sources, node_type, negative]])
        else:
            event_neighbor_edges.extend([[neighbor_sources[i], neighbor_destinations[i], neighbor_edge_times[i], neighbor_edge_idxs[i], neighbor_edge_types[i], neighbor_edge_directions[i], negative] for i in range(len(neighbor_sources))])
    t2 = time.time()

    event_graph = remap_node_id(meta_data, event_timestamp, event_neighbor_edges, [node for (node, node_type) in event_nodes], without_types=meta_data['label'])

    del event_neighbor_edges
    t3 = time.time()
    # print(t2 - t1, t3-t2)
    # hop_center_nodes, neighbor_sources, neighbor_destinations, neighbor_edge_types, neighbor_edge_idxs, neighbor_edge_times, neighbor_edge_directions \
    #     = get_neighbor_tree(neighbor_finder, neg, event_timestamp, num_hops=num_hops, without_types=meta_data['label'])
    # if len(neighbor_edge_types) == 0:
    #     negative_neighbor_edges = [[neighbor_sources, node_type]]
    # else:
    #     negative_neighbor_edges = [[neighbor_sources[i], neighbor_destinations[i], neighbor_edge_times[i], neighbor_edge_idxs[i], neighbor_edge_types[i], neighbor_edge_directions[i]] for i in range(len(neighbor_sources))]
    # negative_graph = remap_node_id(meta_data, event_timestamp, negative_neighbor_edges, [neg], without_types=meta_data['label'])
    # neighbor_graph = HeteroData()
    # neighbor_graph = remap_node_id(meta_data, neighbor_graph, event_timestamp, event_neighbor_edges, [node for (node, node_type) in event_nodes], without_types = meta_data['label'])
    # event_label_padding = np.zeros((1, 10))-1
    # event_label_padding[:len(event_labels)] = np.array([label.split('_')[-1] for label in event_labels]).astype('int64')
    # event_label = torch.LongTensor(np.array(event_label_padding))
    t3 = time.time()
    return event_graph#, negative_graph#, event_label

def construct_graph(data, meta_data, neighbor_finder, rand_sampler, event_ids):
    event_datalist = []
    neighbor_datalist = []
    nega_datalist = []
    batch_labels = []
    t1 = time.time()

    for idx, event in enumerate(event_ids):

        # if event in data.event_labels_dict.keys():
        event_label = data.edge_labels[event]
        # else:
        #     event_label = []
        # event_data = (data.event_edges_dict[event], data.event_nodes_dict[event], data.event_timestamp_dict[event])
        t1 = time.time()
        event_graph = construct_event_Data(neighbor_finder, rand_sampler, data.event_edges_dict[event], data.event_nodes_dict[event], data.event_timestamp_dict[event], meta_data, num_hops=1)
        t2 = time.time()
        #
        batch_labels.append(event_label)
        event_datalist.append(event_graph)
        # nega_datalist.append(nega_graph)
        # neighbor_datalist.append(neighbor_graph)

    # event_dataq.put(event_datalist)
    # # neighbor_dataq.put(neighbor_datalist)
    # batch_labelq.put(batch_labels)

    return event_datalist, batch_labels

def multithreading(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids):
    # event_dataq, batch_labelq = Queue(),Queue()
    # threads = []
    # num_thread = 1
    # data_num_per_thread = len(batch_event_ids)//num_thread
    #
    # for i in range(num_thread):
    #     event_ids = batch_event_ids[i*data_num_per_thread: min((i+1)*data_num_per_thread, len(batch_event_ids))]
    #     t = threading.Thread(target=construct_graph, args = (data, meta_data, neighbor_finder, event_ids, event_dataq, batch_labelq))
    #     threads.append(t)
    #
    # for t in threads:
    #     t.start()
    #
    # for t in threads:
    #     t.join()
    #
    # event_datalist = []
    # neighbor_datalist = []
    # label_data_list = []
    #
    # for i in range(num_thread):
    #     event_datalist.extend(event_dataq.get())
    #     # neighbor_datalist.extend(neighbor_dataq.get())
    #     label_data_list.extend(batch_labelq.get())

    event_datalist, label_data_list = construct_graph(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids)

    batch_event_data = Batch.from_data_list(event_datalist)
    del event_datalist
    # batch_nega_data = Batch.from_data_list(nega_datalist)
    # batch_neighbor_data = Batch.from_data_list(neighbor_datalist)

    # labels = np.concatenate(label_data_list, axis=0).flatten()
    # paper-keyword
    # ylabel = np.zeros([len(label_data_list), 5])
    # for idx, labels in enumerate(label_data_list):
    #     for label in labels:
    #         ylabel[idx, int(label)] = 1
    # ylabel /= (ylabel.sum(axis=1)+1e-4).reshape(-1, 1)

    # paper-citation
    ylabel = np.zeros([len(label_data_list), 1])
    for idx, labels in enumerate(label_data_list):
        ylabel[idx] = labels
    # ylabel /= (ylabel.sum(axis=1) + 1e-4).reshape(-1, 1)

    batch_labels = torch.FloatTensor(ylabel)
    # print(batch_labels)
    return batch_event_data, batch_labels

def get_example_data(data, meta_data, neighbor_finder, rand_sampler, event_per_batch=200):
    event_list = []

    for idx, event in enumerate(data.ranked_event):
        if len(data.event_edges_dict[event])>0:
            event_list.append(event)

    batch_event_ids = event_list[ : event_per_batch]
    batch_event_data, batch_labels = multithreading(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids)
    return batch_event_data, batch_labels

def get_batch_of_data(data, meta_data, neighbor_finder, rand_sampler, event_per_batch=200):
    event_list = []

    for idx, event in enumerate(data.ranked_event):
        if len(data.event_edges_dict[event])>0:
            event_list.append(event)

    num_event = len(event_list)
    batch_num = round(num_event / event_per_batch) + 1

    for batch_idx in range(batch_num):
        gc.disable()
        batch_event_ids = event_list[batch_idx*event_per_batch: min((batch_idx+1)*event_per_batch, num_event-1)]

        if len(batch_event_ids) == 0:
            # print('out of data')
            return
        t1= time.time()
        batch_event_data, batch_labels = multithreading(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids)
        t2= time.time()

        # print(t2-t1)
        gc.enable()

        yield batch_event_data, batch_labels