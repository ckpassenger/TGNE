import random

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
import gc

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
        node_neighbor_edge_times, node_neighbor_edge_directions = neighbor_finder.get_temporal_neighbor(center_nodes, timestamp, 10, without_types=without_types)
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

    for type_id, edge_types in meta_data['edge_types'].items():

        if edge_types == meta_data['edge_types'][without_types]:
            continue

        if edge_types[0] is None:
            break

        edge_type0 = edge_types[0] + '_0_' + edge_types[1] # direction 0 hyperedge - > node
        edge_type1 = edge_types[1] + '_1_' + edge_types[0] # direction 1 node - > hyperedge

        edge_type_map_dict[2*type_id] = [edge_type0, edge_types]
        edge_type_map_dict[2*type_id+1] = [edge_type1, [edge_types[1], edge_types[0]]]

    t1 = time.time()

    observed_edges = []
    citation_count = 0
    for edge in edges:
        t4 = time.time()
        if edge in observed_edges:
            continue
        else:
            observed_edges.append(edge)

        if len(edge) == 3: # meta graph edges
            ori_source, node_type, negative = edge[0], edge[1], edge[2]

            if ori_source == nodes[0]:
                continue

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

        else:
            if len(edge) == 7:# neighbor graph edges
                ori_source, ori_destination, timestamp, edge_idx, edge_type, edge_direction, negative = edge[0], edge[1], edge[2], edge[3], edge[4], edge[5], edge[6]
                rel_timestamp = event_timestamp - timestamp

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
                if ori_source in nodes[:4] and len(edge) == 5:  # only include event corresponding nodes
                    node_mask_dict[src_node_type].append(0)
                elif ori_source in nodes[4:]:
                    node_mask_dict[src_node_type].append(1)
                else:
                    node_mask_dict[src_node_type].append(2)

            elif len(edge) == 5 and ori_source == nodes[0] and edge_type == 3:
                node_remap_dict[src_node_type][source + '_' + str(citation_count)] = len(node_dict[src_node_type])
                node_dict[src_node_type].append(source.split('_')[-1])
                node_negative_dict[src_node_type].append(negative)
                node_mask_dict[src_node_type].append(0)
                source = source + '_' + str(citation_count)
                citation_count += 1

            if destination not in node_remap_dict[dst_node_type].keys():
                node_remap_dict[dst_node_type][destination] = len(node_dict[dst_node_type])
                node_dict[dst_node_type].append(destination.split('_')[-1])
                node_negative_dict[dst_node_type].append(negative)
                if ori_destination in nodes[:4]:
                    node_mask_dict[dst_node_type].append(0)
                elif ori_destination in nodes[4:] and len(edge) == 5:  # only include event corresponding nodes
                    node_mask_dict[dst_node_type].append(1)
                else:
                    node_mask_dict[dst_node_type].append(2)

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

            if len(edge) == 5 and edge_type==3:
                edge_mask_dict[edge_type0].append(0)
                edge_mask_dict[edge_type1].append(0)
            elif negative!=1:
                edge_mask_dict[edge_type0].append(1)
                edge_mask_dict[edge_type1].append(1)
            else:
                edge_mask_dict[edge_type0].append(2)
                edge_mask_dict[edge_type1].append(2)

    Heterograph = HeteroData()
    for edge_types in edge_type_map_dict.values():
        src_node_type, dst_node_type = edge_types[1][0], edge_types[1][1]  # [edge_type, [src_type, dst_type]]

        if src_node_type not in Heterograph.keys:
            if np.array(node_dict[src_node_type]).astype('int64').shape[0] != 0:
                Heterograph[src_node_type].x = torch.LongTensor(np.array(node_dict[src_node_type]).astype('int64'))
                # Heterograph[src_node_type].num_nodes = len(node_dict[src_node_type])
                Heterograph[src_node_type].node_mask = torch.LongTensor(np.array(node_mask_dict[src_node_type]).astype('int64'))
                Heterograph[src_node_type].negative = torch.LongTensor(np.array(node_negative_dict[src_node_type]).astype('int64'))
                Heterograph[src_node_type].timestamp = (torch.zeros_like(Heterograph[src_node_type].node_mask, dtype = torch.float) + event_timestamp).view(-1,1)#(event_timestamp-np.datetime64('2003-01-01T00:00'))/np.timedelta64(60 * 24, 'm'))
            else:
                Heterograph[src_node_type].x = torch.zeros(1).astype('int64').view(1,-1)
                Heterograph[src_node_type].node_mask = torch.zeros(1).astype('int64').view(1,-1)
                Heterograph[src_node_type].negative = torch.zeros(1).astype('int64').view(1,-1)
                Heterograph[src_node_type].timestamp = torch.zeros(1, dtype = torch.float).view(1,-1)

        if dst_node_type not in Heterograph.keys:
            if np.array(node_dict[dst_node_type]).astype('int64').shape[0] != 0:
                Heterograph[dst_node_type].x = torch.LongTensor(np.array(node_dict[dst_node_type]).astype('int64'))
                # Heterograph[dst_node_type].num_nodes = len(node_dict[dst_node_type])
                Heterograph[dst_node_type].node_mask = torch.LongTensor(np.array(node_mask_dict[dst_node_type]).astype('int64'))
                Heterograph[dst_node_type].negative = torch.LongTensor(np.array(node_negative_dict[dst_node_type]).astype('int64'))
                Heterograph[dst_node_type].timestamp = (torch.zeros_like(Heterograph[dst_node_type].node_mask, dtype = torch.float) + event_timestamp).view(-1,1)#(event_timestamp-np.datetime64('2003-01-01T00:00'))/np.timedelta64(60 * 24, 'm'))
            else:
                Heterograph[dst_node_type].x = torch.zeros(1).astype('int64').view(1, -1)
                Heterograph[dst_node_type].node_mask = torch.zeros(1).astype('int64').view(1, -1)
                Heterograph[dst_node_type].negative = torch.zeros(1).astype('int64').view(1, -1)
                Heterograph[dst_node_type].timestamp = torch.zeros(1, dtype = torch.float).view(1, -1)

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

    return Heterograph


def construct_event_Data(neighbor_finder, rand_sampler, event_edges, event_nodes, event_timestamp, meta_data, num_hops=2):
    # construct event_graph
    # contains : event_HeteroData, event_nodes, event_label (classification)

    event_nodes = event_nodes.copy()
    event_corresponding_nodes = [node for (node, node_type) in event_nodes[4:] if node_type=='paper']

    for node in event_corresponding_nodes:
        neg_event_corresponding_nodes, _ = rand_sampler.sample(1,
                                                               event_timestamp)

        while neg_event_corresponding_nodes[0] in event_corresponding_nodes:
            neg_event_corresponding_nodes, _ = rand_sampler.sample(1, event_timestamp)

        event_nodes.append((neg_event_corresponding_nodes[0], 'paper'))

    # construct event_neighbor_graph
    # contains : neighbor_HeteroData, neighbor_tree (for mail delivering)

    event_neighbor_edges = event_edges

    negative = 0

    for node_idx, (node, node_type) in enumerate(event_nodes):
        if num_hops == 0:
            break

        # source, destination, timestamp, edge_idx, edge_type = edge[0], edge[1], edge[2], edge[3], edge[4]  # event/hyperedge nodes
        hop_center_nodes, neighbor_sources, neighbor_destinations, neighbor_edge_types, neighbor_edge_idxs, neighbor_edge_times, neighbor_edge_directions\
            = get_neighbor_tree(neighbor_finder, node, event_timestamp, num_hops=num_hops, without_types = meta_data['label'])

        if node_idx >= len(event_nodes)-len(event_corresponding_nodes):
            negative = 1

        if len(neighbor_edge_types) == 0:
            event_neighbor_edges.extend([[neighbor_sources, node_type, negative]])
        else:
            event_neighbor_edges.extend([[neighbor_sources[i], neighbor_destinations[i], neighbor_edge_times[i], neighbor_edge_idxs[i], neighbor_edge_types[i], neighbor_edge_directions[i], negative] for i in range(len(neighbor_sources))])

    event_graph = remap_node_id(meta_data, event_timestamp, event_neighbor_edges, [node for (node, node_type) in event_nodes], without_types=meta_data['label'])

    del event_neighbor_edges

    return event_graph#, negative_graph#, event_label

def construct_graph(data, meta_data, neighbor_finder, rand_sampler, event_ids):
    event_datalist = []
    batch_labels = []

    for idx, event in enumerate(event_ids):
        event_label = np.zeros(6)#data.event_label[event]

        t1 = time.time()

        event_graph = construct_event_Data(neighbor_finder, rand_sampler, data.event_edges_dict[event], data.event_nodes_dict[event], data.event_timestamp_dict[event], meta_data, num_hops=2)

        #
        batch_labels.append(event_label)
        event_datalist.append(event_graph)

    return event_datalist, batch_labels

def multithreading(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids):

    event_datalist, label_data_list = construct_graph(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids)

    batch_event_data = Batch.from_data_list(event_datalist)
    del event_datalist
    # paper-citation
    ylabel = np.zeros([len(label_data_list), 5])
    for idx, labels in enumerate(label_data_list):
        ylabel[idx] = labels[1:6]

    batch_labels = torch.FloatTensor(ylabel)

    return batch_event_data, batch_labels

def get_example_data(data, meta_data, neighbor_finder, rand_sampler, event_per_batch=200):
    event_list = []

    for idx, event in enumerate(data.ranked_event):
        if len(data.event_edges_dict[event])>0:
            event_list.append(event)

    batch_event_ids = event_list[ : event_per_batch]
    batch_event_data, batch_labels = multithreading(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids)
    return batch_event_data, batch_labels

def split_batch(data, event_list):
    batch = []

    last_time = data.timestamps[0]
    for idx, source in enumerate(event_list):
        time = data.timestamps[idx]

        if time != last_time:
            yield batch
            batch = []

        last_time = time
        batch.append(source)

    return batch

def get_batch_of_data(data, meta_data, neighbor_finder, rand_sampler, event_per_batch=200):
    event_list = []

    for idx, event in enumerate(data.ranked_event):
        if len(data.event_edges_dict[event])>0:
            event_list.append(event)
    num_event = len(event_list)
    batch_num = round(num_event / event_per_batch) + 1

    for batch in split_batch(data, event_list):
        gc.disable()
        # batch_event_ids = event_list[batch_idx*event_per_batch: min((batch_idx+1)*event_per_batch, num_event-1)]
        if len(batch) == 0:
            continue
        batch_event_data, batch_labels = multithreading(data, meta_data, neighbor_finder, rand_sampler, batch)

        gc.enable()

        yield batch_event_data, batch_labels
    return

# def get_batch_of_data(data, meta_data, neighbor_finder, rand_sampler, event_per_batch=200):
#     event_list = []
#
#     for idx, event in enumerate(data.ranked_event):
#         if len(data.event_edges_dict[event])>0:
#             event_list.append(event)
#     num_event = len(event_list)
#     batch_num = round(num_event / event_per_batch) + 1
#
#     for batch_idx in range(batch_num):
#         gc.disable()
#         batch_event_ids = event_list[batch_idx*event_per_batch: min((batch_idx+1)*event_per_batch, num_event-1)]
#         if len(batch_event_ids) == 0:
#             return
#         batch_event_data, batch_labels = multithreading(data, meta_data, neighbor_finder, rand_sampler, batch_event_ids)
#
#         gc.enable()
#
#         yield batch_event_data, batch_labels
#     return