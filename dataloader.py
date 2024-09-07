from collections import defaultdict
import numpy as np
import random
import pandas as pd
import pickle
from tqdm import tqdm
import torch

paper_hypergraph_metadata = {
    'node_types' : {0:'paper_', 1:'author_', 2:'venue_', 3:'keyword_'},
    'edge_types' : {
    0 : ['paper_', 'author_'],
    1 : ['paper_', 'venue_'],
    2 : ['paper_', 'keyword_'],
    3 : ['paper_', 'paper_']},
    'label' : 2
}

class HeteroData:
    def __init__(self, sources, destinations, timestamps, edge_idxs, edge_types, meta_data, event_label_by_source = None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.edge_types = edge_types
        self.event_label = event_label_by_source
        self.meta_data = meta_data
        self.unique_events = np.unique(sources)
        self.event_edges_dict = len(self.unique_events)
        self.event_edges_dict, self.event_nodes_dict, self.event_type_node_dict, self.event_timestamp_dict, self.event_labels_dict, self.event_number_dict, self.ranked_event\
            = self.aggregate_edges(sources, destinations, timestamps, edge_idxs, edge_types, meta_data)
        # self.event_meta_edges_dict, self.num_meta_edges = self.construct_meta_edge()

        with open('label_dicts_citation_aps.pkl', 'rb') as file:
            self.event_labels_dict = pickle.load(file)
        # all_labels = []
        # for paper, labels in self.event_labels_dict.items():
        #     all_labels.append(labels.reshape(1,-1))
        #
        # all_label = np.concatenate(all_labels, axis = 0)
        #
        # print(np.sort(all_label[:,4], axis=-1)[-20:])
        if event_label_by_source is None:
            self.event_label = self.event_labels_dict
        # self.select_labels()
    def construct_meta_edge(self):
        # construct 2 hop meta edge
        event_meta_edge_dict = defaultdict(list)
        num_meta_edge = 0

        for i, source in enumerate(self.ranked_event):
            for j, src in enumerate(self.event_nodes_dict[source][1:-1]):
                for k, dst in enumerate(self.event_nodes_dict[source][j+2:]):
                    src_id, src_type = src
                    dst_id, dst_type = dst

                    meta_edge_type = [src_type, dst_type]
                    meta_edge = [src_id, dst_id, self.event_timestamp_dict[source], len(self.sources)+num_meta_edge, meta_edge_type]
                    event_meta_edge_dict[source].append(meta_edge)
                    num_meta_edge += 1
        return event_meta_edge_dict, num_meta_edge

    def aggregate_edges(self, sources, destinations, timestamps, edge_idxs, edge_types, meta_data):
        event_edges_dict = {}#defaultdict(list)
        event_nodes_dict = {}#defaultdict(list)
        event_number_dict = {}

        def create_meta_dict():
            meta_dict = {}
            for node_type in self.meta_data['node_types'].values():
                meta_dict[node_type] = []
            return meta_dict

        event_type_node_dict = defaultdict(create_meta_dict)
        event_labels_dict = defaultdict(list)
        event_timestamp_dict = {}
        ranked_event = []

        paper_list = []
        for i, source in enumerate(sources):
            src_type = meta_data['edge_types'][edge_types[i]][0]
            dst_type = meta_data['edge_types'][edge_types[i]][1]
            # if edge_types[i] != 3:
            #     continue
            ranked_event.append(i)
            # if source not in event_timestamp_dict.keys():
            event_timestamp_dict[i] = timestamps[i]
            event_edges_dict[i] = []
            event_number_dict[i] = 0
            event_nodes_dict[i] = [(source,src_type)]

            for edge_type in self.meta_data['edge_types']:
                if edge_type != meta_data['label']:
                    dst_type_ = meta_data['edge_types'][edge_type][1]
                    # event_edges_dict[i].append([source, dst_type_+'_0', timestamps[i], 0, edge_type])
                    if dst_type_ != 'paper':
                        event_nodes_dict[i].append([dst_type_+'_0', dst_type_])

            if edge_types[i] != meta_data['label']:
                event_edges_dict[i].append(
                    [source, destinations[i], timestamps[i], edge_idxs[i], edge_types[i]])
                event_nodes_dict[i].append((destinations[i], dst_type))
                event_type_node_dict[source][dst_type].append((destinations[i], dst_type))

                # if edge_types[i] == 3:
                #     event_number_dict[i] += 1

            elif edge_types[i] == meta_data['label']:
                event_labels_dict[source].append(destinations[i].split('_')[-1])

        self.n_interactions = len(self.sources)
        self.n_unique_nodes = len(set(sources) | set(destinations))

        return event_edges_dict, event_nodes_dict, event_type_node_dict, event_timestamp_dict, event_labels_dict, event_number_dict, np.array(ranked_event)

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, edge_types, meta_data, edge_labels, event_label_by_source = None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.edge_types = edge_types
        self.edge_labels = edge_labels
        self.event_label = edge_labels
        self.meta_data = meta_data
        self.unique_events = np.unique(sources)
        self.n_unique_events = len(self.unique_events)
        self.n_interactions = len(self.sources)
        self.n_unique_nodes = len(set(sources) | set(destinations))
        # self.ranked_event = [_ for _ in range(self.n_unique_events)]
        self.event_edges_dict, self.event_nodes_dict, self.event_timestamp_dict, self.event_labels_dict, self.ranked_event\
            = self.aggregate_edges(sources, destinations, timestamps, edge_idxs, edge_types, edge_labels, meta_data)

        # with open('label_dicts_citation_aps.pkl', 'rb') as file:
        #     self.event_labels_dict = pickle.load(file)
        # all_labels = []
        # for paper, labels in self.event_labels_dict.items():
        #     all_labels.append(labels.reshape(1,-1))
        #
        # all_label = np.concatenate(all_labels, axis = 0)
        #
        # print(np.sort(all_label[:,4], axis=-1)[-20:])
        # if event_label_by_source is None:
        #     self.event_label = self.event_labels_dict
        # self.select_labels()


    def aggregate_edges(self, sources, destinations, timestamps, edge_idxs, edge_types, edge_labels, meta_data):
        event_edges_dict = {}#defaultdict(list)
        event_nodes_dict = {}#defaultdict(list)
        event_labels_dict = {}
        event_timestamp_dict = {}
        ranked_event = []

        # new_sources = []
        # new_destinations = []
        # new_timestamps = []
        # new_edge_idxs = []
        # new_edge_types = []

        # for i, source in tqdm(enumerate(sources), total=len(self.sources)):
        for i, source in enumerate(sources):
            src_type = meta_data['edge_types'][edge_types[i]][0]
            dst_type = meta_data['edge_types'][edge_types[i]][1]

            # if source not in event_timestamp_dict.keys():
            event_timestamp_dict[i] = timestamps[i]
            event_labels_dict[i] = edge_labels[i]
            event_edges_dict[i] = []
            event_nodes_dict[i] = [(source,src_type)]
            ranked_event.append(i)
                # for edge_type in self.meta_data['edge_types']:
                #     if edge_type != meta_data['label']:
                #         dst_type_ = meta_data['edge_types'][edge_type][1]
                #         event_edges_dict[i].append([source, dst_type_+'_0', timestamps[i], 0, edge_type])
                #         event_nodes_dict[i].append([dst_type_+'_0', dst_type_])

            if edge_types[i] != meta_data['label']:
                event_edges_dict[i].append([source, destinations[i], timestamps[i], edge_idxs[i], edge_types[i]])
                event_nodes_dict[i].append((destinations[i],dst_type))
                # new_sources.append(source)
                # new_destinations.append(destinations[i])
                # new_timestamps.append(timestamps[i])
                # new_edge_idxs.append(edge_idxs[i])
                # new_edge_types.append(edge_types[i])
            # elif edge_types[i] == meta_data['label']:
            #     event_labels_dict[source].append(destinations[i].split('_')[-1])
        # self.sources = np.array(new_sources)
        # self.destinations = np.array(new_destinations)
        # self.timestamps = np.array(new_timestamps)
        # self.edge_idxs = np.array(new_edge_idxs)
        # self.edge_types = np.array(new_edge_types)
        self.n_interactions = len(self.sources)
        self.n_unique_nodes = len(set(sources) | set(destinations))

        return event_edges_dict, event_nodes_dict, event_timestamp_dict, event_labels_dict, np.array(ranked_event)

    # def select_labels(self):
    #     label_remap_dict = {}
    #     remap_event_label_dict = {}
    #
    #     for paper, labels in self.event_labels_dict.items():
    #         if len(labels) == 0 or len(self.event_edges_dict) == 0:
    #             continue
    #         filter_label = 0
    #         max_label_num = 0
    #         for label in labels:
    #             label_num = self.label_list[label]
    #             if label_num >= max_label_num:
    #                 filter_label = label
    #                 max_label_num = label_num
    #         if max_label_num
    #         if filter_label not in label_remap_dict.keys():
    #             label_remap_dict[filter_label] = len(label_remap_dict.keys())+1
    #
    #         remap_event_label_dict[paper] = label_remap_dict[filter_label]
    #         print(filter_label, max_label_num, labels)
    #     print(len(label_remap_dict.keys()))

def get_HeteroData(dataset_path, metadata, data_num = 500000):
    # initial_data = 0
    ### Load data and train val test split
    print("Reading data...")
    graph_df = pd.read_csv(dataset_path, parse_dates = ['ts'])

   # oag
    ori_sources = graph_df.u.values[:data_num]   #user_id,item_id,timestamp
    ori_destinations = graph_df.i.values[:data_num]
    ori_edge_idxs = graph_df.idx.values[:data_num]
    ori_timestamps =  graph_df.ts.values[:data_num]
    ori_edge_types = graph_df.edge_type.values[:data_num]
    # label = graph_df.state_label.values[:data_num]

    ori_timestamps = (ori_timestamps - np.min(ori_timestamps))/np.timedelta64(60 * 24, 'm')

    unique_nodes = set(ori_sources) | set(ori_destinations)

    print("Remaping...")

    node_set_by_type = defaultdict(dict)
    node_label_by_source = {}
    sources = []
    destinations = []
    edge_idxs = []
    timestamps = []
    edge_types = []
    n_nodes = {}

    with open('label_dicts.pkl', 'rb') as file:
        label_remap_dict = pickle.load(file)

    for idx in tqdm(range(len(ori_sources)), total = len(ori_sources)):
        if ori_edge_types[idx] not in metadata['edge_types'].keys():
            continue

        source_type = metadata['edge_types'][ori_edge_types[idx]][0]
        destination_type = metadata['edge_types'][ori_edge_types[idx]][1]

        edge_idxs.append(ori_edge_idxs[idx])
        timestamps.append(ori_timestamps[idx])
        # if ori_edge_types[idx] == 3:
        #     timestamps.append(ori_timestamps[idx])
        # else:
        #     timestamps.append(ori_timestamps[idx]-0.1)
        edge_types.append(ori_edge_types[idx])

        if ori_sources[idx] not in node_set_by_type[source_type].keys():
            source_remap_id = source_type +'_'+ str(len(node_set_by_type[source_type].keys())+1)
            node_set_by_type[source_type][ori_sources[idx]] = source_remap_id
            sources.append(source_remap_id)
        else:
            sources.append(node_set_by_type[source_type][ori_sources[idx]])

        # if ori_edge_types[idx] == 2:
        if ori_sources[idx] in label_remap_dict.keys():
            node_label_by_source[node_set_by_type[source_type][ori_sources[idx]]] = label_remap_dict[ori_sources[idx]]
            # print(label_remap_dict[ori_sources[idx]])\ count+=
        else:
            node_label_by_source[node_set_by_type[source_type][ori_sources[idx]]] = 0

        if ori_destinations[idx] not in node_set_by_type[destination_type].keys():
            destination_remap_id = destination_type+'_' + str(len(node_set_by_type[destination_type].keys())+1)
            node_set_by_type[destination_type][ori_destinations[idx]] = destination_remap_id
            destinations.append(destination_remap_id)
        else:
            destinations.append(node_set_by_type[destination_type][ori_destinations[idx]])
    sources = np.array(sources)#[node_remap_dict[node] for node in ori_sources])
    destinations = np.array(destinations)#[node_remap_dict[node] for node in ori_destinations])
    edge_idxs = np.array(range(len(edge_idxs)))
    timestamps = np.array(timestamps)
    edge_types = np.array(edge_types)

    for type, nodes in node_set_by_type.items():
        n_nodes[type] = len(nodes.keys())+1
        print(type, len(nodes.keys()))

    # print('user',len(node_set_by_type['user'].keys()))
    # print('item',len(node_set_by_type['item'].keys()))

    with open('map_dicts.pkl', 'wb') as file:
        pickle.dump(node_set_by_type, file)

    print("Creating full data...")
    full_data = HeteroData(sources, destinations, timestamps, edge_idxs, edge_types, metadata)#, node_label_by_source)


    return full_data, n_nodes

def get_Data(dataset_path, metadata, data_num = 500000, hetero = False):
    # initial_data = 0
    ### Load data and train val test split
    print("Reading data...")
    graph_df = pd.read_csv(dataset_path)#, parse_dates = ['ts']
   # oag
   #  ori_sources = graph_df.u.values[:data_num]   #user_id,item_id,timestamp
   #  ori_destinations = graph_df.i.values[:data_num]
   #  ori_edge_idxs = graph_df.idx.values[:data_num]
   #  ori_timestamps =  graph_df.ts.values[:data_num]
   #  ori_edge_types = graph_df.edge_type.values[:data_num]
    # label = graph_df.state_label.values[:data_num]

   # wiki
    ori_sources = graph_df.u.values[:data_num]
    ori_destinations =graph_df.i.values[:data_num]
    ori_edge_idxs = graph_df.idx.values[:data_num]
    # ori_timestamps = graph_df.ts.values[:data_num]
    ori_timestamps = graph_df.ts.values[:data_num]
    ori_edge_types = np.zeros_like(ori_sources)#graph_df.edge_type.values[:data_num]
    label = graph_df.label.values[:data_num]

    # timestamps_mean, timestamps_std = np.mean(ori_timestamps), np.std(ori_timestamps)

    # ori_timestamps = (ori_timestamps - 0) / (1000+1e-4)
    # ori_timestamps = (ori_timestamps - timestamps_mean) / (timestamps_std+1e-4)

    # ori_timestamps = (ori_timestamps - np.min(ori_timestamps))/np.timedelta64(60 * 24, 'm')

    unique_nodes = set(ori_sources) | set(ori_destinations)

    print("Remaping...")

    node_set_by_type = defaultdict(dict)
    node_label_by_source = {}
    sources = []
    destinations = []
    edge_idxs = []
    edge_labels = []
    timestamps = []
    edge_types = []

    with open('label_dicts.pkl', 'rb') as file:
        label_remap_dict = pickle.load(file)

    for idx in tqdm(range(len(ori_sources)), total = len(ori_sources)):
        if ori_edge_types[idx] not in metadata['edge_types'].keys():
            continue

        source_type = metadata['edge_types'][ori_edge_types[idx]][0]
        destination_type = metadata['edge_types'][ori_edge_types[idx]][1]

        edge_idxs.append(ori_edge_idxs[idx])
        timestamps.append(ori_timestamps[idx])
        edge_types.append(ori_edge_types[idx])
        edge_labels.append(label[idx])

        if ori_sources[idx] not in node_set_by_type[source_type].keys():
            source_remap_id = source_type +'_'+ str(len(node_set_by_type[source_type].keys()))
            node_set_by_type[source_type][ori_sources[idx]] = source_remap_id
            sources.append(source_remap_id)
        else:
            sources.append(node_set_by_type[source_type][ori_sources[idx]])

        # if ori_edge_types[idx] == 2:
        if ori_sources[idx] in label_remap_dict.keys():
            node_label_by_source[node_set_by_type[source_type][ori_sources[idx]]] = label_remap_dict[ori_sources[idx]]
            # print(label_remap_dict[ori_sources[idx]])\ count+=
        else:
            node_label_by_source[node_set_by_type[source_type][ori_sources[idx]]] = 0

        if ori_destinations[idx] not in node_set_by_type[destination_type].keys():
            destination_remap_id = destination_type+'_' + str(len(node_set_by_type[destination_type].keys()))
            node_set_by_type[destination_type][ori_destinations[idx]] = destination_remap_id
            destinations.append(destination_remap_id)
        else:
            destinations.append(node_set_by_type[destination_type][ori_destinations[idx]])
    if hetero == False:
        destinations = []
        num_source = len(node_set_by_type[source_type].keys())
        for idx in tqdm(range(len(ori_sources)), total=len(ori_sources)):
            if ori_destinations[idx]+num_source not in node_set_by_type[source_type].keys():
                destination_remap_id = source_type + '_' + str(len(node_set_by_type[source_type].keys()))
                node_set_by_type[source_type][ori_destinations[idx]+num_source] = destination_remap_id
                destinations.append(destination_remap_id)
            else:
                destinations.append(node_set_by_type[destination_type][ori_destinations[idx]+num_source])

    sources = np.array(sources)#[node_remap_dict[node] for node in ori_sources])
    destinations = np.array(destinations)#[node_remap_dict[node] for node in ori_destinations])
    edge_idxs = np.array(range(len(edge_idxs)))
    timestamps = np.array(timestamps)
    edge_types = np.array(edge_types)
    edge_labels = np.array(edge_labels)

    # print('user',len(node_set_by_type['user'].keys()))
    # print('item',len(node_set_by_type['item'].keys()))

    with open('map_dicts.pkl', 'wb') as file:
        pickle.dump(node_set_by_type, file)

    print("Creating full data...")
    full_data = Data(sources, destinations, timestamps, edge_idxs, edge_types, metadata, edge_labels)#, node_label_by_source)


    return full_data

def split_heterodata(full_data, head_rate=0.4, tail_rate=0.9, snapshot_rate = 0.1, val_rate=0.15, test_rate = 0.15):
    sources, destinations, timestamps, edge_idxs, edge_types, labels = full_data.sources, full_data.destinations, \
                                                                       full_data.timestamps, full_data.edge_idxs, full_data.edge_types, full_data.edge_types#, full_data.edge_labels

    start_time = np.min(timestamps)
    end_time = np.max(timestamps)
    # print(start_time, end_time)
    head_time = (end_time - start_time) * head_rate + start_time
    tail_time = (end_time - start_time) * tail_rate + start_time
    # print(head_time, tail_time)
    snapshot_head_rates = [head_rate + snapshot_rate * i for i in range(round((tail_rate - head_rate) / snapshot_rate))]

    # test_time = tail_time - (tail_time - head_time)*test_rate
    # val_time = test_time - (tail_time - head_time)*val_rate if use_validation else test_time

    val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
    # print(test_time, val_time)
    print("Spliting...")

    snapshot_datas = []

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training

    # Mask saying for each source and destination whether they are new test nodes

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)

    initial_mask = timestamps <= head_time
    # train_mask = timestamps<=np.datetime64('2008-12-31')
    train_mask = timestamps <= val_time #if use_validation else timestamps <= test_time
    # define the new nodes sets for testing inductiveness of the model
    train_data = HeteroData(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], edge_types[train_mask], full_data.meta_data,
                      labels)  # , full_data.node_label)

    # test_mask = np.logical_and(timestamps <= np.datetime64('2009-12-31'), timestamps > np.datetime64('2008-12-31'))
    test_mask = np.logical_and(timestamps <= tail_time, timestamps > test_time)
    # val_mask = np.logical_and(timestamps <= np.datetime64('2010-12-31'), timestamps > np.datetime64('2009-12-31')) if use_validation else test_mask
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) #if use_validation else test_mask

    initial_data = HeteroData(sources[initial_mask], destinations[initial_mask], timestamps[initial_mask],
                        edge_idxs[initial_mask], edge_types[initial_mask], full_data.meta_data,
                        labels)  # , full_data.node_label)

    val_data = HeteroData(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], edge_types[val_mask], full_data.meta_data, labels)  # , full_data.node_label)

    test_data = HeteroData(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], edge_types[test_mask], full_data.meta_data,
                     labels)  # , full_data.node_label)

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return initial_data, snapshot_datas, train_data, val_data, test_data#, new_node_val_data, new_node_test_data

def split_data_node(full_data, head_rate=0.4, tail_rate=0.9, snapshot_rate = 0.1, val_rate=0.15, test_rate = 0.15, use_validation=True, different_new_nodes_between_val_and_test = False):
    sources, destinations, timestamps, edge_idxs, edge_types, labels = full_data.sources, full_data.destinations,\
                                                               full_data.timestamps, full_data.edge_idxs, full_data.edge_types, full_data.edge_labels

    start_time = np.min(timestamps)
    end_time = np.max(timestamps)
    # print(start_time, end_time)
    head_time = (end_time - start_time)*head_rate + start_time
    tail_time = (end_time - start_time)*tail_rate + start_time
    # print(head_time, tail_time)
    snapshot_head_rates = [head_rate+snapshot_rate*i for i in range(round((tail_rate-head_rate)/snapshot_rate))]

    # test_time = tail_time - (tail_time - head_time)*test_rate
    # val_time = test_time - (tail_time - head_time)*val_rate if use_validation else test_time

    val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
    # print(test_time, val_time)
    print("Spliting...")


    snapshot_datas = []

    # for snapshot_head_rate in snapshot_head_rates:
    #     snapshot_head_time = (end_time - start_time)*snapshot_head_rate + start_time
    #     snapshot_tail_time = (end_time - start_time)*(snapshot_head_rate+snapshot_rate) + start_time
    #
    #     snapshot_mask = np.logical_and(timestamps <= snapshot_tail_time, timestamps > snapshot_head_time)
    #     snapshot_data = Data(sources[snapshot_mask], destinations[snapshot_mask], timestamps[snapshot_mask],
    #                   edge_idxs[snapshot_mask], edge_types[snapshot_mask], full_data.meta_data, labels)#, full_data.node_label)
    #
    #     snapshot_datas.append(snapshot_data)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time

    test_node_set = []
    for node in np.concatenate([sources[timestamps > val_time], destinations[timestamps > val_time]]):
        if node not in test_node_set:
            test_node_set.append(node)
    # np.unique(np.concatenate([sources[timestamps > val_time], destinations[timestamps > val_time]]))
    # test_node_set = set(sources[timestamps > val_time]).union(
    #     set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    random.seed(2022)

    initial_mask = timestamps <= head_time
    # train_mask = timestamps<=np.datetime64('2008-12-31')
    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    # define the new nodes sets for testing inductiveness of the model
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], edge_types[train_mask], full_data.meta_data,
                      labels[train_mask])  # , full_data.node_label)

    # test_mask = np.logical_and(timestamps <= np.datetime64('2009-12-31'), timestamps > np.datetime64('2008-12-31'))
    test_mask = np.logical_and(timestamps <= tail_time, timestamps > test_time)
    # val_mask = np.logical_and(timestamps <= np.datetime64('2010-12-31'), timestamps > np.datetime64('2009-12-31')) if use_validation else test_mask
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask


    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], edge_types[val_mask], full_data.meta_data, labels[val_mask])#, full_data.node_label)

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], edge_types[test_mask], full_data.meta_data, labels[test_mask])#, full_data.node_label)


    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return train_data, val_data, test_data

def split_data(full_data, head_rate=0.4, tail_rate=0.9, snapshot_rate = 0.1, val_rate=0.15, test_rate = 0.15, use_validation=True, different_new_nodes_between_val_and_test = False):
    sources, destinations, timestamps, edge_idxs, edge_types, labels = full_data.sources, full_data.destinations,\
                                                               full_data.timestamps, full_data.edge_idxs, full_data.edge_types, full_data.event_label

    start_time = np.min(timestamps)
    end_time = np.max(timestamps)
    # print(start_time, end_time)
    head_time = (end_time - start_time)*head_rate + start_time
    tail_time = (end_time - start_time)*tail_rate + start_time
    # print(head_time, tail_time)
    snapshot_head_rates = [head_rate+snapshot_rate*i for i in range(round((tail_rate-head_rate)/snapshot_rate))]

    # test_time = tail_time - (tail_time - head_time)*test_rate
    # val_time = test_time - (tail_time - head_time)*val_rate if use_validation else test_time

    val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
    # print(test_time, val_time)
    print("Spliting...")


    snapshot_datas = []

    # for snapshot_head_rate in snapshot_head_rates:
    #     snapshot_head_time = (end_time - start_time)*snapshot_head_rate + start_time
    #     snapshot_tail_time = (end_time - start_time)*(snapshot_head_rate+snapshot_rate) + start_time
    #
    #     snapshot_mask = np.logical_and(timestamps <= snapshot_tail_time, timestamps > snapshot_head_time)
    #     snapshot_data = Data(sources[snapshot_mask], destinations[snapshot_mask], timestamps[snapshot_mask],
    #                   edge_idxs[snapshot_mask], edge_types[snapshot_mask], full_data.meta_data, labels)#, full_data.node_label)
    #
    #     snapshot_datas.append(snapshot_data)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time

    test_node_set = []
    for node in np.concatenate([sources[timestamps > val_time], destinations[timestamps > val_time]]):
        if node not in test_node_set:
            test_node_set.append(node)
    # np.unique(np.concatenate([sources[timestamps > val_time], destinations[timestamps > val_time]]))
    # test_node_set = set(sources[timestamps > val_time]).union(
    #     set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    random.seed(2022)

    test_node_idx = random.sample(list(range((len(test_node_set)))), int(0.1 * n_total_unique_nodes))

    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
    # print(new_test_node_set)2

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = pd.Series(sources).map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = pd.Series(destinations).map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    initial_mask = timestamps <= head_time
    # train_mask = timestamps<=np.datetime64('2008-12-31')
    train_mask = np.logical_and(timestamps <= val_time if use_validation else timestamps <= test_time, observed_edges_mask)
    # define the new nodes sets for testing inductiveness of the model
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], edge_types[train_mask], full_data.meta_data,
                      labels)  # , full_data.node_label)

    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    # test_mask = np.logical_and(timestamps <= np.datetime64('2009-12-31'), timestamps > np.datetime64('2008-12-31'))
    test_mask = np.logical_and(timestamps <= tail_time, timestamps > test_time)
    # val_mask = np.logical_and(timestamps <= np.datetime64('2010-12-31'), timestamps > np.datetime64('2009-12-31')) if use_validation else test_mask
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    initial_data = Data(sources[initial_mask], destinations[initial_mask], timestamps[initial_mask],
                        edge_idxs[initial_mask], edge_types[initial_mask], full_data.meta_data,
                        labels)  # , full_data.node_label)

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], edge_types[val_mask], full_data.meta_data, labels)#, full_data.node_label)

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], edge_types[test_mask], full_data.meta_data, labels)#, full_data.node_label)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], edge_types[new_node_val_mask], full_data.meta_data, labels)

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask], edge_types[new_node_test_mask], full_data.meta_data, labels)

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    return initial_data, snapshot_datas, train_data, val_data, test_data, new_node_val_data, new_node_test_data
