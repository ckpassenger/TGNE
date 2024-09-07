import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

paper_hypergraph_metadata2 = {
    'node_types' : {0:'paper', 1:'author', 2:'venue', 3:'keyword'},
    'edge_types' : {
    0 : ['paper', 'author'],
    1 : ['paper', 'venue'],
    2 : ['paper', 'keyword'],
    3 : ['paper', 'paper'],
    -1 : [None, None]},
    'label' : 2
}

def get_label(dataset_path, metadata):
    graph_df = pd.read_csv(dataset_path, parse_dates=['ts'])

    ori_sources = graph_df.u.values
    ori_destinations = graph_df.i.values
    ori_edge_idxs = graph_df.idx.values
    ori_timestamps = graph_df.ts.values
    ori_edge_types = graph_df.edge_type.values

    ori_timestamps = (ori_timestamps - np.min(ori_timestamps)) / np.timedelta64(60 * 24, 'm')

    event_labels_dict = defaultdict(list)

    def zero():
        return 0
    label_num_dict = defaultdict(zero)
    for idx in tqdm(range(len(ori_sources)), total = len(ori_sources)):
        if ori_edge_types[idx] == metadata['label']:
            event_labels_dict[ori_sources[idx]].append(ori_destinations[idx])
            label_num_dict[ori_destinations[idx]] += 1

    labels = np.array(list(label_num_dict.keys()))
    label_nums = np.array(list(label_num_dict.values()))
    total_label_num = 34833
    label_remap_dict = defaultdict(zero)
    for label_idx, label_rank in enumerate(np.argsort(label_nums)[-total_label_num:]):
        label_remap_dict[labels[label_rank]] = label_idx+1
        # print(label_idx)

    label_longtail = np.argsort(label_nums)[-total_label_num]
    print(label_nums[np.argsort(label_nums)[-1]])
    print(label_nums[label_longtail])
    remap_event_label_dict = {}
    count = 0
    for paper, labels in event_labels_dict.items():
        if len(labels) == 0:
            continue
        filter_label = 0
        max_label_num = 0
        for label in labels:
            label_num = label_num_dict[label]
            if label_num >= max_label_num:
                filter_label = label
                max_label_num = label_num
        if max_label_num >= label_nums[label_longtail]:
            # if filter_label not in label_remap_dict.keys():
            # label_remap_dict[filter_label] = len(label_remap_dict.keys())+1
            # print(max_label_num, label_remap_dict[filter_label])
            remap_event_label_dict[paper] = label_remap_dict[filter_label]
        else:
            remap_event_label_dict[paper] = 0
            count+=1
    print(count, len(event_labels_dict.keys()))
    with open('label_dicts.pkl', 'wb') as file:
        pickle.dump(remap_event_label_dict, file)


get_label('./data/aminer/ml_aminer.csv', paper_hypergraph_metadata2)