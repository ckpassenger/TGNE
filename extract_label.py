from collections import defaultdict
from time import time
import numpy as np
import random
import pandas as pd
import pickle
import datetime
from tqdm import tqdm

from dataloader import get_Data, split_data
from neighbor_finder import get_neighbor_finder
from event_constructor import get_batch_of_data

paper_hypergraph_metadata2 = {
    'node_types' : {0:'paper', 1:'author', 2:'venue', 3:'keyword'},
    'edge_types' : {
    0 : ['paper', 'author'],
    1 : ['paper', 'venue'],
    2 : ['paper', 'keyword'],
    3 : ['paper', 'paper']},
    'label' : 2
}

dataset = 'aps'

full_data = get_Data('./data/{}/ml_{}.csv'.format(dataset, dataset), paper_hypergraph_metadata2, data_num=40000000000)
print('full', len(full_data.ranked_event))

# initial_data, snapshot_datas, train_data, val_data, test_data = split_data(full_data, head_rate=0.0, tail_rate=1.0)
# print('initial', len(initial_data.ranked_event))
# print('train', len(train_data.ranked_event))
# print('val', len(val_data.ranked_event))
# print('test', len(test_data.ranked_event))

ngf = get_neighbor_finder(full_data, uniform=False)

label_dict = {}
max_labels = [0,0,0,0,0,0,0,0]
for paper_id in full_data.ranked_event:
    paper_time = full_data.event_timestamp_dict[paper_id]
    paper_year = paper_time//365
    label_dict[paper_id] = np.zeros(8)
    for i in range(8):
        cited_bys, cited_by_edge_types, cited_by_edgeidxs, cited_by_edge_times, cited_by_edge_directions = ngf.find_future_links(paper_id,
                                                                                                   365*(1+i+paper_year))

        label_dict[paper_id][i] = len(cited_bys)

        if label_dict[paper_id][i]>max_labels[i]:
            max_labels[i] = label_dict[paper_id][i]
    if(label_dict[paper_id][0]>0):
        print(paper_id, paper_time, label_dict[paper_id])
print(max_labels)
with open('label_dicts_citation_aps.pkl', 'wb') as file:
    pickle.dump(label_dict, file)
