from collections import defaultdict
from time import time
import numpy as np
import random
import pandas as pd
import pickle
import datetime
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sys.path.append(r'dynamicEvent')

import torch
from tqdm import tqdm
from torch_geometric.data import Batch, HeteroData
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from model.event_tgn import TGN, Decoder

from dataloader import get_Data, split_data
from neighbor_finder import get_neighbor_finder
from event_constructor import get_batch_of_data

paper_hypergraph_metadata2 = {
    'node_types' : {0:'paper', 1:'author', 2:'venue', 3:'keyword'},
    'edge_types' : {
    # 0 : ['paper', 'author'],
    # 1 : ['paper', 'venue'],
    # 2 : ['paper', 'keyword'],
    3 : ['paper', 'paper'],
    -1 : ['a', 'a']},
    'label' : -1
}

paper_hypergraph_metadata2 = {
    'node_types' : {0:'user', 1:'item'},
    'edge_types' : {
    0 : ['user', 'item'],
    -1 : ['a', 'a']},
    'label' : -1
}

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)

get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/checkpoint-wikipedia-{epoch}.pth'

full_data = get_Data('./data/wiki/ml_wikipedia.csv', paper_hypergraph_metadata2, data_num=160000)

print('full', len(full_data.ranked_event))

initial_data, snapshot_datas, train_data, val_data, test_data = split_data(full_data, head_rate=0.0, tail_rate=1.0)

print('initial', len(initial_data.ranked_event))
print('train', len(train_data.ranked_event))
print('val', len(val_data.ranked_event))
print('test', len(test_data.ranked_event))

ngf = get_neighbor_finder(full_data, uniform=False)
# print(ngf.find_before('paper_0', 0))
# tgn_model = TGN(ngf, 'cpu')
print('constructing tgn model...')

edge_feature = np.load('./data/wiki/ml_{}.npy'.format('wikipedia'))

for batch_event_data, batch_neighbor_data, _ in get_batch_of_data(train_data, paper_hypergraph_metadata2, ngf, event_per_batch = 3):
    print(batch_event_data.collect('x'))
    print(batch_event_data.edge_index_dict)
    example_event_data = batch_event_data[0]
    example_neighbor_data = batch_neighbor_data[0]
    break
print(example_event_data.metadata())
# n_nodes = {'paper':5000000, 'author':1000000,'venue':100000,'keyword':1000000,}
n_nodes = {'user':5000000, 'item':1000000}
device_string = 'cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')
tgn = TGN(n_nodes, device, edge_feature,embedding_dimension=64,metadata=example_event_data.metadata())
node_decoder = Decoder(64,drop=0.1)
node_decoder = node_decoder.to(device)
tgn.load_state_dict(torch.load(get_checkpoint_path(1)), strict=True)
criterion = torch.nn.BCELoss()
classification_criterion = torch.nn.BCELoss()#torch.nn.CrossEntropyLoss(reduction = 'mean')
node_optimizer = torch.optim.Adam(node_decoder.parameters(), lr=1e-3)

def gen_node_emb():
    labels = []
    embs = defaultdict(list)

    event_list = full_data.ranked_event
    batch_size = 200
    num_event = len(event_list)
    batch_num = round(num_event / batch_size) + int(num_event % batch_size > 0)

    for idx, (batch_event_data, batch_neighbor_data, batch_labels) in tqdm(enumerate(
            get_batch_of_data(full_data, paper_hypergraph_metadata2, ngf, event_per_batch=batch_size)),
            total=batch_num):
        event_graph, neighbor_graph = batch_event_data.to(device), batch_neighbor_data.to(device)

        # batch = event_graph.collect('batch')
        # mask = event_graph.collect('node_mask')
        # pos_scores, neg_scores, pred_class = tgn.compute_event_probabilities(batch_event_data.to(device), batch_neighbor_data.to(device))
        event_node_embedding, _, _, _ = tgn.compute_temporal_embeddings(
            event_graph, neighbor_graph)

        labels.append(batch_labels)

        for node_type in example_event_data.metadata()[0]:
            embs[node_type].append(event_node_embedding[node_type])
    labels = torch.cat(labels, dim=0).view(-1)
    for node_type in example_event_data.metadata()[0]:
        embs[node_type] = torch.cat(embs[node_type], dim=0)
    return labels, embs

try:
    labels = torch.load('wiki_label.pth')
    embs = torch.load('wiki_embs.pth')
except:
    labels, embs = gen_node_emb()

    torch.save(labels, 'wiki_label.pth')
    torch.save(embs, 'wiki_embs.pth')



for epoch in range(20):
    event_list = train_data.ranked_event
    batch_size = 200
    num_event = len(event_list)
    batch_num = round(num_event / batch_size) + int(num_event % batch_size > 0)

    for idx, event in enumerate(event_list):
