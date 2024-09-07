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

from utils.utils import *

from torch_geometric.nn import Linear, HGTConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool

sys.path.append(r'dynamicEvent')

import torch
from tqdm import tqdm
from torch_geometric.data import Batch, HeteroData
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

from model.event_tgn import TGN, Decoder

from dataloader import get_Data, get_HeteroData, split_data, split_heterodata
from neighbor_finder import get_neighbor_finder, get_hetero_neighbor_finder
from event_constructor_heterograph import get_batch_of_data, get_example_data

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


paper_hypergraph_metadata2 = {
    'node_types': {0: 'paper', 1: 'author', 2: 'venue', 3: 'keyword'},
    'edge_types': {
        0: ['paper', 'author'],
        1: ['paper', 'venue'],
        2: ['paper', 'keyword'],
        3: ['paper', 'paper'],
        -1: ['a', 'a']},
    'label': -1
}

paper_hypergraph_metadata = {
    'node_types': {0: 'user', 1: 'item'},
    'edge_types': {
        0: ['user', 'item'],
        -1: ['a', 'a']},
    'label': -1
}

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)

dataset = 'aps'

get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/checkpoint-wikipedia-{epoch}.pth'

full_data, n_nodes = get_HeteroData('./data/{}/ml_{}_connected_5.csv'.format(dataset, dataset), paper_hypergraph_metadata2, data_num=6000000)

print('full', len(full_data.ranked_event))

initial_data, snapshot_datas, train_data, val_data, test_data = split_heterodata(full_data, head_rate=0.0, tail_rate=1.0)

# print('initial', len(initial_data.ranked_event))
# print('train', len(train_data.ranked_event))
# print('val', len(val_data.ranked_event))
# print('test', len(test_data.ranked_event))
# print('new_val', len(new_node_val_data.ranked_event))
# print('new_test', len(new_node_test_data.ranked_event))

train_ngf = get_hetero_neighbor_finder(train_data, uniform=False)
ngf = get_hetero_neighbor_finder(full_data, uniform=False)

train_rand_sampler = TimeRandEdgeSampler(train_data.sources[train_data.edge_types==3], train_data.destinations[train_data.edge_types==3], train_data.timestamps[train_data.edge_types==3])
val_rand_sampler = TimeRandEdgeSampler(full_data.sources[full_data.edge_types==3], full_data.destinations[full_data.edge_types==3], full_data.timestamps[full_data.edge_types==3], seed=0)
# nn_val_rand_sampler = RandHeteroEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
#                                       seed=1)
test_rand_sampler = TimeRandEdgeSampler(full_data.sources[full_data.edge_types==3], full_data.destinations[full_data.edge_types==3], full_data.timestamps[full_data.edge_types==3], seed=2)
# nn_test_rand_sampler = RandHeteroEdgeSampler(new_node_test_data.sources,
#                                        new_node_test_data.destinations,
#                                        seed=3)
# print(ngf.find_before('paper_0', 0))
# tgn_model = TGN(ngf, 'cpu')
print('constructing tgn model...')

# edge_feature = np.zeros((len(full_data.sources), 100))  # np.load('./data/{}/ml_{}.npy'.format(dataset, dataset))
# edge_feature = np.load('./data/{}/ml_{}.npy'.format(dataset, dataset))
edge_feature = np.zeros((full_data.destinations.shape[0], 2))
# print(edge_feature.shape)

batch_event_data, _ = get_example_data(full_data, paper_hypergraph_metadata2, ngf, train_rand_sampler, event_per_batch=3)
    # print(batch_event_data.collect('x'))
    # print(batch_event_data.edge_index_dict)
    # print(batch_event_data.collect('negative'))
example_event_data = batch_event_data[0]

    # example_neighbor_data = batch_neighbor_data[0]
# print(example_event_data.metadata())
# n_nodes = {'paper':28209, 'author':71678,'venue':17,'keyword':20123,}
# n_nodes = {'user': 20000, 'item': 3000}
device_string = 'cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

tgn = TGN(n_nodes, device, edge_feature, embedding_dimension=32, metadata=example_event_data.metadata(), deliver_to='all')
# node_decoder = Decoder(64,drop=0.1)
# node_decoder = node_decoder.to(device)
# tgn.load_state_dict(torch.load(get_checkpoint_path(2)), strict=True)
criterion = torch.nn.BCELoss()
# classification_criterion = torch.nn.CrossEntropyLoss()
# classification_criterion = torch.nn.KLDivLoss(reduction='batchmean')
classification_criterion = torch.nn.MSELoss()
tgn = tgn.to(device)
optimizer = torch.optim.Adam(tgn.parameters(), lr=5e-4)
# node_optimizer = torch.optim.Adam(node_decoder.parameters(), lr=1e-3)

batch_size = 10

alpha = 0.25
gamma = 2

embs = defaultdict(list)

pretrain_epoch = 1

best_val_ap = 0
def evaluate_link(pos_score, neg_score):

    with torch.no_grad():
        pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
        neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)
    print(torch.mean(pos_score), torch.mean(neg_score))
    # acc = 0.5 * (float(torch.sum(pos_score > 0.5).detach().cpu().item()) / float(pos_score.shape[0]) + float(torch.sum(
    #     neg_score < 0.5).detach().cpu().item()) / float(neg_score.shape[0]))

    acc = accuracy_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
                                              neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
                              np.concatenate([pos_score.detach().cpu().numpy().flatten() > 0.5,
                                              neg_score.detach().cpu().numpy().flatten() > 0.5], axis=0))
    ap = average_precision_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
                                              neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
                              np.concatenate([pos_score.detach().cpu().numpy().flatten(),
                                              neg_score.detach().cpu().numpy().flatten()], axis=0))
        # 0.5 * (float(torch.sum(pos_score).detach().cpu().item()) / float(pos_score.shape[0]) + float(
        # torch.sum(
        #     1 - neg_score).detach().cpu().item()) / float(neg_score.shape[0]))
    auc_score = roc_auc_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
                                              neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
                              np.concatenate([pos_score.detach().cpu().numpy().flatten(),
                                              neg_score.detach().cpu().numpy().flatten()], axis=0))

    return acc, ap, auc_score


def train_one_batch_link(event_graph):
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_negative = event_graph.collect('negative')  # 0 for current event node, 1 for negative sample
    node_timestamp = event_graph.collect('timestamp')
    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    optimizer.zero_grad()
    loss = 0
    author_venue_pos, author_venue_neg = tgn.compute_heteroevent_probabilities(node_x, node_mask, node_negative, node_timestamp, edge_index,
                                                           edge_rel_times, edge_idx, edge_mask, batch)
    # print(out[:2], label[:2])

    venue_pos_score = torch.sigmoid(author_venue_pos)
    venue_neg_score = torch.sigmoid(author_venue_neg)
    # print(torch.mean(pos_score), torch.mean(neg_score))
    with torch.no_grad():
        venue_pos_label = torch.ones_like(venue_pos_score, dtype=torch.float, device=device)
        venue_neg_label = torch.zeros_like(venue_neg_score, dtype=torch.float, device=device)

    # loss += 1 * criterion(author_pos_score, author_pos_label)
    # loss += 1 * criterion(author_neg_score, author_neg_label)

    loss += 1 * criterion(venue_pos_score, venue_pos_label)
    loss += 1 * criterion(venue_neg_score, venue_neg_label)


    # loss += 1 * criterion(neg_reg, neg_label)
    # loss += 0.1*(criterion(torch.sigmoid(pos_reg), pos_label) + criterion(torch.sigmoid(neg_reg), neg_label))

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(tgn.parameters(), 0.5, norm_type=2)
    optimizer.step()
    # print(torch.mean(pos_score.detach()), torch.mean(neg_score.detach()))
    return venue_pos_score.flatten().detach(), venue_neg_score.flatten().detach()

def eval_one_batch_link(event_graph):
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_negative = event_graph.collect('negative')  # 0 for current event node, 1 for negative sample
    node_timestamp = event_graph.collect('timestamp')

    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    loss = 0
    author_venue_pos, author_venue_neg = tgn.compute_heteroevent_probabilities(
        node_x, node_mask, node_negative, node_timestamp, edge_index,
        edge_rel_times, edge_idx, edge_mask, batch)
    # print(out[:2], label[:2])

    venue_pos_score = torch.sigmoid(author_venue_pos)
    venue_neg_score = torch.sigmoid(author_venue_neg)
    # print(torch.mean(pos_score), torch.mean(neg_score))
    with torch.no_grad():
        venue_pos_label = torch.ones_like(venue_pos_score, dtype=torch.float, device=device)
        venue_neg_label = torch.zeros_like(venue_neg_score, dtype=torch.float, device=device)

    loss += 1 * criterion(venue_pos_score, venue_pos_label)
    loss += 1 * criterion(venue_neg_score, venue_neg_label)

    return venue_pos_score.flatten().detach(), venue_neg_score.flatten().detach()

def pass_one_batch_link(event_graph):
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_negative = event_graph.collect('negative')  # 0 for current event node, 1 for negative sample
    node_timestamp = event_graph.collect('timestamp')

    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    loss = 0
    _, _ = tgn.pass_link(
        node_x, node_mask, node_negative, node_timestamp, edge_index,
        edge_rel_times, edge_idx, edge_mask, batch)
    # print(out[:2], label[:2])
    return _, _

print('training...')

for epoch in range(200):

    tgn.train()
    # node_decoder.train()
    event_list = train_data.ranked_event
    num_event = len(event_list)
    batch_num = np.max(train_data.timestamps)#round(num_event / batch_size) + int(num_event % batch_size > 0)
    for _ in range(1):

        tgn.memory.__init_memory__()
        # tgn.memory.restore_memory(memory_backup)
        loss = 0

        author_pos_scores = []
        author_neg_scores = []
        venue_pos_scores = []
        venue_neg_scores = []

        for idx, (batch_event_data1, batch_labels1, batch_event_data2, batch_labels2) in tqdm(enumerate(
                get_batch_of_data(train_data, paper_hypergraph_metadata2, train_ngf, train_rand_sampler, event_per_batch=batch_size)),
                total=batch_num):

            _, _ = pass_one_batch_link(batch_event_data1.to(device))
            if batch_event_data2 is not None:
                venue_pos_score, venue_neg_score = train_one_batch_link(batch_event_data2.to(device))

                venue_pos_scores.append(venue_pos_score)
                venue_neg_scores.append(venue_neg_score)

        venue_pos_scores = torch.cat(venue_pos_scores, dim=0)
        venue_neg_scores = torch.cat(venue_neg_scores, dim=0)

        # acc, ap, auc_score = evaluate_link(author_pos_scores, author_neg_scores)
        #
        # print('train epoch',epoch,'acc', acc, 'ap', ap, 'auc', auc_score)

        acc, ap, auc_score = evaluate_link(venue_pos_scores, venue_neg_scores)

        print('train epoch', epoch, 'acc', acc, 'ap', ap, 'auc', auc_score)
    train_memory_backup = tgn.memory.backup_memory()
    # torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
    '''
    val
    '''
    tgn.eval()
    event_list = val_data.ranked_event
    num_event = len(event_list)
    batch_num = np.max(val_data.timestamps)#round(num_event / batch_size) + int(num_event % batch_size > 0)

    # tgn.memory.__init_memory__()
    # tgn.memory.restore_memory(memory_backup)
    loss = 0

    author_pos_scores = []
    author_neg_scores = []
    venue_pos_scores = []
    venue_neg_scores = []

    for idx, (batch_event_data1, batch_labels1, batch_event_data2, batch_labels2) in tqdm(enumerate(
            get_batch_of_data(val_data, paper_hypergraph_metadata2, ngf, val_rand_sampler, event_per_batch=batch_size)),
            total=batch_num):
        _, _ = pass_one_batch_link(batch_event_data1.to(device))
        if batch_event_data2 is not None:
            venue_pos_score, venue_neg_score = eval_one_batch_link(batch_event_data2.to(device))

            venue_pos_scores.append(venue_pos_score)
            venue_neg_scores.append(venue_neg_score)

    venue_pos_scores = torch.cat(venue_pos_scores, dim=0)
    venue_neg_scores = torch.cat(venue_neg_scores, dim=0)

    # acc, ap, auc_score = evaluate_link(author_pos_scores, author_neg_scores)
    #
    # print('val epoch', epoch, 'acc', acc, 'ap', ap, 'auc', auc_score)
    acc, ap, auc_score = evaluate_link(venue_pos_scores, venue_neg_scores)
    if best_val_ap<ap:
        best_val_ap = ap
        print('######################## best #################################')
        print('######################## best #################################')
        print('######################## best #################################')


    print('val epoch',epoch,'acc', acc, 'ap', ap, 'auc', auc_score)
    #
    # val_memory_backup = tgn.memory.backup_memory()
    # Restore memory we had at the end of training to be used when validating on new nodes.
    # Also backup memory after validation so it can be used for testing (since test edges are
    # strictly later in time than validation edges)
    # tgn.memory.restore_memory(train_memory_backup)
    # del train_memory_backup

    # '''
    # new_val
    # '''
    # event_list = new_node_val_data.ranked_event
    # num_event = len(event_list)
    # batch_num = round(num_event / batch_size) + int(num_event % batch_size > 0)
    #
    # # tgn.memory.__init_memory__()
    # # tgn.memory.restore_memory(memory_backup)
    # loss = 0
    #
    # pos_scores = []
    # neg_scores = []
    #
    # for idx, (batch_event_data, batch_labels) in tqdm(enumerate(
    #         get_batch_of_data(new_node_val_data, paper_hypergraph_metadata2, ngf, val_rand_sampler, event_per_batch=batch_size)),
    #         total=batch_num):
    #     pos_score, neg_score = eval_one_batch_link(batch_event_data.to(device))
    #
    #     pos_scores.append(pos_score)
    #     neg_scores.append(neg_score)
    #
    # pos_scores = torch.cat(pos_scores, dim=0)
    # neg_scores = torch.cat(neg_scores, dim=0)
    #
    # acc, ap, auc_score = evaluate_link(pos_scores, neg_scores)
    #
    # print('new val epoch',epoch,'acc', acc, 'ap', ap, 'auc', auc_score)
    #
    # tgn.memory.restore_memory(val_memory_backup)

    '''
    test
    '''

    event_list = test_data.ranked_event
    num_event = np.max(test_data.timestamps)#len(event_list)
    batch_num = round(num_event / batch_size) + int(num_event % batch_size > 0)

    # tgn.memory.__init_memory__()
    # tgn.memory.restore_memory(memory_backup)
    loss = 0

    author_pos_scores = []
    author_neg_scores = []
    venue_pos_scores = []
    venue_neg_scores = []

    for idx, (batch_event_data1, batch_labels1, batch_event_data2, batch_labels2) in tqdm(enumerate(
            get_batch_of_data(test_data, paper_hypergraph_metadata2, ngf, test_rand_sampler, event_per_batch=batch_size)),
            total=batch_num):
        _, _ = pass_one_batch_link(batch_event_data1.to(device))
        if batch_event_data2 is not None:
            venue_pos_score, venue_neg_score = eval_one_batch_link(batch_event_data2.to(device))

            venue_pos_scores.append(venue_pos_score)
            venue_neg_scores.append(venue_neg_score)

    venue_pos_scores = torch.cat(venue_pos_scores, dim=0)
    venue_neg_scores = torch.cat(venue_neg_scores, dim=0)

    # acc, ap, auc_score = evaluate_link(author_pos_scores, author_neg_scores)
    #
    # print('test epoch', epoch, 'acc', acc, 'ap', ap, 'auc', auc_score)

    acc, ap, auc_score = evaluate_link(venue_pos_scores, venue_neg_scores)

    print('test epoch', epoch, 'acc', acc, 'ap', ap, 'auc', auc_score)

    # tgn.memory.restore_memory(val_memory_backup)
    # del val_memory_backup
    # '''
    #     new_test
    #     '''
    # event_list = new_node_test_data.ranked_event
    # num_event = len(event_list)
    # batch_num = round(num_event / batch_size) + int(num_event % batch_size > 0)
    #
    # # tgn.memory.__init_memory__()
    # # tgn.memory.restore_memory(memory_backup)
    # loss = 0
    #
    # pos_scores = []
    # neg_scores = []
    #
    # for idx, (batch_event_data, batch_labels) in tqdm(enumerate(
    #         get_batch_of_data(new_node_test_data, paper_hypergraph_metadata2, ngf, nn_test_rand_sampler,
    #                           event_per_batch=batch_size)),
    #         total=batch_num):
    #     pos_score, neg_score = eval_one_batch_link(batch_event_data.to(device))
    #
    #     pos_scores.append(pos_score)
    #     neg_scores.append(neg_score)
    #
    # pos_scores = torch.cat(pos_scores, dim=0)
    # neg_scores = torch.cat(neg_scores, dim=0)
    #
    # acc, ap, auc_score = evaluate_link(pos_scores, neg_scores)
    #
    # print('new test epoch', epoch, 'acc', acc, 'ap', ap, 'auc', auc_score)

