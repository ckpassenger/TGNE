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
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from model.event_tgn import TGN, Decoder

from dataloader import get_Data, split_data
from neighbor_finder import get_neighbor_finder
from event_constructor import get_batch_of_data

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

paper_hypergraph_metadata = {
    'node_types' : {0:'paper', 1:'author', 2:'venue', 3:'keyword'},
    'edge_types' : {
    0 : ['paper', 'author'],
    1 : ['paper', 'venue'],
    2 : ['paper', 'keyword'],
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

dataset = 'wikipedia'

get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/checkpoint-wikipedia-{epoch}.pth'

full_data = get_Data('./data/{}/ml_{}.csv'.format(dataset, dataset), paper_hypergraph_metadata2, data_num=4000000)

print('full', len(full_data.ranked_event))

initial_data, snapshot_datas, train_data, val_data, test_data = split_data(full_data, head_rate=0.0, tail_rate=0.1)

print('initial', len(initial_data.ranked_event))
print('train', len(train_data.ranked_event))
print('val', len(val_data.ranked_event))
print('test', len(test_data.ranked_event))

ngf = get_neighbor_finder(full_data, uniform=True)
# print(ngf.find_before('paper_0', 0))
# tgn_model = TGN(ngf, 'cpu')
print('constructing tgn model...')

edge_feature = np.zeros((len(full_data.sources), 100))#np.load('./data/{}/ml_{}.npy'.format(dataset, dataset))

for batch_event_data, _ in get_batch_of_data(train_data, paper_hypergraph_metadata2, ngf, event_per_batch = 3):
    print(batch_event_data.collect('x'))
    print(batch_event_data.edge_index_dict)
    example_event_data = batch_event_data[0]
    # example_neighbor_data = batch_neighbor_data[0]
    break
print(example_event_data.metadata())
# n_nodes = {'paper':800000, 'author':800000,'venue':1000,'keyword':100000,}
n_nodes = {'user':5000000, 'item':1000000}
device_string = 'cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
tgn = TGN(n_nodes, device, edge_feature,embedding_dimension=64,metadata=example_event_data.metadata())
# node_decoder = Decoder(64,drop=0.1)
# node_decoder = node_decoder.to(device)
# tgn.load_state_dict(torch.load(get_checkpoint_path(2)), strict=True)
criterion = torch.nn.BCELoss()
# classification_criterion = torch.nn.CrossEntropyLoss()
# classification_criterion = torch.nn.KLDivLoss(reduction='batchmean')
classification_criterion = torch.nn.MSELoss()
tgn = tgn.to(device)
optimizer = torch.optim.Adam(tgn.parameters(), lr=1e-1)
# node_optimizer = torch.optim.Adam(node_decoder.parameters(), lr=1e-3)

batch_size = 64

alpha = 0.25
gamma = 2

embs = defaultdict(list)

pretrain_epoch = 1

def myPCC(pred, label):
    pred_mean, label_mean = np.mean(pred, axis = 0), np.mean(label, axis = 0)
    pred_std, label_std = np.std(pred, axis = 0), np.std(label, axis = 0)
    return np.around(np.mean((pred-pred_mean)*(label - label_mean)/(pred_std*label_std), axis = 0), 4)

def evaluate(pred, label):
    print(pred[-5:], label[-5:])
    RMSE = np.zeros(5)
    MAE = np.zeros(5)
    PCC = np.zeros(5)
    for i in range(5):
        RMSE[i] = np.sqrt(mean_squared_error(np.log(pred[:,i]+1), np.log(label[:,i]+1)))
        MAE[i] = mean_absolute_error(np.log(pred[:,i]+1), np.log(label[:,i]+1))
        PCC[i] = myPCC(np.log(pred[:,i]+1), np.log(label[:,i]+1))

    return RMSE, MAE, PCC

def evaluate_link(pos_score, neg_score):
    with torch.no_grad():
        pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
        neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)

    acc = 0.5 * (float(torch.sum(pos_score > 0.5).detach().cpu().item()) / float(pos_score.shape[0]) + float(torch.sum(
            neg_score < 0.5).detach().cpu().item()) / float(neg_score.shape[0]*neg_score.shape[1]))
    ap = 0.5 * (float(torch.sum(pos_score).detach().cpu().item()) / float(pos_score.shape[0]) + float(
        torch.sum(
            1 - neg_score).detach().cpu().item()) / float(neg_score.shape[0] * neg_score.shape[1]))
    auc_score = roc_auc_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
                                               neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
                               np.concatenate([pos_score.detach().cpu().numpy().flatten(),
                                               neg_score.detach().cpu().numpy().flatten()], axis=0))

    return acc, ap, auc_score

def train_one_batch_link(event_graph):
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_timestamp = event_graph.collect('timestamp')

    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    optimizer.zero_grad()
    loss=0
    pos_score, neg_score = tgn.compute_event_probabilities(node_x, node_mask, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)
    # print(out[:2], label[:2])
    pos_score = torch.sigmoid(pos_score)
    neg_score = torch.sigmoid(neg_score)

    with torch.no_grad():
        pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
        neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)

    loss += 1 * criterion(pos_score, pos_label)
    loss += 1 * criterion(neg_score, neg_label)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(tgn.parameters(), 0.5, norm_type=2)
    optimizer.step()

    return pos_score, neg_score

def eval_one_batch_link(event_graph):
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_timestamp = event_graph.collect('timestamp')

    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    loss=0
    pos_score, neg_score = tgn.compute_event_probabilities(node_x, node_mask, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)
    # print(out[:2], label[:2])
    pos_score = torch.sigmoid(pos_score)
    neg_score = torch.sigmoid(neg_score)

    with torch.no_grad():
        pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
        neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)

    loss += 1 * criterion(pos_score, pos_label)
    loss += 1 * criterion(neg_score, neg_label)

    return pos_score, neg_score

def train_one_batch(event_graph, label):
    # for _ in range(1):
    #     prob, neg = model.update(data.collect('x'), data.edge_index_dict, data.collect('edge_rel_times'), data.collect('batch'), data['paper'].mask)
    #     optimizer.zero_grad()
    #     pre_loss = bceloss(prob, torch.ones_like(prob)) + bceloss(neg, torch.zeros_like(neg))
    #     pre_loss.backward()
    #     optimizer.step()

    # extract information from input graph
    # node information
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_timestamp = event_graph.collect('timestamp')

    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    optimizer.zero_grad()
    out = tgn.predict(node_x, node_mask, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)
    # print(out[:2], label[:2])
    loss = torch.nn.functional.mse_loss(torch.log(out+1), torch.log(label+1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tgn.parameters(), 0.5, norm_type=2)
    optimizer.step()

    return out, loss.detach()

def eval_one_batch(event_graph):
    # extract information from input graph
    # node information
    node_x = event_graph.collect('x')  # for feature & memory
    node_mask = event_graph.collect('node_mask')  # 0 for current event center node, 1 for event nodes, 2 for neighbors
    node_timestamp = event_graph.collect('timestamp')

    # edge information
    edge_index = event_graph.collect('edge_index')
    edge_rel_times = event_graph.collect('edge_rel_times')
    edge_idx = event_graph.collect('edge_idxs')  # for feature
    edge_mask = event_graph.collect('edge_mask')  # 0 for current event, 1 for neighbors

    batch = event_graph.collect('batch')

    out = tgn.predict(node_x, node_mask, node_timestamp, edge_index, edge_rel_times, edge_idx, edge_mask, batch)

    return out

for epoch in range(20):
    # tgn.memory.__init_memory__()
    # event_list = full_data.ranked_event
    # num_event = len(event_list)
    # batch_num = round(num_event / batch_size)+int(num_event%batch_size>0)
    # train_loss = 0
    # pos_score_pred = 0#{'paper':0, 'author':0, 'venue':0}
    # neg_score_pred = 0#{'paper':0, 'author':0, 'venue':0}
    # acc = 0
    # ap = 0
    # auc_score = 0
    # # batch_size = 400
    # tgn.train()
    # for idx,(batch_event_data, batch_neighbor_data, _) in tqdm(enumerate(get_batch_of_data(full_data, paper_hypergraph_metadata2, ngf, event_per_batch = batch_size)), total = batch_num):
    #     if epoch > pretrain_epoch:
    #         break
    #     for node_type in example_event_data.metadata()[0]:
    #         if torch.sum(batch_event_data[node_type]['node_mask']) < torch.sum(batch_neighbor_data[node_type]['node_mask'])-batch_size:
    #             print(node_type,batch_event_data[node_type]['x'])
    #             print(node_type,batch_neighbor_data[node_type]['x'][batch_neighbor_data[node_type]['node_mask']==1])
    #
    #     t1=time.time()
    #     pos_scores, neg_scores, pos_score_reg, neg_score_reg = tgn.compute_event_probabilities(batch_event_data.to(device), batch_neighbor_data.to(device))
    #     loss=0
    #     t2=time.time()
    #
    #     # for node_type in example_event_data.metadata()[0]:
    #
    #     pos_score = torch.sigmoid(pos_scores)
    #     neg_score = torch.sigmoid(neg_scores)
    #     print(torch.mean(pos_score).data, torch.mean(neg_score).data)
    #     pos_score_pred += torch.mean(pos_score).data
    #     neg_score_pred += torch.mean(neg_score).data
    #
    #     with torch.no_grad():
    #         pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
    #         neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)
    #
    #     loss += 1 * criterion(pos_score, pos_label)
    #     loss += 1 * criterion(neg_score, neg_label)
    #
    #     # loss += torch.mean(torch.clamp(neg_score_reg, -5, 5)-torch.clamp(pos_score_reg, -5, 5))
    #
    #     acc += 0.5 * (float(torch.sum(pos_score > 0.5).detach().cpu().item()) / float(pos_score.shape[0]) + float(torch.sum(
    #         neg_score < 0.5).detach().cpu().item()) / float(neg_score.shape[0]*neg_score.shape[1]))
    #     ap += 0.5 * (float(torch.sum(pos_score).detach().cpu().item()) / float(pos_score.shape[0]) + float(
    #         torch.sum(
    #             1 - neg_score).detach().cpu().item()) / float(neg_score.shape[0] * neg_score.shape[1]))
    #     auc_score += roc_auc_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
    #                                                neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
    #                                np.concatenate([pos_score.detach().cpu().numpy().flatten(),
    #                                                neg_score.detach().cpu().numpy().flatten()], axis=0))
    #         # loss += (-torch.mean(pos_score) + torch.mean(neg_score))
    #     train_loss += loss.data
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(tgn.parameters(), 0.25, norm_type=2)
    #
    #     # print(tgn.memory_updater.memory_updater['user'].weight_ih.grad)
    #     # print(tgn.embedding_module.convs[0].out_lin['user'].weight.grad)
    #
    #     optimizer.step()
    #     tgn.memory.detach_memory()
    #     # t3 = time.time()
    # if epoch <= pretrain_epoch:
    #     torch.save(tgn.state_dict(), get_checkpoint_path(2))
    #
    # print('train loss epoch', epoch, train_loss/(idx+1))
    # # for node_type in example_event_data.metadata()[0]:
    # print('train score epoch', epoch, pos_score_pred / (idx+1), neg_score_pred / (idx+1), 'acc', acc/(idx+1), 'ap', ap/(idx+1), 'auc', auc_score/(idx+1))
    #
    # if epoch<=pretrain_epoch:
    #     continue
########################################################################################################
    # node classification

#######################################################################################################3

    # memory_backup = tgn.memory.backup_memory()
    # batch_size = 100
    # tgn.init_classification()
    tgn.train()
    # node_decoder.train()
    event_list = train_data.ranked_event
    num_event = len(event_list)
    batch_num = round(num_event / batch_size) + int(num_event%batch_size>0)
    for _ in range(1):

        tgn.memory.__init_memory__()
        # tgn.memory.restore_memory(memory_backup)
        loss = 0

        y_trues = []
        y_preds = []

        for idx, (batch_event_data, batch_labels) in tqdm(enumerate(
                get_batch_of_data(train_data, paper_hypergraph_metadata2, ngf, event_per_batch=batch_size)),
                total=batch_num):
            t1 = time.time()
            pred_class, train_loss = train_one_batch(batch_event_data.to(device), batch_labels.to(device))

            t2 = time.time()
            # print('forward',t2-t1)
            # if len(event_rep[batch_labels == 1.0]) > 0:
            #     print(event_graph.collect('x')['user'][batch_labels == 1.0], event_rep[batch_labels == 1.0][:, 0])

            # if len(pred_class[batch_labels == 1.0]) == 0:
            #     continue
            #
            # pred_class_pos = pred_class[batch_labels == 1.0]
            # label_pos = torch.ones_like(pred_class_pos)
            # indice = torch.randperm(len(pred_class[batch_labels != 1.0]))[:len(pred_class_pos)]
            # pred_class_neg = pred_class[batch_labels != 1.0][indice]
            # label_neg = torch.zeros_like(pred_class_neg)#batch_labels[batch_labels != 1.0][indice]

            pred_class_batch = pred_class#[torch.sum(batch_labels, dim=-1) > 0]  #
            # pred_class_batch = torch.cat([pred_class_pos, pred_class_neg], dim=0).view(-1)
            label_batch = batch_labels#[torch.sum(batch_labels, dim=-1) > 0]  #
            # label_batch = torch.cat([label_pos, label_neg], dim=0).view(-1)

            # ce_loss = torch.nn.functional.cross_entropy(torch.sigmoid(pred_class.squeeze(1)), batch_labels.to(device).long(),
            #                                             reduction='none')  # important to add reduction='none' to keep per-batch-item loss
            # pt = torch.exp(-ce_loss)
            # focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
            # train_loss = focal_loss


            loss += train_loss
            # tgn.memory.detach_memory()
            del (batch_event_data)
            y_true = label_batch.float().numpy()
            # y_pred = (pred_class_batch.sigmoid()).data.float().cpu().numpy().flatten()
            y_pred = torch.relu(pred_class_batch).data.float().cpu().numpy()#np.argsort(pred_class.data.cpu().numpy(), axis=-1)[:,-1]

            y_trues.append(y_true)
            y_preds.append(y_pred)

        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
        # print(np.sum(y_trues))

        # print(np.mean(y_trues), np.min(y_preds), np.max(y_preds), np.mean(y_preds))
        # node_ap = 0#average_precision_score(y_trues.astype('int'), (y_preds>0.5).astype('int'))
        # test_res = []
        # for ai, bi in zip(y_trues, np.argsort(-y_preds, axis=-1)):
        #     test_res += [ai[bi]]
        # test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        # print('Last Test NDCG: %.4f' % np.average(test_ndcg))
        # test_mrr = mean_reciprocal_rank(test_res)
        # print('Last Test MRR:  %.4f' % np.average(test_mrr))

        RMSE, MAE, PCC = evaluate(y_preds, y_trues)

        print('RMSE', RMSE, 'MAE', MAE, 'PCC', PCC)

        auc = 0#roc_auc_score(y_trues, y_preds)
        print('train node loss epoch', epoch, loss / (idx+1))#, 'ndcg', np.average(test_ndcg), 'mrr', np.average(test_mrr), 'auc', auc)
        # if epoch<4:
        #     break
    # torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
    '''
    val
    '''

    loss = 0
    node_loss = 0
    acc=0
    ap=0
    auc_score=0
    y_trues = []
    y_preds = []
    pos_score_pred = 0 #{'paper': 0, 'author': 0, 'venue': 0}
    neg_score_pred = 0 #{'paper': 0, 'author': 0, 'venue': 0}
    event_list = val_data.ranked_event
    num_event = len(event_list)
    batch_num = round(num_event / batch_size) + int(num_event%batch_size>0)
    tgn.eval()
    # node_decoder.eval()
    for idx, (batch_event_data, batch_labels) in tqdm(enumerate(
            get_batch_of_data(val_data, paper_hypergraph_metadata2, ngf, event_per_batch = batch_size)), total=batch_num):
        # pos_scores, neg_scores = tgn.compute_event_probabilities(batch_event_data.to(device), batch_neighbor_data.to(device))

        # for node_type in example_event_data.metadata()[0]:

        # batch = event_graph.collect('batch')
        # mask = event_graph.collect('node_mask')
        # # pos_scores, neg_scores, pred_class = tgn.compute_event_probabilities(batch_event_data.to(device), batch_neighbor_data.to(device))
        # event_node_embedding, updated_event_node_embedding, negative_node_embedding, updated_negative_node_embedding = tgn.compute_temporal_embeddings(
        #     event_graph)
        #
        # pos_pool, updated_pos_pool = 0, 0
        # neg_pool, updated_neg_pool = 0, 0
        # for node_type in example_event_data.metadata()[0]:
        #     if torch.sum(mask[node_type] == 0) > 0:
        #         pos_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 0],
        #                                      batch[node_type][mask[node_type] == 0])
        #         neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 0],
        #                                      batch[node_type][mask[node_type] == 0])
        #
        #     if torch.sum(mask[node_type] == 1) > 0:
        #         updated_pos_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1],
        #                                              batch[node_type][mask[node_type] == 1])
        #         shuffle_index = torch.randperm(updated_pos_pool.shape[0])
        #         # updated_neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])[shuffle_index]#
        #         updated_neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])[shuffle_index]
        #
        #
        # pos_score = torch.sum(
        #     pos_pool * updated_pos_pool,
        #     dim=-1).view(-1, 1)
        # neg_score = torch.sum(
        #     neg_pool * updated_neg_pool,
        #     dim=-1).view(-1, 1)
        #
        # # pos_score, _ = tgn.distance(
        # #     torch.cat([pos_pool + updated_pos_pool],
        # #               dim=-1))  # self.distance(self.hadamad_distance(pos_pool, updated_pos_pool))
        # # neg_score, _ = tgn.distance(
        # #     torch.cat([neg_pool + updated_neg_pool], dim=-1))
        # pos_score = pos_score.sigmoid()
        # neg_score = neg_score.sigmoid()
        #
        # pos_score_pred += torch.mean(pos_score).data
        # neg_score_pred += torch.mean(neg_score).data
        #
        # with torch.no_grad():
        #     pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
        #     neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)
        #
        # loss += 1 * criterion(pos_score, pos_label)
        # loss += 1 * criterion(neg_score, neg_label)
        # acc += 0.5 * (float(torch.sum(pos_score > 0.5).detach().cpu().item()) / float(pos_score.shape[0]) + float(
        #     torch.sum(
        #         neg_score < 0.5).detach().cpu().item()) / float(neg_score.shape[0] * neg_score.shape[1]))
        # ap += 0.5 * (float(torch.sum(pos_score).detach().cpu().item()) / float(pos_score.shape[0]) + float(
        #     torch.sum(
        #         1 - neg_score).detach().cpu().item()) / float(neg_score.shape[0] * neg_score.shape[1]))
        # auc_score += roc_auc_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
        #                                            neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
        #                            np.concatenate([pos_score.detach().cpu().numpy().flatten(),
        #                                            neg_score.detach().cpu().numpy().flatten()], axis=0))


        pred_class = eval_one_batch(batch_event_data.to(device))

        # if len(pred_class[batch_labels == 1.0]) == 0:
        #     continue
        #
        pred_class_pos = pred_class[batch_labels == 1.0]
        label_pos = torch.ones_like(pred_class_pos)
        indice = torch.randperm(len(pred_class[batch_labels != 1.0]))[:len(pred_class_pos)]
        pred_class_neg = pred_class[batch_labels != 1.0][indice]
        label_neg = torch.zeros_like(pred_class_neg)  # batch_labels[batch_labels != 1.0][indice]

        pred_class_batch = torch.relu(pred_class)  #
        # pred_class_batch = torch.cat([pred_class_pos, pred_class_neg], dim=0).view(-1)
        label_batch = batch_labels#[torch.sum(batch_labels, dim=-1) > 0]  #
        # label_batch = torch.cat([label_pos, label_neg], dim=0).view(-1)

        # ce_loss = torch.nn.functional.cross_entropy(torch.sigmoid(pred_class.squeeze(1)), batch_labels.to(device).long(),
        #                                             reduction='none')  # important to add reduction='none' to keep per-batch-item loss
        # pt = torch.exp(-ce_loss)
        # focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
        # train_loss = focal_loss
        # ce_loss = torch.nn.functional.cross_entropy(pred_class, batch_labels.to(device),
        #                                             reduction='none')  # important to add reduction='none' to keep per-batch-item loss
        # pt = torch.exp(-ce_loss)
        # focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
        # node_loss += focal_loss

        y_true = label_batch.float().numpy()
        y_pred = torch.relu(pred_class_batch).data.float().cpu().numpy()#(pred_class_batch.sigmoid()).data.float().cpu().numpy().flatten()#np.argsort(pred_class.data.cpu().numpy(), axis=-1)[:, -1]

        y_trues.append(y_true)
        y_preds.append(y_pred)

    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    print(np.sum(y_trues))
    print(np.mean(y_trues), np.min(y_preds), np.max(y_preds), np.mean(y_preds))

    # node_ap = 0#average_precision_score(y_trues.astype('int'), (y_preds>0.5).astype('int'))
    # auc = 0#roc_auc_score(y_trues, y_preds)
    # print('val loss epoch', epoch, loss.data / (idx+1), node_loss.data / (idx+1), 'ap', node_ap, 'auc', auc)

    # test_res = []
    # for ai, bi in zip(y_trues, np.argsort(-y_preds, axis=-1)):
    #     test_res += [ai[bi]]
    # test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    # print('Last Test NDCG: %.4f' % np.average(test_ndcg))
    # test_mrr = mean_reciprocal_rank(test_res)
    # print('Last Test MRR:  %.4f' % np.average(test_mrr))

    RMSE, MAE, PCC = evaluate(y_preds, y_trues)

    print('RMSE', RMSE, 'MAE', MAE, 'PCC', PCC)

    print('val node loss epoch', epoch, loss / (idx + 1))#, 'ndcg', np.average(test_ndcg), 'mrr', np.average(test_mrr),'auc', auc)


    '''
    test
    '''

    loss = 0
    node_loss = 0
    acc = 0
    ap = 0
    auc_score = 0
    y_trues = []
    y_preds = []
    pos_score_pred = 0#{'paper': 0, 'author': 0, 'venue': 0}
    neg_score_pred = 0#{'paper': 0, 'author': 0, 'venue': 0}
    event_list = test_data.ranked_event
    num_event = len(event_list)
    batch_num = round(num_event / batch_size) + int(num_event%batch_size>0)

    for idx,(batch_event_data, batch_labels) in tqdm(enumerate(
            get_batch_of_data(test_data, paper_hypergraph_metadata2, ngf, event_per_batch=batch_size)), total=batch_num):
        # batch = event_graph.collect('batch')
        # mask = event_graph.collect('node_mask')
        # # pos_scores, neg_scores, pred_class = tgn.compute_event_probabilities(batch_event_data.to(device), batch_neighbor_data.to(device))
        # event_node_embedding, updated_event_node_embedding, negative_node_embedding, updated_negative_node_embedding = tgn.compute_temporal_embeddings(
        #     event_graph)
        #
        # pos_pool, updated_pos_pool = 0, 0
        # neg_pool, updated_neg_pool = 0, 0
        # for node_type in example_event_data.metadata()[0]:
        #     if torch.sum(mask[node_type] == 0) > 0:
        #         pos_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 0],
        #                                      batch[node_type][mask[node_type] == 0])
        #         neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 0],
        #                                      batch[node_type][mask[node_type] == 0])
        #
        #     if torch.sum(mask[node_type] == 1) > 0:
        #         updated_pos_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1],
        #                                              batch[node_type][mask[node_type] == 1])
        #         shuffle_index = torch.randperm(updated_pos_pool.shape[0])
        #         # updated_neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])[shuffle_index]#
        #         updated_neg_pool += global_mean_pool(event_node_embedding[node_type][mask[node_type] == 1], batch[node_type][mask[node_type] == 1])[shuffle_index]
        #
        # pos_score = torch.sum(
        #     pos_pool * updated_pos_pool,
        #     dim=-1).view(-1, 1)
        # neg_score = torch.sum(
        #     neg_pool * updated_neg_pool,
        #     dim=-1).view(-1, 1)
        #
        # # pos_score, _ = tgn.distance(
        # #     torch.cat([pos_pool + updated_pos_pool],
        # #               dim=-1))  # self.distance(self.hadamad_distance(pos_pool, updated_pos_pool))
        # # neg_score, _ = tgn.distance(
        # #     torch.cat([neg_pool + updated_neg_pool], dim=-1))
        # pos_score = pos_score.sigmoid()
        # neg_score = neg_score.sigmoid()
        #
        # pos_score_pred += torch.mean(pos_score).data
        # neg_score_pred += torch.mean(neg_score).data
        #
        # with torch.no_grad():
        #     pos_label = torch.ones_like(pos_score, dtype=torch.float, device=device)
        #     neg_label = torch.zeros_like(neg_score, dtype=torch.float, device=device)
        #
        # loss += 1 * criterion(pos_score, pos_label)
        # loss += 1 * criterion(neg_score, neg_label)
        # acc += 0.5 * (float(torch.sum(pos_score > 0.5).detach().cpu().item()) / float(pos_score.shape[0]) + float(
        #     torch.sum(
        #         neg_score < 0.5).detach().cpu().item()) / float(neg_score.shape[0] * neg_score.shape[1]))
        # ap += 0.5 * (float(torch.sum(pos_score).detach().cpu().item()) / float(pos_score.shape[0]) + float(
        #     torch.sum(
        #         1 - neg_score).detach().cpu().item()) / float(neg_score.shape[0] * neg_score.shape[1]))
        # auc_score += roc_auc_score(np.concatenate([pos_label.detach().cpu().numpy().flatten().astype('int'),
        #                                            neg_label.detach().cpu().numpy().flatten().astype('int')], axis=0),
        #                            np.concatenate([pos_score.detach().cpu().numpy().flatten(),
        #                                            neg_score.detach().cpu().numpy().flatten()], axis=0))
        #
        # event_rep = 0
        # for node_type in example_event_data.metadata()[0]:
        #     event_mask =  mask[node_type]<2
        #     event_rep += global_mean_pool(updated_event_node_embedding[node_type][event_mask],
        #                                   batch[node_type][event_mask])
            # event_rep += event_node_embedding[node_type]
        # print(torch.mean(event_rep))
        pred_class = eval_one_batch(batch_event_data.to(device))
        # if len(event_rep[batch_labels == 1.0]) > 0:
        #     print(event_graph.collect('x')['user'][batch_labels == 1.0])
        #     print(event_rep[batch_labels == 1.0][:, 0])
        # if len(pred_class[batch_labels == 1.0]) == 0:
        #     continue
        #
        pred_class_pos = pred_class[batch_labels == 1.0]
        label_pos = torch.ones_like(pred_class_pos)
        indice = torch.randperm(len(pred_class[batch_labels != 1.0]))[:len(pred_class_pos)]
        pred_class_neg = pred_class[batch_labels != 1.0][indice]
        label_neg = torch.zeros_like(pred_class_neg)  # batch_labels[batch_labels != 1.0][indice]

        pred_class_batch = torch.relu(pred_class)  #
        # pred_class_batch = torch.cat([pred_class_pos, pred_class_neg], dim=0).view(-1)
        label_batch = batch_labels  # [torch.sum(batch_labels, dim=-1) > 0]  #
        # label_batch = torch.cat([label_pos, label_neg], dim=0).view(-1)

        # ce_loss = torch.nn.functional.cross_entropy(torch.sigmoid(pred_class.squeeze(1)), batch_labels.to(device).long(),
        #                                             reduction='none')  # important to add reduction='none' to keep per-batch-item loss
        # pt = torch.exp(-ce_loss)
        # focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
        # train_loss = focal_loss
        node_loss += torch.nn.functional.mse_loss(torch.log(pred_class_batch+1), torch.log(label_batch.to(device).float()+1)).data
        del (batch_event_data)
        y_true = label_batch.float().numpy()
        y_pred = torch.relu(pred_class_batch).data.float().cpu().numpy()#(pred_class_batch.sigmoid()).data.float().cpu().numpy().flatten()#np.argsort(pred_class.data.cpu().numpy(), axis=-1)[:, -1]

        y_trues.append(y_true)
        y_preds.append(y_pred)

    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    # print(np.sum(y_trues))
    # node_ap = 0#average_precision_score(y_trues.astype('int'), (y_preds>0.5).astype('int'))
    # node_ap = 0  # average_precision_score(y_trues.astype('int'), (y_preds>0.5).astype('int'))
    # auc = 0#roc_auc_score(y_trues, y_preds)
    # print('test loss epoch', epoch, loss.data / (idx + 1), node_loss.data / (idx + 1), 'ap', node_ap, 'auc', auc)

    # test_res = []
    # for ai, bi in zip(y_trues, np.argsort(-y_preds, axis=-1)):
    #
    #     test_res += [ai[bi]]
    # test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    # print('Last Test NDCG: %.4f' % np.average(test_ndcg))
    # test_mrr = mean_reciprocal_rank(test_res)
    # print('Last Test MRR:  %.4f' % np.average(test_mrr))

    RMSE, MAE, PCC = evaluate(y_preds, y_trues)

    print('RMSE', RMSE, 'MAE', MAE, 'PCC', PCC)

    print('test node loss epoch', epoch, loss / (idx + 1))#, 'ndcg', np.average(test_ndcg), 'mrr', np.average(test_mrr),'auc', auc)

