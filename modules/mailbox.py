import torch
from torch import nn

from collections import defaultdict
from torch_geometric.nn.inits import glorot, reset, ones, zeros
from copy import deepcopy
import random
import math
import time

def detach_param_dict(param_dict):
    for type, param in param_dict.items():
        param_dict[type] = param.detach()
    return param_dict

class Mailbox(nn.Module):
    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                   device="cpu", combination_method='sum', metadata = None):
        super(Mailbox, self).__init__()
        # if isinstance(n_nodes, dict):
        #     in_channels = {node_type: n_nodes for node_type in metadata[0]}
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.metadata = metadata

        self.combination_method = combination_method

        # self.weights = torch.exp(0.2*torch.arange(1,10)).cuda()

        self.memory = torch.nn.ModuleDict()
        self.last_memory = torch.nn.ModuleDict()
        self.last_memory2 = torch.nn.ModuleDict()
        self.neg_memory = torch.nn.ModuleDict()
        self.feature = torch.nn.ParameterDict()
        self.feature_enc = torch.nn.ModuleDict()
        # self.embedding = torch.nn.ParameterDict()
        self.last_update = torch.nn.ParameterDict()
        self.messages = {}
        self.neg_messages = {}

        for node_type in self.metadata[0]:
            self.memory[node_type] = nn.Embedding(self.n_nodes[node_type], self.memory_dimension)
                # torch.zeros((self.n_nodes[node_type], self.memory_dimension)).to(self.device),
                #                                        requires_grad=False)
            self.last_memory[node_type] = nn.Embedding(self.n_nodes[node_type], self.memory_dimension)
            self.last_memory2[node_type] = nn.Embedding(self.n_nodes[node_type], self.memory_dimension)
            # nn.Parameter(
            #     torch.zeros((self.n_nodes[node_type], self.memory_dimension)).to(self.device),
            #     requires_grad=False)
            self.neg_memory[node_type] = nn.Embedding(self.n_nodes[node_type], self.memory_dimension)
            # nn.Parameter(
            #     torch.zeros((self.n_nodes[node_type], self.memory_dimension)).to(self.device),
            #     requires_grad=False)
            self.feature[node_type] = nn.Parameter(
                torch.randn((self.n_nodes[node_type], self.memory_dimension)).to(self.device),
                                                       requires_grad=False)
            # self.feature_enc[node_type] = nn.Linear(3*self.memory_dimension, self.memory_dimension)
            self.feature_enc[node_type] = nn.RNN(self.memory_dimension, self.memory_dimension)
            # self.embedding[node_type] = nn.Parameter(
            #     torch.zeros((self.n_nodes[node_type], self.memory_dimension)).to(self.device),
            #     requires_grad=False)
            self.last_update[node_type] = nn.Parameter(torch.zeros(self.n_nodes[node_type], 1).to(self.device),
                                                       requires_grad=False)
            self.messages[node_type] = defaultdict(list)
            self.neg_messages[node_type] = defaultdict(list)

        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        if self.metadata is not None:
            self.messages = {}
            self.neg_messages = {}

            zeros(self.memory)
            zeros(self.last_memory)
            zeros(self.last_memory2)
            zeros(self.neg_memory)
            zeros(self.last_update)
            for node_type in self.metadata[0]:
                self.messages[node_type] = defaultdict(list)
                self.neg_messages[node_type] = defaultdict(list)
                # self.feature_enc[node_type].reset_parameters()
                # self.negative_memory[node_type] = []

        else:
            self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                                       requires_grad=False)

            self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                            requires_grad=False)

            self.messages = defaultdict(list)
            self.aug_messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages, node_type=None):
        if node_type is not None:
            for node in nodes:
                self.messages[node_type][node].extend(node_id_to_messages[node])
        else:
            for node in nodes:
                self.messages[node].extend(node_id_to_messages[node])

    def store_neg_messages(self, nodes, node_id_to_messages, node_type=None):
        if node_type is not None:
            for node in nodes:
                self.neg_messages[node_type][node].extend(node_id_to_messages[node])
        else:
            for node in nodes:
                self.messages[node].extend(node_id_to_messages[node])
        # print(self.messages[node_type][])

    def get_memory(self, node_idxs, node_type=None):
        if node_type is not None:
            # return self.feature_enc[node_type](torch.cat([self.memory[node_type](node_idxs).detach().clone(), self.last_memory[node_type](node_idxs).detach().clone(), self.last_memory2[node_type](node_idxs).detach().clone()], dim=-1))

            x, _ = self.feature_enc[node_type](torch.cat([self.memory[node_type](node_idxs).detach().clone().unsqueeze(0), self.last_memory[node_type](node_idxs).detach().clone().unsqueeze(0), self.last_memory2[node_type](node_idxs).detach().clone().unsqueeze(0)], dim=0))
            return x[-1]
            # return self.feature_enc[node_type](torch.cat([self.memory[node_type](node_idxs).detach().clone(),
            #                                               self.last_memory[node_type](node_idxs).detach().clone()],
            #                                              dim=-1))
            # return self.feature_enc[node_type](self.memory[node_type](node_idxs).detach().clone())
            # rnn / transformer
            # return self.memory[node_type](node_idxs).detach()
        else:
            return self.memory[node_idxs]

    def get_neg_memory(self, node_idxs, node_type=None):
        if node_type is not None:
            return self.feature_enc[node_type](torch.cat([self.neg_memory[node_type](node_idxs).detach(), self.last_memory[node_type](node_idxs).detach(), self.last_memory2[node_type](node_idxs).detach()], dim=-1))
        else:
            return self.neg_memory[node_idxs]

    def set_neg_memory(self, node_idxs, values, node_type=None):
        if node_type is not None:
            # self.last_memory[node_type][node_idxs, :] = self.memory[node_type][node_idxs, :]
            # self.neg_memory[node_type][node_idxs, :] = values
            self.neg_memory[node_type].weight.data[node_idxs, :] = values
        else:
            self.memory[node_idxs, :] = values
    #
    # def get_embedding(self, node_idxs, node_type=None):
    #     if node_type is not None:
    #         return self.embedding[node_type][node_idxs]
    #     else:
    #         return self.embedding[node_idxs]

    # def set_embedding(self, node_idxs, values, node_type=None):
    #     if node_type is not None:
    #         self.embedding[node_type][node_idxs, :] = values
    #     else:
    #         self.embedding[node_idxs, :] = values

    def set_memory(self, node_idxs, values, node_type=None):
        if node_type is not None:
            self.last_memory2[node_type].weight.data[node_idxs, :] = self.last_memory[node_type].weight.data[node_idxs, :]
            self.last_memory[node_type].weight.data[node_idxs, :] = self.memory[node_type].weight.data[node_idxs, :]
            self.memory[node_type].weight.data[node_idxs, :] = values
        else:
            self.memory[node_idxs, :] = values

    def backup_memory(self):
        memory_clone = {}
        for node_type, type_memory in self.memory.items():
            memory_clone[node_type] = type_memory.weight.data.clone()

        last_memory_clone = {}
        for node_type, type_memory in self.last_memory.items():
            last_memory_clone[node_type] = type_memory.weight.data.clone()

        last_memory2_clone = {}
        for node_type, type_memory in self.last_memory2.items():
            last_memory2_clone[node_type] = type_memory.weight.data.clone()

        last_update_clone = {}
        for node_type, type_last_update in self.last_update.items():
            last_update_clone[node_type] = type_last_update.data.clone()

        messages_clone = {}
        for node_type, type_message in self.messages.items():
            messages_clone[node_type] = {}
            for k, v in type_message.items():
                messages_clone[node_type][k] = [(x[0].clone(), x[1].clone()) for x in v]

        return (memory_clone, last_memory_clone, last_memory2_clone, last_update_clone, messages_clone)

    def restore_memory(self, memory_backup):
        memory_clone, last_memory_clone, last_memory2_clone, last_update_clone, messages_clone = memory_backup
        self.messages = {}
        for node_type in self.metadata[0]:
            self.memory[node_type].weight.data = memory_clone[node_type].clone()
            self.last_memory[node_type].weight.data = last_memory_clone[node_type].clone()
            self.last_memory2[node_type].weight.data = last_memory2_clone[node_type].clone()
            self.last_update[node_type].data = last_update_clone[node_type].clone()
            self.messages[node_type] = defaultdict(list)
            for k, v in messages_clone[node_type].items():
                self.messages[node_type][k] = [(x[0].clone(), x[1].clone()) for x in v]

        # self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
        #
        # self.messages = defaultdict(list)
        # for k, v in memory_backup[2].items():
        #     self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        if self.metadata is not None:
            for node_type in self.metadata[0]:
                # self.memory[node_type].detach_()
                for k, v in self.messages[node_type].items():
                    new_node_messages = []
                    for message in v:
                        new_node_messages.append((message[0].detach(), message[1]))

                    self.messages[node_type][k] = new_node_messages
        else:
            self.memory.detach_()

            # Detach all stored messages
            for k, v in self.messages.items():
                new_node_messages = []
                for message in v:
                    new_node_messages.append((message[0].detach(), message[1]))

                self.messages[k] = new_node_messages

    def clear_messages(self, nodes, node_type=None):
        for node in nodes:
            self.messages[node_type][node] = []
            self.neg_messages[node_type][node] = []
    # def aggregate_messages(self, nodes, node_type=None, agg_type = 'last'):
    #     messages = []
    #     timestamps = []
    #     to_update_node_ids = []
    #     for idx, node in enumerate(nodes):
    #         if len(self.messages[node_type][node]) > 0:
    #             to_update_node_ids.append(idx)
    #             if agg_type == 'mean':
    #                 aggregated_message = torch.mean(
    #                     torch.cat([node_message[0].unsqueeze(0) for node_message in self.messages[node_type][node]], dim=0), dim=0)
    #             elif agg_type == 'last':
    #                 aggregated_message = self.messages[node_type][node][-1][0]
    #             timestamps.append(self.messages[node_type][node][-1][1])
    #             messages.append(aggregated_message)
    #     messages = torch.stack(messages) if len(to_update_node_ids) > 0 else []
    #     timestamps = torch.stack(timestamps) if len(to_update_node_ids) > 0 else []
    #
    #     to_update_node_ids = torch.LongTensor(to_update_node_ids).to(self.device)
    #     return to_update_node_ids, messages, timestamps


