from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
    """
    Abstract class for the message aggregator module, which given a batch of node ids and
    corresponding messages, aggregates messages with the same node id.
    """

    def __init__(self, device):
        super(MessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages, node_type):
        """
        Given a list of node ids, and a list of messages of the same length, aggregate different
        messages for the same id using one of the possible strategies.
        :param node_ids: A list of node ids of length batch_size
        :param messages: A tensor of shape [batch_size, message_length]
        :param timestamps A tensor of shape [batch_size]
        :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
        """

    def group_by_id(self, node_ids, messages, timestamps):
        node_id_to_messages = defaultdict(list)

        for i, node_id in enumerate(node_ids):
            node_id_to_messages[node_id].append((messages[i], timestamps[i]))

        return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(LastMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, node_messages, node_type):
        """Only keep the last message for each node"""
        # unique_node_ids = np.unique(node_ids)
        messages = []
        timestamps = []

        to_update_node_ids = []

        node_ids = node_ids.cpu().numpy().tolist()

        for idx, node_id in enumerate(node_ids):

            # print(type(node_id.data.to('cpu').numpy()))
            if len(node_messages[node_id]) > 0:
                to_update_node_ids.append(idx)
                messages.append(node_messages[node_id][-1][0])
                timestamps.append(node_messages[node_id][-1][1])
        # print(len(to_update_node_ids))
        messages = torch.stack(messages) if len(to_update_node_ids) > 0 else []
        timestamps = torch.stack(timestamps) if len(to_update_node_ids) > 0 else []

        return np.array(to_update_node_ids), messages, timestamps


class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(MeanMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, node_messages, node_type):
        """Only keep the last message for each node"""

        messages = []
        timestamps = []

        to_update_node_ids = []

        for node_id in node_ids:
            if len(node_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                messages.append(torch.mean(torch.stack([m[0] for m in node_messages[node_type][node_id]]), dim=0))
                timestamps.append(node_messages[node_type][node_id][-1][1])

        messages = torch.stack(messages) if len(to_update_node_ids) > 0 else []
        timestamps = torch.stack(timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, messages, timestamps


# class RNNMessageAggregator(MessageAggregator):
#     def __init__(self, device, dim):
#         super(RNNMessageAggregator, self).__init__(device)
#         self.rnn = torch.nn.RNNCell(dim, dim)
#
#     def aggregate(self, node_ids, node_messages, node_type):
#         """Only keep the last message for each node"""
#
#         messages = []
#         timestamps = []
#
#         to_update_node_ids = []
#
#         for node_id in node_ids:
#             if len(node_messages[node_id]) > 0:
#                 to_update_node_ids.append(node_id)
#                 messages.append(torch.mean(torch.stack([m[0] for m in node_messages[node_type][node_id]]), dim=0))
#                 timestamps.append(node_messages[node_type][node_id][-1][1])
#
#         messages = torch.stack(messages) if len(to_update_node_ids) > 0 else []
#         timestamps = torch.stack(timestamps) if len(to_update_node_ids) > 0 else []
#
#         return to_update_node_ids, messages, timestamps


def get_message_aggregator(aggregator_type, device):
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    else:
        raise ValueError("Message aggregator {} not implemented".format(aggregator_type))