import torch
from torch import nn
from torch_geometric.nn.dense import Linear
from torch_geometric.nn import Linear, HGTConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool

class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None

# class HeteroMessageFunction(MessageFunction):
#   def __init__(self, node_features, edge_features, memory, time_encoder, n_layers,
#                n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
#                n_heads=2, dropout=0.1, use_memory=True, metadata=None):
#     super(HeteroMessageFunction, self).__init__()
#
#     self.use_memory = use_memory
#     self.device = device
#
#     self.convs = torch.nn.ModuleList()
#     for _ in range(n_layers):
#       conv = TimeHANConv(embedding_dimension, embedding_dimension, metadata,
#                          n_heads)
#       self.convs.append(conv)
#
#     self.lin = torch.nn.ModuleDict()
#     for node_type in metadata.node_types:
#       self.lin[node_type] = Linear(embedding_dimension, embedding_dimension)
#
#   def compute_message(self, x_dict, edge_index_dict, edge_time_dict, batch):
#
#     for conv in self.convs:
#       x_dict = conv(x_dict, edge_index_dict, edge_time_dict)
#     for node_type, x in x_dict.items():
#       x_dict[node_type] = self.lin[node_type](x)
#
#     out = x_dict
#
#     for node_type, x in x_dict.items():
#       out[node_type] = global_mean_pool(x, batch[node_type])
#
#     return out

class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension, metadata):
    super(MLPMessageFunction, self).__init__()

    self.mlp = nn.ModuleDict()

    for node_type in metadata[0]:
      self.mlp[node_type] = nn.Sequential(
        nn.Linear(raw_message_dimension, raw_message_dimension // 2),
        nn.ReLU(),
        nn.Linear(raw_message_dimension // 2, message_dimension),
      )

  def compute_message(self, raw_messages, node_type):
    messages = self.mlp[node_type](raw_messages)

    return messages


class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):

    return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension, metadata):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension, metadata)
  elif module_type == "identity":
    return IdentityMessageFunction()
