from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass

class HyperSequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(HyperSequenceMemoryUpdater, self).__init__()
    self.memory = memory
    # self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  # def update_aug_memory(self, unique_node_ids, aug_messages):
  #   if len(unique_node_ids) <= 0:
  #     return
  #   aug_memory = self.memory.get_aug_memory(unique_node_ids)
  #
  #   updated_aug_memory = self.memory_updater(aug_messages, aug_memory)
  #
  #   self.memory.set_aug_memory(unique_node_ids, updated_aug_memory)

  def update_neg_memory(self, node_ids, to_update_node_ids, unique_messages, node_type, timestamps):
    if len(to_update_node_ids) <= 0:
      return

    # assert (self.memory.get_last_update(to_update_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"

    memory = self.memory.get_neg_memory(node_ids[to_update_node_ids], node_type).data.clone()
    # neighbor_memory = self.memory.get_neighbor_memory(unique_node_ids)
    # aug_memory = self.memory.get_aug_memory(unique_node_ids)
    # self.memory.last_update[node_type][node_ids[to_update_node_ids]] = timestamps

    # updated_aug_memory = self.memory_updater(aug_messages, aug_memory
    updated_memory = self.layer_norm[node_type](self.memory_updater[node_type](unique_messages.detach(), memory))

    self.memory.set_neg_memory(node_ids[to_update_node_ids], updated_memory, node_type)

  def get_updated_neg_memory(self, node_ids, to_update_node_ids, unique_messages, node_type, timestamps):
    if len(to_update_node_ids) <= 0:
      return self.memory.get_neg_memory(node_ids, node_type), self.memory.last_update[node_type][
        node_ids].data.clone()
    # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"
    # neighbor_memory = self.memory.get_neighbor_memory(unique_node_ids).data.clone()
    updated_memory = self.memory.get_neg_memory(node_ids, node_type)
    # updated_last_update = self.memory.last_update[node_type][node_ids].data.clone()
    # updated_aug_memory = self.memory.aug_memory.data.clone()

    # updated_aug_memory[unique_node_ids] = self.memory_updater(unique_aug_messages, updated_aug_memory[unique_node_ids])
    updated_memory[to_update_node_ids] = self.layer_norm[node_type](self.memory_updater[node_type](unique_messages.detach(), updated_memory[to_update_node_ids]))

    updated_last_update = self.memory.last_update[node_type][node_ids].data.clone()
    updated_last_update[to_update_node_ids] = timestamps

    # self.memory.last_update[node_type][node_ids[to_update_node_ids]] = timestamps

    return updated_memory, updated_last_update

  def update_memory(self, node_ids, to_update_node_ids, unique_messages, node_type, timestamps):
    if len(to_update_node_ids) <= 0:
      return

    # assert (self.memory.get_last_update(to_update_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"

    memory = self.memory.get_memory(node_ids[to_update_node_ids], node_type).data.clone()
    # neighbor_memory = self.memory.get_neighbor_memory(unique_node_ids)
    # aug_memory = self.memory.get_aug_memory(unique_node_ids)
    self.memory.last_update[node_type][node_ids[to_update_node_ids]] = timestamps

    # updated_aug_memory = self.memory_updater(aug_messages, aug_memory
    updated_memory = self.layer_norm[node_type](self.memory_updater[node_type](unique_messages.detach(), memory))

    self.memory.set_memory(node_ids[to_update_node_ids], updated_memory, node_type)

  def get_updated_memory(self, node_ids, to_update_node_ids, unique_messages, node_type, timestamps):
    if len(to_update_node_ids) <= 0:
      return self.memory.get_memory(node_ids, node_type), self.memory.last_update[node_type][
        node_ids].data.clone()
    # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"
    # neighbor_memory = self.memory.get_neighbor_memory(unique_node_ids).data.clone()
    updated_memory = self.memory.get_memory(node_ids, node_type)
    # updated_last_update = self.memory.last_update[node_type][node_ids].data.clone()
    # updated_aug_memory = self.memory.aug_memory.data.clone()

    # updated_aug_memory[unique_node_ids] = self.memory_updater(unique_aug_messages, updated_aug_memory[unique_node_ids])
    updated_memory[to_update_node_ids] = self.layer_norm[node_type](self.memory_updater[node_type](unique_messages.detach(), updated_memory[to_update_node_ids]))

    updated_last_update = self.memory.last_update[node_type][node_ids].data.clone()
    updated_last_update[to_update_node_ids] = timestamps

    # self.memory.last_update[node_type][node_ids[to_update_node_ids]] = timestamps

    return updated_memory, updated_last_update

class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.message_dimension = message_dimension
    self.device = device

  def update_aug_memory(self, unique_node_ids, aug_messages):
    if len(unique_node_ids) <= 0:
      return
    aug_memory = self.memory.get_aug_memory(unique_node_ids)

    updated_aug_memory = self.memory_updater(aug_messages, aug_memory)

    self.memory.set_aug_memory(unique_node_ids, updated_aug_memory)

  def update_memory(self, node_ids, to_update_node_ids, unique_messages, node_type, timestamps):
    if len(to_update_node_ids) <= 0:
      return

    # assert (self.memory.get_last_update(to_update_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"

    memory = self.memory.get_memory(node_ids[to_update_node_ids], node_type)
    # neighbor_memory = self.memory.get_neighbor_memory(unique_node_ids)
    # aug_memory = self.memory.get_aug_memory(unique_node_ids)
    self.memory.last_update[node_type][node_ids[to_update_node_ids]] = timestamps

    # updated_aug_memory = self.memory_updater(aug_messages, aug_memory)
    updated_memory = self.layer_norm(self.memory_updater(unique_messages, memory))

    self.memory.set_memory(node_ids[to_update_node_ids], updated_memory, node_type)


  def get_updated_memory(self, node_ids, to_update_node_ids, unique_messages, node_type, timestamps):
    if len(to_update_node_ids) <= 0:
      return self.memory.memory[node_type][node_ids].data.clone(), self.memory.last_update[node_type][node_ids].data.clone()

    # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"
    # neighbor_memory = self.memory.get_neighbor_memory(unique_node_ids).data.clone()
    updated_memory = self.memory.memory[node_type][node_ids]
    # updated_last_update = self.memory.last_update[node_type][node_ids].data.clone()
    # updated_aug_memory = self.memory.aug_memory.data.clone()

    # updated_aug_memory[unique_node_ids] = self.memory_updater(unique_aug_messages, updated_aug_memory[unique_node_ids])
    updated_memory[to_update_node_ids] = self.layer_norm(self.memory_updater(unique_messages,updated_memory[to_update_node_ids]))

    updated_last_update = self.memory.last_update[node_type][node_ids].data.clone()
    updated_last_update[to_update_node_ids] = timestamps

    # self.memory.last_update[node_type][node_ids[to_update_node_ids]] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(HyperSequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, metadata = None):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
    self.memory_updater = torch.nn.ModuleDict()
    self.layer_norm = torch.nn.ModuleDict()
    if metadata is not None:
      for node_type in metadata[0]:
        self.memory_updater[node_type] = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
        self.layer_norm[node_type] = torch.nn.LayerNorm(memory_dimension)



class RNNMemoryUpdater(HyperSequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, metadata = None):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
    self.memory_updater = torch.nn.ModuleDict()
    self.layer_norm = torch.nn.ModuleDict()
    if metadata is not None:
      for node_type in metadata[0]:
        self.memory_updater[node_type] = nn.RNNCell(input_size=message_dimension,
                                                    hidden_size=memory_dimension)
        self.layer_norm[node_type] = torch.nn.LayerNorm(memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device, metadata = None):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device, metadata = metadata)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device, metadata = metadata)