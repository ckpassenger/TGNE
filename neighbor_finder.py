from collections import defaultdict
import numpy as np

class NeighborFinder:
  def __init__(self, neighbor_list, uniform=False, seed=None):
    self.neighbor_list = neighbor_list

    self.neighbors = {}
    self.neighbor_edge_types = {}
    self.neighbor_edge_idxs = {}
    self.neighbor_edge_timestamps = {}
    self.neighbor_edge_directions = {}

    for node, neighbors in neighbor_list.items():
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[3])
      self.neighbors[node] = np.array([x[0] for x in sorted_neighhbors])
      self.neighbor_edge_types[node] = np.array([x[1] for x in sorted_neighhbors])
      self.neighbor_edge_idxs[node] = np.array([x[2] for x in sorted_neighhbors])
      self.neighbor_edge_timestamps[node] = np.array([x[3] for x in sorted_neighhbors])
      self.neighbor_edge_directions[node] = np.array([x[4] for x in sorted_neighhbors])

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_future_links(self, src_idx, cut_time, edge_type_idx = 3):
      if src_idx not in self.neighbors.keys():
          return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

      edge_type = (self.neighbor_edge_types[src_idx] == edge_type_idx)
      direction = (self.neighbor_edge_directions[src_idx][edge_type] == 1)

      i = np.searchsorted(self.neighbor_edge_timestamps[src_idx][edge_type][direction], cut_time)

      return self.neighbors[src_idx][edge_type][direction][:i], self.neighbor_edge_types[src_idx][edge_type][direction][:i], self.neighbor_edge_idxs[src_idx][edge_type][direction][:i],\
             self.neighbor_edge_timestamps[src_idx][edge_type][direction][:i], self.neighbor_edge_directions[src_idx][edge_type][direction][:i]

  def find_before(self, src_idx, cut_time, without_types = -1):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    edge_type = self.neighbor_edge_types[src_idx] != without_types

    if src_idx not in self.neighbors.keys():
        return np.array([]), np.array([]), np.array([]), np.array([])

    i = np.searchsorted(self.neighbor_edge_timestamps[src_idx][edge_type], cut_time)

    return self.neighbors[src_idx][edge_type][:i], self.neighbor_edge_types[src_idx][edge_type][:i], self.neighbor_edge_idxs[src_idx][edge_type][:i],\
           self.neighbor_edge_timestamps[src_idx][edge_type][:i], self.neighbor_edge_directions[src_idx][edge_type][:i]

  def get_temporal_neighbor(self, source_nodes, timestamp, n_neighbors=5, without_types = []):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1

    # NB! All interactions described in these matrices are sorted in each row by time
    sources = []  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    destinations = []  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.array([]).astype('int64')# each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.array([]).astype('int64')# each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_types = np.array([]).astype('int64')
    edge_directions = np.array([]).astype('int64')

    for i, source_node in enumerate(source_nodes):
        neighbor_idxs, neighbor_types, neighbor_edge_idxs, neighbor_edge_times, neighbor_edge_directions = self.find_before(source_node, timestamp, without_types = without_types)

        if len(neighbor_idxs) > 0:
          if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
            if len(neighbor_idxs)>tmp_n_neighbors:
              sampled_idx = np.random.randint(0, len(neighbor_idxs), tmp_n_neighbors)

              sources.append(np.array([source_node for _ in range(len(neighbor_idxs[sampled_idx]))]))
              destinations.append(neighbor_idxs[sampled_idx])
              # edges[-1] = np.concatenate([np.array([source_node for _ in range(len(edges[-1]))]).reshape(1,-1), edges[-1].reshape(1,-1)], axis=0)
              edge_times = np.append(edge_times, neighbor_edge_times[sampled_idx])#.astype(np.datetime64))
              edge_idxs = np.append(edge_idxs, neighbor_edge_idxs[sampled_idx])
              edge_types = np.append(edge_types, neighbor_types[sampled_idx])
              edge_directions = np.append(edge_directions, neighbor_edge_directions[sampled_idx])
            else:
              sources.append(np.array([source_node for _ in range(len(neighbor_idxs))]))
              destinations.append(neighbor_idxs)

              edge_times = np.append(edge_times, neighbor_edge_times)  # .astype(np.datetime64))
              edge_idxs = np.append(edge_idxs, neighbor_edge_idxs)
              edge_types = np.append(edge_types, neighbor_types)
              edge_directions = np.append(edge_directions, neighbor_edge_directions)

          else:
            # Take most recent interactions
            sources.append(np.array([source_node for _ in range(len(neighbor_idxs[-tmp_n_neighbors:]))]))
            destinations.append(neighbor_idxs[-tmp_n_neighbors:])
            # edges[-1] = np.concatenate([np.array([source_node for _ in range(len(edges[-1]))]).reshape(1,-1), edges[-1].reshape(1,-1)], axis=0)
            edge_times = np.append(edge_times, neighbor_edge_times[-tmp_n_neighbors:])#.astype(np.datetime64))
            edge_idxs = np.append(edge_idxs, neighbor_edge_idxs[-tmp_n_neighbors:])
            edge_types = np.append(edge_types, neighbor_types[-tmp_n_neighbors:])
            edge_directions = np.append(edge_directions, neighbor_edge_directions[-tmp_n_neighbors:])

    if len(destinations) > 0:
      sources = np.concatenate(sources, axis=-1).astype('str').reshape(-1)
      destinations = np.concatenate(destinations, axis=-1).astype('str').reshape(-1)
    else:
      sources = np.array([]).astype('str')
      destinations = np.array([]).astype('str')

    return sources, destinations, edge_types, edge_idxs, edge_times, edge_directions

def get_hetero_neighbor_finder(data, uniform, max_node_idx=None):
  # max_node_idx = max(len(data.sources), len(data.destinations)) if max_node_idx is None else max_node_idx
  neighbor_list = defaultdict(list)
  neighbor_list['paper_0'] = []#[('author_0', 0, 0, -1, 0), ('venue_0', 1, 0, -1, 0), ('keyword_0', 2, 0, -1, 0)]
  neighbor_list['author_0'] = []#[('paper_0', 0, 0, -1, 1)]
  neighbor_list['keyword_0'] = []#[('paper_0', 2, 0, -1, 1)]
  neighbor_list['venue_0'] = []#[('paper_0', 1, 0, -1, 1)]
  print("Creating adj list...")
  for source, destination, edge_type, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                                 data.edge_types,
                                                                 data.edge_idxs,
                                                                 data.timestamps):
    neighbor_list[source].append((destination, edge_type, edge_idx, timestamp, 0))
    neighbor_list[destination].append((source, edge_type, edge_idx, timestamp, 1))
  print("Creating neighbor finder...")

  return NeighborFinder(neighbor_list, uniform=uniform)

def get_neighbor_finder(data, uniform, max_node_idx=None):
  # max_node_idx = max(len(data.sources), len(data.destinations)) if max_node_idx is None else max_node_idx
  neighbor_list = defaultdict(list)
  # neighbor_list['paper_0'] = [('author_0', 0, 0, -1, 0), ('venue_0', 1, 0, -1, 0), ('keyword_0', 2, 0, -1, 0)]
  # neighbor_list['author_0'] = [('paper_0', 0, 0, -1, 1)]
  # neighbor_list['keyword_0'] = [('paper_0', 2, 0, -1, 1)]
  # neighbor_list['venue_0'] = [('paper_0', 1, 0, -1, 1)]
  print("Creating adj list...")
  for source, destination, edge_type, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_types,
                                                      data.edge_idxs,
                                                      data.timestamps):
      neighbor_list[source].append((destination, edge_type, edge_idx, timestamp, 0))
      neighbor_list[destination].append((source, edge_type, edge_idx, timestamp, 1))
  print("Creating neighbor finder...")

  return NeighborFinder(neighbor_list, uniform=uniform)