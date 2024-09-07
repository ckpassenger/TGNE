from typing import Dict, List, Optional, Union
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset, ones
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax

def group(xs: List[Tensor], q: nn.Parameter,
          k_lin: nn.Module) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out

def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect

class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = nn.Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        # scores = scores

        vals, topk_indices = scores.view(-1).topk(self.k)

        topk_indices = topk_indices[vals > -float("Inf")]
        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = nn.Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = nn.Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = nn.Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class mat_GRU_cell(torch.nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.update = mat_GRU_gate(rows,
                                   cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(rows,
                                  cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(rows,
                                   cols,
                                   torch.nn.Tanh())

        self.choose_topk = TopK(feats=rows,
                                k=cols)

    def forward(self, prev_Q, prev_Z):
        z_topk = self.choose_topk(prev_Z)

        update = torch.clamp(self.update(z_topk, prev_Q), min=1e-4, max=1-1e-4)
        reset = torch.clamp(self.reset(z_topk, prev_Q), min=1e-4, max=1-1e-4)

        h_cap = reset * prev_Q
        h_cap =torch.clamp(self.htilda(z_topk, h_cap), min=-(1-1e-4), max=1-1e-4)
        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

class TimeHANConv(MessagePassing):
    r"""
    The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/pdf/1903.07293.pdf>`_ paper.

    .. note::

        For an example of using HANConv, see `examples/hetero/han_imdb.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/han_imdb.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        n_edge_features: int,
        metadata: Metadata,
        time_encoder,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.Tensor(1, out_channels))

        self.time_encoder = time_encoder
        #
        # self.global_memory = nn.ParameterDict()
        # for node_type, in_channels in self.in_channels.items():
        #     self.global_memory[node_type] = nn.Parameter(
        #         torch.FloatTensor(in_channels, 1))  # Linear(in_channels, out_channels, bias = False)

        self.proj_weight = nn.ModuleDict()

        self.out_lin = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj_weight[node_type] = Linear(in_channels, out_channels)
            self.out_lin[node_type] = Linear(2*in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        self.lin_time = nn.ParameterDict()
        self.lin_feature = nn.ParameterDict()
        self.edge_proj_weight = nn.ModuleDict()
        self.edge_time_weight = nn.ModuleDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_time[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_feature[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.edge_proj_weight[edge_type] = Linear(n_edge_features, out_channels)
            self.edge_time_weight[edge_type] = Linear(out_channels, out_channels)
            # self.time_encoder[edge_type] = nn.Embedding(4000, dim, padding_idx=0)
        self.param_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_src)
        glorot(self.lin_dst)
        glorot(self.lin_time)
        self.k_lin.reset_parameters()
        glorot(self.q)

    # def updated_weight(self, node_type, event_list):
    #     return self.param_updaters[node_type](self.global_memory[node_type], event_list)

    def forward(
        self, x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType,
                              Adj] ,
        x_time_dict,
        edge_feature_dict,
        edge_time_dict = None)-> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict: (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The ouput node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, time_dict, out_dict = {}, {}, {}

        # Iterate over node types:
        for node_type, x_node in x_dict.items():
            # x_node_dict[node_type] = self.proj_weight[node_type](x_node+self.global_memory[node_type].transpose(0,1)).view(
            #     -1, H, D)
            time_dict[node_type] = self.time_encoder(x_time_dict[node_type]).view(-1, self.out_channels)
            x_node_dict[node_type] = self.proj_weight[node_type](x_node).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():

            src_type, _, dst_type = edge_type
            edge_type_ = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type_]
            lin_dst = self.lin_dst[edge_type_]
            lin_time = self.lin_time[edge_type_]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]

            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)

            edge_feature = self.edge_proj_weight[edge_type_](edge_feature_dict[edge_type]).view(-1, H, D)
            alpha_edge_feature = (self.lin_feature[edge_type_]*edge_feature).sum(dim=-1)

            alpha = (alpha_src, alpha_dst)
            # propagate_type: (x_dst: Tensor, alpha: PairTensor)
            if edge_time_dict is not None:
                time = self.edge_time_weight[edge_type_](self.time_encoder(edge_time_dict[edge_type])).view(-1, H, D)
                alpha_time = (time * lin_time).sum(dim=-1)
                # print(edge_time_dict[edge_type][:5], time[:5,0,:5])

                out = self.propagate(edge_index, x_dst=x_dst, alpha=alpha,
                                     size=None, alpha_edge = alpha_edge_feature, edge_feature = edge_feature+time, alpha_edge_time = alpha_time)
            else:
                out = self.propagate(edge_index, x_dst=x_dst, alpha=alpha,
                                     size=None, alpha_edge = alpha_edge_feature, edge_feature = edge_feature, alpha_edge_time=None)

            # out = F.relu(out)
            out_dict[dst_type].append(out)

        # iterate over node types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.q, self.k_lin)

            if out is None:
                out_dict[node_type] = None
                continue
            out_dict[node_type] = torch.tanh(out)#torch.tanh(self.out_lin[node_type](torch.cat([out, x_dict[node_type]+time_dict[node_type]], dim=-1)))
            # out_dict[node_type] = torch.tanh(self.out_lin[node_type](torch.cat([out, x_dict[node_type]], dim=-1)))

        return out_dict


    def message(self, x_dst_i: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int], alpha_edge, edge_feature, alpha_edge_time) -> Tensor:

        alpha = alpha_j + alpha_i #+ alpha_edge
        # if alpha_edge_time is not None:
        #     alpha += alpha_edge_time
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = (x_dst_i+edge_feature) * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')

class TimeHGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        group (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        time_encoder,
        heads: int = 1,
        group: str = "sum",
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group

        self.time_encoder = time_encoder

        self.k_lin1 = nn.Linear(out_channels, out_channels)

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        self.param_updaters = torch.nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = nn.Parameter(torch.Tensor(1))
            self.param_updaters[node_type] = mat_GRU_cell(in_channels, 1)

        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        self.time_rel = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = nn.Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = nn.Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = nn.Parameter(torch.Tensor(heads))
            self.time_rel[edge_type] = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict,
        edge_time_dict = None # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict: (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The ouput node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type_ = '__'.join(edge_type)

            a_rel = self.a_rel[edge_type_]
            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)

            m_rel = self.m_rel[edge_type_]
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)

            if edge_time_dict is not None:
                # lin_time = self.lin_time[edge_type_]
                time = self.time_encoder(edge_time_dict[edge_type])
                alpha_time = (time).sum(dim=-1)
                out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                     rel=self.p_rel[edge_type_], size=None, edge_time=alpha_time)
            else:
                out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                     rel=self.p_rel[edge_type_], size=None,
                                     edge_time=None)

            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.group, self.k_lin1)

            if out is None:
                out_dict[node_type] = None
                continue

            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict


    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int], edge_time) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        if edge_time is not None:
            alpha += edge_time
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')