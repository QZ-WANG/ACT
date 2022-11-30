from typing import Union, List, Optional, Callable

from torch import Tensor
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk

import torch
from torch_sparse import SparseTensor


class SageUnsupSampler(RawNeighborSampler):

    def __init__(self, edge_index: Union[Tensor, SparseTensor], sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True, transform: Callable = None, q=1, **kwargs):
        super().__init__(edge_index, sizes, node_idx, num_nodes, return_e_id, transform, **kwargs)
        self.q = q

    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):

        pos_batch = random_walk(row, col, batch.repeat_interleave(self.q), walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel() * self.q,),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)

