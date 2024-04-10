from .Alignor import Alignor
from .Selector import Selector
from .Utils import SlicedTorchData

from torch import (
    cat as torch_cat,
    Tensor as torch_Tensor
)
from torch_geometric.data import (
    Batch as tg_data_Batch,
    Data as tg_data_Data
)
from torch_geometric.utils import (
    degree as tg_utils_degree,
    subgraph as tg_utils_subgraph
)

class Preprocessor:
    def __init__(self, graph):
        self.graph = graph
        self.selector = Selector(graph)
        self.alignor = Alignor(graph)

    def getClasses(self) -> torch_Tensor:
        _alignor = self.alignor
        _selector = self.selector
        selection = _selector.toPatchIndices(indices=None)
        _, _, (eigval, _) = _alignor.getNormalVotingTensorTransforms(selection)
        return _alignor.getGroups(eigval)
    
    def getGraphs(self, indices: torch_Tensor = None):
        N = len(indices) if indices is not None else self.graph.num_nodes
        selection = self.selector.toPatchIndices(indices=indices)
        _, sf, eigh = self.alignor.getNormalVotingTensorTransforms(selection)
        RInv = self.alignor.getRInv(eigh, selection, indices)
        graph_list = [self.getGraph(x, selection, RInv, sf) for x in range(N)]
        return tg_data_Batch.from_data_list(graph_list)
    
    def getGraph(self, index: int, selection: SlicedTorchData, R_inv: torch_Tensor, scale_factors: torch_Tensor) -> tg_data_Data:
        # (Pi,)
        p = selection[index]
        _g = self.graph
        _edge_index = _g.edge_index
        # (Pi, 3)
        c = _g.pos[p]
        # (Pi, 3)
        nj = _g.n[p]
        # (3, 3)
        R_inv_i = R_inv[index]
        # (Pi, 3)
        nj_R_inv = nj @ R_inv_i
        # (3,)
        gt = _g.n[index]
        # (1, 3)
        gt_R_inv = gt[None] @ R_inv_i
        # (Pi, 3)
        n = nj_R_inv
        # (Pi, 1)
        a = (_g.mass[p] * scale_factors[index])[:, None]
        # (Pi, 1)
        d = tg_utils_degree(_edge_index[0])[p, None]
        # (Pi, 8)
        x = torch_cat((c, n, a, d), dim=1)
        # (2, E)
        edge_index = tg_utils_subgraph(p, _edge_index, relabel_nodes=True)[0]
        # (1, 3)
        y = gt_R_inv
        return tg_data_Data(x=x, edge_index=edge_index, y=y)