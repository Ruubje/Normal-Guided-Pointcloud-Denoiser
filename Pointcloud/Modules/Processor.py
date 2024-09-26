from .Alignor import Alignor
from .Denoiser import Denoiser
from .GraphBuilder import GraphBuilder
from .Noise import Noise
from .Object import Pointcloud
from .Selector import Selector
from .Utils import GeneralUtils, SlicedTorchData, TorchUtils

from torch import (
    cat as torch_cat,
    Tensor as torch_Tensor
)
from torch_geometric.data import (
    Data as tg_data_Data
)
from torch_geometric.utils import (
    degree as tg_utils_degree,
    subgraph as tg_utils_subgraph
)
from tqdm import tqdm

class Processor:
    def __init__(self, pointcloud: Pointcloud):
        self.pointcloud = pointcloud
        self.graphBuilder = GraphBuilder(pointcloud)
        _graph = self.graphBuilder.graph
        self.graph = _graph
        self.selector = Selector(_graph)
        self.alignor = Alignor(_graph)
        self.noise = Noise(_graph)
        self.denoiser = Denoiser(_graph)

    def getMDFeatures(self) -> torch_Tensor:
        GeneralUtils.validateAttributes(self, ["alignor", "selector"])
        _alignor = self.alignor
        selection = self.selector.getMDSelection(indices=None)
        _, _, (eigval, _) = _alignor.getMDTransformation(selection)
        return _alignor.getMDFeatures(eigval)
    
    def getMDPatches(self, indices: torch_Tensor = None):
        GeneralUtils.validateAttributes(self, ["alignor", "selector"])
        _alignor = self.alignor

        N = len(indices) if indices is not None else self.graph.num_nodes
        selection = self.selector.getMDSelection(indices=indices)
        _, sf, eigh = _alignor.getMDTransformation(selection)
        self.RInv = _alignor.getRInv(eigh, indices)
        self.graph_list = [self.getMDPatch(x if indices is None else indices[x], x, selection, self.RInv, sf) for x in tqdm(range(N), desc="Creating patches in list")]
        return self.graph_list, self.RInv
    
    def getMDPatch(self, global_index: int, index: int, selection: SlicedTorchData, R_inv: torch_Tensor, scale_factors: torch_Tensor) -> tg_data_Data:
        GeneralUtils.validateAttributes(self, ["graph"])
        # (Pi,)
        p = selection[index]
        scale_factor = scale_factors[index]
        _g = self.graph
        _edge_index = _g.edge_index
        # (Pi, 3)
        nj = _g.n[p]
        # (3, 3)
        R_inv_i = R_inv[index]
        # (Pi, 3)
        nj_R_inv = nj @ R_inv_i
        # (3,)
        gt = _g.gt_n[global_index] if hasattr(_g, "gt_n") else _g.n[global_index]
        # (Pi, 3)
        _pos = _g.pos[p]
        # (Pi, 3)
        c = (_pos - _pos.mean(dim=0)) * scale_factor @ R_inv_i
        # (Pi, 3)
        n = nj_R_inv
        # (Pi, 1)
        a = (_g.mass[p] * scale_factor)[:, None]
        # (Pi, 1)
        d = tg_utils_degree(_edge_index[0])[p, None]
        # (Pi, 8)
        x = torch_cat((c, n, a, d), dim=1)
        # (2, E)
        edge_index = tg_utils_subgraph(p, _edge_index, relabel_nodes=True)[0]
        # (1, 3)
        y = gt[None] @ R_inv_i
        return tg_data_Data(x=x, edge_index=edge_index, y=y)
    
    def getVUFeatures(self, k: int = 6, r_scale: float = 4):
        _graph = self.graph
        _graph.edge_index = self.graphBuilder.getKNNEdgeIndex(k)
        _pos = _graph.pos
        _edge_index = _graph.edge_index
        _selector = self.selector
        _alignor = self.alignor
        mean_graph_edge_length = TorchUtils.averageEdgeLength(_pos, _edge_index)
        r = r_scale * mean_graph_edge_length
        selection = _selector.getPointsInRangeSelection(r)
        filtered_normals = _alignor.getVUFilteredNormals(selection)
        eigval, _ = _alignor.getVUDecomposition(
            selection, 
            filtered_normals
        )
        features = _alignor.getVUFeatures(eigval, mean_graph_edge_length, k)
        return features