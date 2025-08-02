from . import Config as config
from .Utils import (
    GeneralUtils,
    TorchUtils
)

from dataclasses import dataclass
from numpy import (
    concatenate as np_concatenate
)
from scipy.spatial import KDTree as scipy_spatial_KDTree
from tqdm import tqdm
from torch import (
    arange as torch_arange,
    cat as torch_cat,
    float as torch_float,
    from_numpy as torch_from_numpy,
    full as torch_full,
    is_tensor as torch_is_tensor,
    isin as torch_isin,
    long as torch_long,
    ones as torch_ones,
    repeat_interleave as torch_repeat_interleave,
    stack as torch_stack,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    vstack as torch_vstack,
    zeros as torch_zeros
)
from torch.sparse import (
    FloatTensor as ts_FloatTensor
)
from torch_geometric.data import Data as tg_data_Data
from torch_geometric.utils import sort_edge_index as tg_sort_edge_index
from torch_scatter import (
    scatter_max as torch_scatter_max,
    scatter_mean as torch_scatter_mean,
    scatter_sum as torch_scatter_sum
)

@dataclass
class Selection:
    i: torch_Tensor
    j: torch_Tensor
    slices: torch_Tensor

    def __len__(self):
        return self.slices.size(0) - 1
    
    def __getitem__(self, key: int):
        assert key >= 0 and key <= len(self)
        _slices = self.slices
        return self.j[_slices[key]:_slices[key+1]]
    
    def __setattr__(self, name: str, value: torch_Tensor):
        if name == "i":
            self._assertI(value)
        if name == "j":
            self._assertJ(value)
        elif name == "slices":
            self._assertSlices(value)
        object.__setattr__(self, name, value)

    def _assertI(self, _i: torch_Tensor):
        assert _i.dim() == 1, f"Actual size of i: {_i.size()}"
        assert not _i.is_floating_point()
        if hasattr(self, "slices"):
            assert _i.size(0) == len(self)

    def _assertJ(self, _j: torch_Tensor):
        assert _j.dim() == 1, f"Actual size of data: {_j.size()}"
        assert not _j.is_floating_point()
        if hasattr(self, "slices"):
            assert _j.size(0) == self.slices[-1]

    def _assertSlices(self, _slices: torch_Tensor):
        assert _slices.dim() == 1
        assert not _slices.is_floating_point()
        assert _slices[0] == 0, f"Slider start: {_slices[0]}"
        if hasattr(self, "i"):
            assert self.i.size(0) == _slices.size(0) - 1
        if hasattr(self, "j"):
            assert self.j.size(0) == _slices[-1], f"Data size: {self.j.size(0)}\nSlices last entry: {_slices[-1]}"
    
    def filter(self, indices: torch_Tensor):
        new_i = self.i[indices]
        _slices = self.slices
        starts = _slices[indices]
        ends = _slices[indices+1]
        new_j = self.j[TorchUtils.rangeBoundariesToIndices(starts, ends)]
        new_slices = torch_cat([torch_tensor([0]), (ends - starts).cumsum_(dim=0)])
        return Selection(new_i, new_j, new_slices)

    @classmethod
    def fromEdgeIndex(cls, edge_index):
        _edge_index = tg_sort_edge_index(edge_index)
        start = _edge_index[0]
        end = _edge_index[1]
        unique, counts = start.unique(return_counts=True)
        i = unique
        j = end
        N = unique.size(0)
        slices = torch_zeros((N + 1,), dtype=int)
        slices[1:] = counts.cumsum(dim=0)
        return Selection(i, j, slices)

    def getEdgeIndex(self) -> torch_Tensor:
        _slices = self.slices
        assert _slices.dim() == 1, "slices must have 2 dimensions"
        assert not _slices.is_floating_point(), "slices must contain integers"
        assert _slices is not None, "slices cannot be None"

        starts = _slices[:-1]
        ends = _slices[1:]
        _i = self.i
        start = torch_repeat_interleave(_i, ends - starts)
        end = self.j[TorchUtils.rangeBoundariesToIndices(starts, ends)]
        return torch_vstack([start[None], end[None]])
    
    def getBatchIndex(self) -> torch_Tensor:
        _i = self.i
        _slices = self.slices
        N = _i.size(0)
        indices = _i if _i[-1] == N - 1 else torch_arange(N)
        return torch_repeat_interleave(indices, _slices[1:] - _slices[:-1])
    
    def scatter(self, source: torch_Tensor, reduce: str) -> torch_Tensor:
        batch_index = self.getBatchIndex()
        if reduce == "add":
            return torch_scatter_sum(src=source, index=batch_index, dim=0, dim_size=len(self))
        elif reduce == "max":
            return torch_scatter_max(src=source, index=batch_index, dim=0, dim_size=len(self))
        elif reduce == "mean":
            return torch_scatter_mean(src=source, index=batch_index, dim=0, dim_size=len(self))

class Selector:
    
    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ["pos"])
        self.graph = graph
        self.kdtree = scipy_spatial_KDTree(graph.pos.cpu())

    '''
        Patch Selection
    '''
    
    def getVectorizedKRing(self, k: int, _edge_index: torch_Tensor, indices: torch_Tensor=None, num_nodes: int=None) -> Selection:
        assert k > 0, f"k: {k}"
        TorchUtils.validateEdgeIndex(_edge_index)
        TorchUtils.validateIndices(indices)

        device = _edge_index.device
        ROUNDING_MODE = "floor"
        start, end = _edge_index
        N = num_nodes if num_nodes is not None else _edge_index.max() + 1
        E = _edge_index.size(1)
        I = indices.size(0) if indices is not None else N
        nodes_mask = torch_zeros((I, N), dtype=bool, device=device)
        nodes_mask[torch_arange(I, device=device), indices] = True
        J = config.BATCH_SIZE
        B = E.__floordiv__(J) + 1
        for ki in range(k):
            updates = None
            for b in tqdm(range(B), desc=f"Process edges for ring {ki+1}"):
                bb = J*b
                be = min(J*(b+1), E)
                Y = be - bb
                batched_edges_start_nodes = start[bb:be]
                _temp_IY_indices = nodes_mask[:, batched_edges_start_nodes].view(-1).nonzero().squeeze(1)
                _temp_end_node_indices = end[_temp_IY_indices.remainder(Y) + bb]
                _temp_ring_indices = _temp_IY_indices.div_(Y, rounding_mode=ROUNDING_MODE)
                _temp_IN_indices = _temp_end_node_indices + N * _temp_ring_indices
                patch_idx = _temp_IN_indices.div(N, rounding_mode=ROUNDING_MODE)
                ring_idx = _temp_IN_indices.remainder_(N)
                local_updates = torch_stack((patch_idx, ring_idx))
                updates = local_updates if updates is None else torch_cat((updates, local_updates), dim=1)
            nodes_mask[updates[0], updates[1]] = True
        slicedTorchData = TorchUtils.maskToSlicedTorchData(nodes_mask)
        return Selection(indices, slicedTorchData.data, slicedTorchData.slices)
    
    def getSparseVectorizedKRing(self, k: int, _edge_index: torch_Tensor, row_indices: torch_Tensor=None, N: int=None) -> Selection:
        N = N if N is not None else _edge_index.max() + 1
        E = _edge_index.size(1)
        adjacency_matrix = ts_FloatTensor(_edge_index, torch_ones((E,)))
        result = ts_FloatTensor(
            torch_arange(N).repeat(2, 1),
            torch_ones(N),
        ).to(adjacency_matrix.device)
        base = adjacency_matrix.clone()

        while k > 0:
            if k % 2 == 1:
                result = result.mm(base)
            base = base.mm(base)
            k //= 2
        
        new_edge_index = result.coalesce().indices()
        if row_indices is None:
            return Selection.fromEdgeIndex(new_edge_index)
        else:
            mask = torch_isin(new_edge_index[0], row_indices)
            filtered_edge_index = new_edge_index[:, mask]
            return Selection.fromEdgeIndex(filtered_edge_index)
    
    def __getRadiiVectorized(self, tworings: Selection, k: int=config.K_PATCH_RADIUS) -> torch_Tensor:
        _g = self.graph
        GeneralUtils.validateAttributes(_g, ["mass"])
        _mass = _g.mass

        meantworingmasses = tworings.scatter(_mass[tworings.j], "mean")
        radii = k*meantworingmasses**0.5
        return radii
    
    def getPointsInRangeSelectionVectorized(self, radii: torch_Tensor, indices: torch_Tensor = None) -> Selection:
        TorchUtils.validateIndices(indices)

        _graph = self.graph
        N = _graph.num_nodes if indices is None else indices.size(0)
        device = radii.device

        assert radii.dim() == 1
        assert radii.size(0) == N, f"Actual: {radii.size(0)}\nExpected: {N}"
        assert radii.is_floating_point()

        _pos = _graph.pos if indices is None else _graph.pos[indices]
        nearby_vertices = self.kdtree.query_ball_point(_pos.cpu(), radii.cpu())
        j = torch_from_numpy(np_concatenate(nearby_vertices)).to(device=device, dtype=torch_long)
        slices = torch_zeros(radii.size(0) + 1, dtype=int, device=device)
        slices[1:] = torch_tensor([len(x) for x in nearby_vertices]).cumsum(dim=0)
        return Selection(indices if indices is not None else torch_arange(N), j, slices)
    
    def getPointsInRangeSelection(self, radius: float, indices: torch_Tensor = None) -> Selection:
        return self.getPointsInRangeSelectionVectorized(torch_full((self.graph.num_nodes,), radius, dtype=torch_float), indices)

    def getKNNSelection(self, k: int, indices: torch_Tensor=None) -> Selection:
        _pos = self.graph.pos
        _device = _pos.device
        if indices is None:
            indices = torch_arange(_pos.size(0), dtype=torch_long, device=_device)
        if not torch_is_tensor(indices) or indices.is_floating_point():
            raise ValueError(f"indices should contain integer values and not floating point values.")
        
        knn = torch_tensor(self.kdtree.query(_pos[indices].cpu(), k=k)[1], dtype=torch_long, device=_device)
        data = knn.flatten()
        slices = torch_arange(knn.size(0) + 1, device=knn.device, dtype=torch_long) * k
        return Selection(indices, data, slices)

    # Collecting patches from the object.
    def getMDSelection(self, indices: torch_Tensor=None) -> Selection:
        _g = self.graph
        if indices is None:
            indices = torch_arange(_g.pos.size(0))
        if indices is not None and (not torch_is_tensor(indices) or indices.is_floating_point()):
            raise ValueError(f"indices should contain integer values and not floating point values.")
        
        tworings = self.getSparseVectorizedKRing(2, _g.edge_index, indices)
        # Calculate ball radii
        radii = self.__getRadiiVectorized(tworings)
        # Select points in ball
        nearby_vertices = self.getPointsInRangeSelectionVectorized(radii, indices)
        # Get graph nodes from vertices in range
        return nearby_vertices