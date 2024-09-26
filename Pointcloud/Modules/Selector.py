from . import Config as config
from .Utils import (
    GeneralUtils,
    SlicedTorchData,
    TorchUtils
)

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
    long as torch_long,
    stack as torch_stack,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch_cluster import (
    knn_graph as tc_knn_graph
)
from torch_geometric.data import Data as tg_data_Data

class Selector:
    
    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ["pos"])
        self.graph = graph
        self.kdtree = scipy_spatial_KDTree(graph.pos.cpu())

    '''
        Patch Selection
    '''
    
    def getVectorizedKRing(self, k: int, _edge_index: torch_Tensor, indices: torch_Tensor=None, num_nodes: int=None) -> torch_Tensor:
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
        return TorchUtils.maskToSlicedTorchData(nodes_mask)
    
    def getRadiiVectorized(self, tworings: SlicedTorchData, k: int=config.K_PATCH_RADIUS) -> torch_Tensor:
        _g = self.graph
        device = tworings.data.device

        GeneralUtils.validateAttributes(_g, ["mass"])
        _mass = _g.mass
        assert device == _mass.device

        tworings.data = _mass[tworings.data]
        meantworingmasses = tworings.scatterReduce("mean")
        radii = k*meantworingmasses**0.5
        return radii
    
    def getPointsInRangeSelectionVectorized(self, radii: torch_Tensor, indices: torch_Tensor = None) -> SlicedTorchData:
        TorchUtils.validateIndices(indices)

        _graph = self.graph
        N = _graph.num_nodes if indices is None else indices.size(0)
        device = radii.device

        assert radii.dim() == 1
        assert radii.size(0) == N, f"Actual: {radii.size(0)}\nExpected: {N}"
        assert radii.is_floating_point()

        _pos = _graph.pos if indices is None else _graph.pos[indices]
        nearby_vertices = self.kdtree.query_ball_point(_pos.cpu(), radii.cpu())
        data = torch_from_numpy(np_concatenate(nearby_vertices)).to(device=device, dtype=torch_long)
        slices = torch_zeros(radii.size(0) + 1, dtype=int, device=device)
        slices[1:] = torch_tensor([len(x) for x in nearby_vertices]).cumsum(dim=0)
        return SlicedTorchData(data, slices)
    
    def getPointsInRangeSelection(self, radius: float, indices: torch_Tensor = None) -> SlicedTorchData:
        return self.getPointsInRangeSelectionVectorized(torch_full((self.graph.num_nodes,), radius, dtype=torch_float), indices)
    
    def getKNN(self, k: int, indices: torch_Tensor=None) -> torch_Tensor:
        _pos = self.graph.pos
        _device = _pos.device
        if indices is None:
            indices = torch_arange(_pos.size(0), dtype=torch_long, device=_device)
        if not torch_is_tensor(indices) or indices.is_floating_point():
            raise ValueError(f"indices should contain integer values and not floating point values.")
        
        return torch_tensor(self.kdtree.query(_pos[indices].cpu(), k=k)[1], dtype=torch_long, device=_device)

    def getKNNSelection(self, k: int, indices: torch_Tensor=None) -> SlicedTorchData:
        tensor = self.getKNN(k, indices)
        data = tensor.flatten()
        slices = torch_arange(tensor.size(0) + 1, device=tensor.device, dtype=torch_long) * k
        return SlicedTorchData(data, slices)

    # Collecting patches from the object.
    def getMDSelection(self, indices: torch_Tensor=None) -> SlicedTorchData:
        _g = self.graph
        if indices is None:
            indices = torch_arange(_g.pos.size(0))
        if indices is not None and (not torch_is_tensor(indices) or indices.is_floating_point()):
            raise ValueError(f"indices should contain integer values and not floating point values.")
        
        tworings = self.getVectorizedKRing(2, _g.edge_index, indices)
        # Calculate ball radii
        # radii = [self.getRadius(tr) for tr in tqdm(tworings, desc="Collecting radii")]
        radii = self.getRadiiVectorized(tworings)
        # Select points in ball
        nearby_vertices = self.getPointsInRangeSelectionVectorized(radii, indices)
        # Get graph nodes from vertices in range
        return nearby_vertices

    def getVUSelection(self, indices: torch_Tensor=None) -> SlicedTorchData:
        _g = self.graph
        if indices is None:
            indices = torch_arange(_g.pos.size(0))
        if indices is not None and (not torch_is_tensor(indices) or indices.is_floating_point()):
            raise ValueError(f"indices should contain integer values and not floating point values.")
        
        tworings = self.getVectorizedKRing(2, _g.edge_index, indices)
        # Calculate ball radii
        # radii = [self.getRadius(tr) for tr in tqdm(tworings, desc="Collecting radii")]
        radii = self.getRadiiVectorized(tworings)
        # Select points in ball
        nearby_vertices = self.getPointsInRangeSelectionVectorized(radii, indices)
        # Get graph nodes from vertices in range
        return nearby_vertices