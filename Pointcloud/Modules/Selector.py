from . import Config as config
from .Utils import (
    GeneralUtils,
    SlicedTorchData,
    TorchUtils
)

from deprecated import deprecated
from math import ceil as math_ceil
from numpy import (
    concatenate as np_concatenate
)
from scipy.spatial import KDTree as scipy_spatial_KDTree
from tqdm import tqdm
from torch import (
    arange as torch_arange,
    cat as torch_cat,
    from_numpy as torch_from_numpy,
    is_tensor as torch_is_tensor,
    long as torch_long,
    stack as torch_stack,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
import torch.cuda as torch_cuda
from torch_geometric.data import Data as tg_data_Data

import time

class Selector:
    
    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ("pos", "edge_index"))
        self.graph = graph
        self.kdtree = scipy_spatial_KDTree(graph.pos.cpu())

    '''
        Patch Selection
    '''
    
    @deprecated("Getting a K ring per graph vertex is slow. This should be done in batch per ring using getVectorizedKRing()")
    def getKRing(self, k: int, _edge_index: torch_Tensor, i: int, num_nodes: int=None) -> torch_Tensor:
        TorchUtils.validateEdgeIndex(_edge_index)
        device = _edge_index.device

        num_nodes = num_nodes if num_nodes is not None and num_nodes > 0 else _edge_index.max()
        node_mask = torch_zeros(num_nodes, dtype=bool, device=device)
        node_mask[i] = True
        for i in range(k):
            edge_mask = node_mask[_edge_index[0]]
            node_mask[_edge_index[1, edge_mask]] = True
        return node_mask.nonzero()
    
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
        f = torch_cuda.memory_reserved(0) - torch_cuda.memory_allocated(0)
        S = I * E
        B = math_ceil(S / f)
        J = math_ceil(E / B)
        for ki in range(k):
            updates = None
            for b in tqdm(range(B), desc=f"Process edges for ring {ki+1}"):
                bb = J*b
                be = min(J*(b+1), E)
                Y = be - bb
                batched_edges_start_nodes = start[bb:be]
                _temp_IY_indices = nodes_mask[:, batched_edges_start_nodes].view(-1).nonzero().squeeze()
                _temp_end_node_indices = end[_temp_IY_indices.remainder(Y) + bb]
                _temp_ring_indices = _temp_IY_indices.div_(Y, rounding_mode=ROUNDING_MODE)
                _temp_IN_indices = _temp_end_node_indices + N * _temp_ring_indices
                patch_idx = _temp_IN_indices.div(N, rounding_mode=ROUNDING_MODE)
                ring_idx = _temp_IN_indices.remainder_(N)
                local_updates = torch_stack((patch_idx, ring_idx))
                updates = local_updates if updates is None else torch_cat((updates, local_updates), dim=1)
            nodes_mask[updates[0], updates[1]] = True
        return TorchUtils.maskToSlicedTorchData(nodes_mask)
    
    @deprecated("Calculates K-ring per node, but is replaced by a vectorized method.")
    def getTwoRingsIndividually(self):
        _g = self.graph
        _edge_index = _g.edge_index
        _device = _edge_index.device
        N = _g.num_nodes
        list_result = [self.getKRing(2, _edge_index, i, N) for i in range(N)]
        data = torch_cat(list_result).squeeze_(1)
        slices = torch_zeros(len(list_result) + 1, dtype=int, device=_device)
        slices[1:] = torch_tensor([x.size(0) for x in list_result], dtype=torch_long, device=_device).cumsum_(dim=0)
        return SlicedTorchData(data, slices)
    
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
    
    def getPointsInRangeVectorized(self, radii: torch_Tensor, indices: torch_Tensor = None) -> SlicedTorchData:
        TorchUtils.validateIndices(indices)

        _graph = self.graph
        N = _graph.num_nodes if indices is None else indices.size(0)
        device = radii.device

        assert radii.dim() == 1
        assert radii.size(0) == N, f"Actual: {radii.size(0)}\nExpected: {N}"
        assert radii.is_floating_point()

        nearby_vertices = self.kdtree.query_ball_point(_graph.pos[indices].cpu(), radii.cpu())
        data = torch_from_numpy(np_concatenate(nearby_vertices)).to(device=device, dtype=torch_long)
        slices = torch_zeros(radii.size(0) + 1, dtype=int, device=device)
        slices[1:] = torch_tensor([len(x) for x in nearby_vertices]).cumsum(dim=0)
        return SlicedTorchData(data, slices)
    
    # Collecting patches from the object.
    def toPatchIndices(self, indices: torch_Tensor=None) -> SlicedTorchData:
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
        nearby_vertices = self.getPointsInRangeVectorized(radii, indices)
        # Get graph nodes from vertices in range
        return nearby_vertices