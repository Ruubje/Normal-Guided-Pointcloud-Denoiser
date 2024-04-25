from . import Config as config
from .Utils import (
    GeneralUtils,
    SlicedTorchData,
    TorchUtils
)

from deprecated import deprecated
from numpy import (
    concatenate as np_concatenate
)
from scipy.spatial import KDTree as scipy_spatial_KDTree
from tqdm import tqdm
from torch import (
    arange as torch_arange,
    div as torch_div,
    eye as torch_eye,
    from_numpy as torch_from_numpy,
    int64 as torch_int64,
    is_tensor as torch_is_tensor,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch_geometric.data import Data as tg_data_Data
from warnings import warn as warnings_warn

class Selector:
    
    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ("pos", "edge_index"))
        self.graph = graph
        self.kdtree = scipy_spatial_KDTree(graph.pos)

    '''
        Patch Selection
    '''
    
    @deprecated("Getting a K ring per pgraph vertex is slow. This should be done in batch per ring using getVectorizedKRing()")
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
        if not (indices is None):
            I = indices.size(0)
            nodes_mask = torch_zeros((I, N), dtype=bool, device=device)
            nodes_mask[torch_arange(I, device=device), indices] = True
        else:
            nodes_mask = torch_eye(N, dtype=bool, device=device)
        for _ in range(k):
            _temp_edges = nodes_mask[:, start].view(-1)
            _temp_num_edges = _temp_edges.nonzero().view(-1)
            _temp_indices = end[_temp_num_edges % E] + N * torch_div(_temp_num_edges, E, rounding_mode=ROUNDING_MODE)
            nodes_mask[torch_div(_temp_indices, N, rounding_mode=ROUNDING_MODE), _temp_indices % N] = True
        data = (nodes_mask.view(-1).nonzero().squeeze() % N).view(-1)
        slices = TorchUtils.maskToSlices(nodes_mask)
        return SlicedTorchData(data, slices)
        
    def getTwoRings(self, _edge_index: torch_Tensor, indices: torch_Tensor=None) -> SlicedTorchData:
        # Batched algorithm can take up too much space, so if it does, the it while looping over array
        TorchUtils.validateEdgeIndex(_edge_index)
        try:
            result = self.getVectorizedKRing(2, _edge_index, indices)
            return result
            # return [result[i].nonzero() for i in tqdm(range(result.size(0)), desc="Extracting Tworings per node.")]
        except Exception as e:
            warnings_warn(str(e.with_traceback()), category=Warning)
            raise ValueError("Oh nooooooo")
            # _g = self.graph
            # _edge_index = _g.edge_index
            # N = _g.num_nodes
            # return [self.getKRing(2, _edge_index, i, N) for i in range(N)]
    
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

        nearby_vertices = self.kdtree.query_ball_point(_graph.pos[indices], radii)
        data = torch_from_numpy(np_concatenate(nearby_vertices)).to(device=device, dtype=torch_int64)
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
        
        tworings = self.getTwoRings(_g.edge_index, indices)
        # Calculate ball radii
        # radii = [self.getRadius(tr) for tr in tqdm(tworings, desc="Collecting radii")]
        radii = self.getRadiiVectorized(tworings)
        # Select points in ball
        nearby_vertices = self.getPointsInRangeVectorized(radii, indices)
        # Get graph nodes from vertices in range
        return nearby_vertices