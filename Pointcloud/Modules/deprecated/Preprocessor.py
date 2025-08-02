from ..Object import (
    Pointcloud,
    FileMesh,
    FilePointcloud
)

from dataclasses import dataclass
from itertools import zip_longest as itertools_zip_longest
from numba import jit
from numpy import (
    arange as np_arange,
    argsort as np_argsort,
    array as np_array,
    average as np_average,
    cross as np_cross,
    einsum as np_einsum,
    exp as np_exp,
    logical_and as np_logical_and,
    max as np_max,
    ndarray as np_ndarray,
    sort as np_sort,
    sum as np_sum,
    transpose as np_transpose,
    zeros as np_zeros
)
from numpy.linalg import (
    det as np_linalg_det,
    eigh as np_linalg_eigh,
    norm as np_linalg_norm
)
from robust_laplacian import point_cloud_laplacian as robust_pointcloud_laplacian
from sklearn.preprocessing import normalize as sklearn_preprocessing_normalize
from tqdm import tqdm
from torch import (
    arange as torch_arange,
    cat as torch_cat,
    div as torch_div,
    eye as torch_eye,
    float32 as torch_float32,
    from_numpy as torch_from_numpy,
    int64 as torch_int64,
    is_tensor as torch_is_tensor,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch_geometric.data import Data as tg_data_Data
from torch_geometric.utils import (
    degree as tg_utils_degree,
    subgraph as tg_utils_subgraph
)
from typing import Tuple

# TODO Change all code such that it only uses Torch.
# We can transfer an object to a graph and therefore it is no longer needed to use Numpy.
# Why convert back to Numpy if the graph can have all the information?

@jit(nopython=True)
def getJITTwoRing(i, _edge_index):
    nodes = [i]
    for _ in range(2):
        start = _edge_index[0]
        end = _edge_index[1]
        new_nodes = []
        for k in range(start.shape[0]):
            if start[k] in nodes:
                new_nodes.append(end[k])
        nodes = nodes.append(new_nodes)
    return torch_tensor(list(set(nodes)))

@dataclass
class Selection:
    indices: np_ndarray
    neighbourhood: np_ndarray
    mask: np_ndarray

    def __post_init__(self):
        indices_shape = self.indices.shape
        neighbourhood_shape = self.neighbourhood.shape
        mask_shape = self.mask.shape
        if len(indices_shape) != 1 or len(neighbourhood_shape) != 2 or len(mask_shape) != 2:
            raise ValueError(f"One of the array has a wrong shape.\nindices: {indices_shape}\nneighbourhood: {neighbourhood_shape}\nmask: {mask_shape}")
        if not (indices_shape[0] == neighbourhood_shape[0] and neighbourhood_shape == mask_shape):
            raise ValueError(f"Shape should be the same, but aren't correct.\nindices: {indices_shape}\nneighbourhood: {neighbourhood_shape}\nmask: {mask_shape}")

class Preprocessor:

    DEFAULT_PATCH_RESOLUTION_K = 4
    DEFAULT_SELECTION_MODE = 0
    
    def __init__(self, object):
        if not isinstance(object, FilePointcloud):
            raise ValueError(f"A Patch Generator expects a Pytorch Geometric Data object from which it can create patches.")

        self.object = object

    '''
        Patch Selection
    '''

    def getJITTwoRings(self):
        N = self.object.g.pos.size(0)
        _edge_index = self.object.g.edge_index.numpy()
        return [self.getJITTwoRing(i, _edge_index) for i in range(N)]
    
    def getTwoRing(self, i, _edge_index, N):
        node_mask = torch_zeros(N, dtype=bool)
        node_mask[i] = True
        edge_mask = node_mask[_edge_index[0]]
        node_mask[_edge_index[1, edge_mask]] = True
        edge_mask = node_mask[_edge_index[0]]
        node_mask[_edge_index[1, edge_mask]] = True
        return node_mask.nonzero()
    
    def getVectorizedKRing(self, k: int, indices: torch_Tensor=None) -> torch_Tensor:
        ROUNDING_MODE = "floor"
        _g = self.object.g
        start, end = _g.edge_index
        N = _g.pos.size(0)
        E = start.size(0)
        if not (indices is None):
            I = indices.size(0)
            nodes_mask = torch_zeros((I, N), dtype=bool)
            nodes_mask[torch_arange(I), indices] = True
        else:
            nodes_mask = torch_eye(N, dtype=bool)
        for _ in range(k):
            _temp_edges = nodes_mask[:, start].view(-1)
            _temp_num_edges = _temp_edges.nonzero().view(-1)
            _temp_indices = end[_temp_num_edges % E] + N * torch_div(_temp_num_edges, E, rounding_mode=ROUNDING_MODE)
            nodes_mask[torch_div(_temp_indices, N, rounding_mode=ROUNDING_MODE), _temp_indices % N] = True
        return nodes_mask
        
    def getTwoRings(self, indices: torch_Tensor=None) -> list[torch_Tensor]:
        # Batched algorithm can take up too much space, so if it does, the it while looping over array
        try:
            # return self.getJITTwoRings()
            result = self.getVectorizedKRing(2, indices)
            return [result[i].nonzero() for i in tqdm(range(result.size(0)), desc="Collecting Tworings")]
        except:
            _g = self.object.g
            _edge_index = _g.edge_index
            N = _g.pos.size(0)
            return [self.getTwoRing(i, _edge_index, N) for i in range(N)]

    def getRadius(self, tworing, k=DEFAULT_PATCH_RESOLUTION_K):
        _object = self.object
        a = np_average(_object.getAreas()[tworing])
        radius = k*a**0.5
        return radius

    def getNodes(self, neighbours):
        _object = self.object
        if isinstance(_object, FileMesh):
            _vta = _object.vta
            _vta1 = _vta[1]
            # No sort or unique needed. Ranges are unique and sorted.
            ts_i = HelperFunctions.rangeBoundariesToIndices(_vta1[neighbours], _vta1[neighbours+1])
            # Values in vta0 are not unique, because multiple vertices can have the same triangle.
            ts = np_array(list(set(_vta[0][ts_i])))
            return ts
        else:
            return neighbours
        
    def toMasked2DArray(self, indices: np_ndarray[list[int]]) -> Tuple[np_ndarray, np_ndarray]:
        # https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values
        indices_2d = np_array(list(itertools_zip_longest(*indices, fillvalue=-1))).T
        mask = indices_2d == -1 # Detect mask
        indices_2d[mask] = 0 # Set mask to center node to not throw errors while indexing
        return indices_2d, mask
    
    # Collecting patches from the object.
    # mode is 0 or 1.
    #   - 0 represents the method from the original paper and;
    #   - 1 represents a new sped up version.
    def toPatchIndices(self, indices: torch_Tensor=None, mode=DEFAULT_SELECTION_MODE, k=DEFAULT_PATCH_RESOLUTION_K) -> Selection:
        _object = self.object
        _g = _object.g
        if indices is None:
            indices = torch_arange(_g.pos.size(0))            
        if indices is not None and (not torch_is_tensor(indices) or indices.is_floating_point()):
            raise ValueError(f"indices should contain integer values and not floating point values.")
        _pos = _g.pos[indices]
        if mode == 0:
            # tworings = [self.getTwoRing(i) for i in range(_pos.size(0))]
            tworings = self.getTwoRings(indices)
            # Calculate ball radii
            radii = [self.getRadius(tr, k) for tr in tqdm(tworings, desc="Collecting radii")]
            # Select points in ball
            nearby_vertices = np_array(_object.kdtree.query_ball_point(_pos.numpy(), radii))
            # Get graph nodes from vertices in range
            nodes = [self.getNodes(np_array(neighbours)) for neighbours in tqdm(nearby_vertices, desc="Collecting nodes from vertices.")]
        elif mode == 1:
            print("Collecting nearby vertices")
            _, nearby_vertices = _object.kdtree.query(_pos, k=64, workers=-1)
            print("Collecting nodes")
            nodes = [self.getNodes(np_array(neighbours)) for neighbours in nearby_vertices]
        neighbours_2d, neighbourhood_mask = self.toMasked2DArray(nodes)
        return Selection(indices.numpy(), neighbours_2d, neighbourhood_mask)
    
    '''
        Patch Alignment
    '''

    def getEigh(self, selection: Selection):
        SIGMA_1 = 3
        P = selection.neighbourhood.shape[1]

        _g = self.object.g
        # To Numpy convertions
        # (N, 3)
        n = _g.n.numpy()
        # (N, 3)
        ci = _g.pos.numpy()
        # (N, 1)
        _a = _g.a.numpy()

        # (N, P, 3)
        cj = ci[selection.neighbourhood]
        # (N, P, 3)
        cjci = cj - ci[selection.indices, None, :]
        temp_norms = np_linalg_norm(cjci, axis=2)
        temp_norms[selection.mask] = 0
        # (N,)
        scale_factors = 1 / np_max(temp_norms, axis=1)
        # (N, P, 3) (Translated and scaled)
        dcs = cjci * scale_factors[:, None, None]
        # (N, P, 3)
        nj = n[selection.neighbourhood]
        # (N, P, 3)
        wj = np_cross(np_cross(dcs, nj, axis=2), dcs).reshape(-1, 3) # Reshape is done for normalize method to work
        sklearn_preprocessing_normalize(wj, copy=False) # Normalize wj in place
        wj = wj.reshape(-1, P, 3)
        # (N, P, 3)
        njprime = 2 * np_sum(nj * wj, axis=2)[:, :, None] * wj - nj
        # Big problem. Pointclouds don't have areas. What do we do about this?!?!
        # Reasoning: If the area of the normal vector is small, is doesn't influence the final voting tensor.
        # Should we use neighbourhood density to have a scaling factor or not?
        # (N, P)
        areas = _a[selection.neighbourhood] * scale_factors[:, None] ** 2
        areas[selection.mask] = 0
        # (N,)
        maxArea = np_max(areas, axis=1)
        # (N, P)
        ddcs = np_linalg_norm(dcs, axis=2)
        # (N, P)
        muj = (areas / maxArea[:, None])*np_exp(-ddcs*SIGMA_1)
        # (N, P, 3, 3)
        outer = njprime[..., None] * njprime[..., None, :]
        # (N, P, 3, 3)
        Tj = muj[..., None, None] * outer
        # Before summing, set the nonsense values to zero!
        Tj[selection.mask] = 0
        # (N, 3, 3)
        Ti = np_sum(Tj, axis=1)
        # ((N, 3), (N, 3, 3))
        eigh = np_linalg_eigh(Ti)
        return ci, scale_factors, eigh
    
    @classmethod
    def getGroups(cls, ev):
        N = ev.shape[0]
        ev_f = np_sort(ev, axis=1)[:, ::-1]
        flat = np_logical_and(ev_f[:, 1] < 0.01, ev_f[:, 2] < 0.001)
        edge = np_logical_and(ev_f[:, 1] > 0.01, ev_f[:, 2] < 0.1)
        corner = ev_f[:, 2] > 0.1
        char = np_zeros(N)
        char[flat] = 1
        char[edge] = 2
        char[corner] = 3
        return char
    
    def getRInv(self, eigh, selection: Selection):
        N = eigh[0].shape[0]
        _g = self.object.g
        # (N, 3)
        n = _g.n.numpy()[selection.indices]
        # (N, 3)
        ev_order = np_argsort(eigh[0], axis=1)[:, ::-1]
        # (N, 3, 3)
        eigh_T = np_transpose(eigh[1], axes=(0, 2, 1))
        # (N, 3, 3)
        R = eigh_T[np_arange(N)[:, None, None], ev_order[..., None], np_arange(3)[None, None]]
        # (N,)
        R[np_sum(R[:, 0, :] * n, axis=1) < 0] *= -1
        # (N,)
        R[np_linalg_det(R) < 0, 2] *= -1
        # (N, 3, 3)
        R_inv = np_transpose(R, axes = (0, 2, 1))
        return R_inv

    @classmethod
    def applyRInv(cls, R_inv, other):
        R_inv_shape = R_inv.shape
        other_shape = other.shape
        len_other_shape = len(other_shape)
        if len(R_inv_shape) == 3 and R_inv_shape[0] == other_shape[0]:
            if len_other_shape == 2:
                return np_einsum("ni,nij->nj", other, R_inv)
            elif len_other_shape == 3:
                return np_einsum("npi,nij->npj", other, R_inv)
        else:
            raise ValueError(f"Input arrays were not of the correct shape:\nR_inv shape: {R_inv_shape}\nOther shape: {other_shape}")
    
    def getGraph(self, index, selection: Selection, R_inv, scale_factors):
        p = selection.neighbourhood[index][~selection.mask[index]]
        _object = self.object
        _g = _object.g
        c = _g.pos[p]
        nj = _g.n[p]
        R_inv_i = R_inv[index]
        nj_R_inv = nj @ R_inv_i
        gt = _g.y[index]
        gt_R_inv = gt[None] @ R_inv_i
        n = nj_R_inv
        a = torch_from_numpy(_object.getAreas()[p] * scale_factors[index])[:, None]
        d = tg_utils_degree(_g.edge_index[0].to(torch_int64))[p, None]
        x = torch_cat((c, n, a, d), dim=1).to(torch_float32)
        edge_index = tg_utils_subgraph(torch_from_numpy(p).long(), _g.edge_index, relabel_nodes=True)[0]
        y = gt_R_inv
        return tg_data_Data(x=x, edge_index=edge_index, y=y)
    
    '''
        Facade methods
    '''

    def getClasses(self, mode=DEFAULT_SELECTION_MODE, k=DEFAULT_PATCH_RESOLUTION_K):
        selection = self.toPatchIndices(indices=None, mode=mode, k=k)
        _, _, eigh = self.getEigh(selection)
        return torch_from_numpy(Preprocessor.getGroups(eigh[0]))
    
    def getGraphs(self, indices, mode=DEFAULT_SELECTION_MODE, k=DEFAULT_PATCH_RESOLUTION_K):
        selection = self.toPatchIndices(indices=indices, mode=mode, k=k)
        _, sf, eigh = self.getEigh(selection)
        RInv = self.getRInv(eigh, selection)
        return [self.getGraph(x, selection, RInv, sf) for x in range(len(indices))]