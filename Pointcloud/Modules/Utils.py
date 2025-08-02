from dataclasses import dataclass
from igl import (
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency,
    per_vertex_normals as igl_per_vertex_normals,
    per_face_normals as igl_per_face_normals
)
from itertools import zip_longest as itertools_zip_longest
from numpy import (
    any as np_any,
    arange as np_arange,
    argwhere as np_argwhere,
    array as np_array,
    cross as np_cross,
    logical_not as np_logical_not,
    nan_to_num as np_nan_to_num,
    ndarray as np_ndarray,
    ones as np_ones,
    r_ as np_r_,
    unique as np_unique,
    zeros as np_zeros
)
from sklearn.preprocessing import normalize as sklearn_preprocessing_normalize
from torch import (
    arange as torch_arange,
    bool as torch_bool,
    cat as torch_cat,
    float as torch_float,
    from_numpy as torch_from_numpy,
    long as torch_long,
    ones as torch_ones,
    repeat_interleave as torch_repeat_interleave,
    stack as torch_stack,
    Tensor as torch_Tensor,
    zeros as torch_zeros,
    zeros_like as torch_zeros_like
)
from torch.nn.functional import (
    normalize as t_nn_f_normalize
)
from torch_geometric.data import (
    Data as tg_data_Data
)
from torch_geometric.nn.pool import (
    knn as tg_nn_pool_knn
)
from typing import (
    Any as typing_Any,
    Collection as typing_Collection,
    Tuple as typing_Tuple
)
from warnings import warn

class GeneralUtils:
    
    @classmethod
    def validateAttributes(cls, obj: typing_Any, attrs: typing_Collection[str]) -> None:
        r"""
        Validates if the object has all attributes listed.

        Args:
            obj: The object to validate.
            attrs: A list of attribute names that the object must contain.
        """
        for attr in attrs:
            if not hasattr(obj, attr) or getattr(obj, attr) is None:
                raise ValueError(f"Object does not have attribute '{attr}'.")

class NumpyUtils:

    DEGENERATE_NORMAL_PLACEHOLDER = np_zeros(3)

    @classmethod
    def calculateVertexNormals(cls, v: np_ndarray, f: np_ndarray) -> np_ndarray:
        return np_nan_to_num(igl_per_vertex_normals(v, f), copy=False, nan=0)

    @classmethod
    def calculateFaceNormals(cls, v: np_ndarray, f: np_ndarray) -> np_ndarray:
        return igl_per_face_normals(v, f, cls.DEGENERATE_NORMAL_PLACEHOLDER)
    
    @classmethod
    # https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
    # starts: 1D numpy array of indices of size n representing the starts of the ranges.
    # ends: 1D numpy array of indices of size n representing the ends of the ranges.
    # return: 1D numpy array of indices with indices that are within one of the given ranges.
    # WARNING: The list can be unsorted and contain duplicates.
    def rangeBoundariesToIndices(cls, starts: np_ndarray, ends: np_ndarray) -> np_ndarray:
        l = ends - starts
        nonsense_ids = l <= 0
        if np_any(nonsense_ids):
            # Lengths of ranges that are zero are nonsense and should be ignored.
            # Ranges with a negative length have the start and end reversed.
            #   This should be fixed before calling this function for efficiency.
            warn(f"Nonsensible ranges are given and will be ignored! (IDs: {np_arange(len(nonsense_ids))[nonsense_ids]}, Range lengths: {l[nonsense_ids]})")
            sensible_ids = np_logical_not(nonsense_ids)
            starts = starts[sensible_ids]
            ends = ends[sensible_ids]
            l = ends - starts
        clens = l.cumsum()
        ids = np_ones(clens[-1],dtype=int)
        ids[0] = starts[0]
        ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
        return ids.cumsum()
    
            
    @classmethod
    def packSequence(cls, data2d: np_ndarray, mask: np_ndarray):
        assert data2d.ndim == 2 and mask.ndim == 2
        assert data2d.shape == mask.shape
        assert mask.dtype == bool

        data = data2d.reshape(-1)[~mask.reshape(-1)]
        slices = np_r_[np_zeros(1, dtype=int), (1 - mask).sum(axis=1).cumsum()]
        return data, slices

    @classmethod
    def toMasked2DArray(cls, indices: np_ndarray[list[int]]) -> typing_Tuple[np_ndarray, np_ndarray]:
        # https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values
        indices_2d = np_array(list(itertools_zip_longest(*indices, fillvalue=-1))).T
        mask = indices_2d == -1 # Detect mask
        indices_2d[mask] = 0 # Set mask to center node to not throw errors while indexing
        return indices_2d, mask

@dataclass
class SlicedTorchData:
    data: torch_Tensor
    slices: torch_Tensor

    def __post_init__(self):
        _data = self.data
        _slices = self.slices
        assert _data.dim() == 1 and _slices.dim() == 1
        assert not _slices.is_floating_point()
        assert _slices[0] == 0 and _slices[-1] == _data.size(0)

    def __len__(self):
        return self.slices.size(0) - 1
    
    def __getitem__(self, key: int):
        assert key >= 0 and key <= len(self)
        _slices = self.slices
        return self.data[_slices[key]:_slices[key+1]]
    
    def __setattr__(self, name: str, value: torch_Tensor):
        if name == "data":
            self._assertData(value)
        elif name == "slices":
            self._assertSlices(value)
        object.__setattr__(self, name, value)

    def _assertData(self, _data: torch_Tensor):
        assert _data.dim() == 1, f"Actual size of data: {_data.size()}"
        if hasattr(self, "slices"):
            assert _data.size(0) == self.slices[-1]

    def _assertSlices(self, _slices: torch_Tensor):
        assert _slices.dim() == 1
        assert not _slices.is_floating_point()
        assert _slices[0] == 0, f"Slider start: {_slices[0]}"
        if hasattr(self, "data"):
            assert self.data.size(0) == _slices[-1], f"Data size: {self.data.size(0)}\nSlices last entry: {_slices[-1]}"
    
    def _batchIndices(self, indices: torch_Tensor = None) -> torch_Tensor:
        _slices = self.slices
        assert _slices.dim() == 1, "slices must have 2 dimensions"
        assert not _slices.is_floating_point(), "slices must contain integers"
        assert _slices is not None, "slices cannot be None"

        starts = _slices[:-1]
        ends = _slices[1:]
        lengths = ends - starts
        if indices is None:
            indices = torch_arange(lengths.size(0), dtype=torch_long, device=_slices.device)
            repeat_indices = indices
        else:
            TorchUtils.validateIndices(indices)
            repeat_indices = torch_arange(indices.size(0), dtype=torch_long, device=_slices.device)
        i = torch_repeat_interleave(repeat_indices, lengths[indices])
        j = self.data[TorchUtils.rangeBoundariesToIndices(starts[indices], ends[indices])]
        return i, j
    
    def scatterReduce(self, reduction: str) -> torch_Tensor:
        assert reduction in ("sum", "prod", "mean", "amax", "amin")

        _data = self.data
        means = torch_zeros(len(self), dtype=torch_float, device=_data.device) \
            .scatter_reduce_(
                dim=0, 
                index=self._batchIndices()[0], 
                src=_data.to(torch_float), 
                reduce=reduction
            )
        return means

class TorchUtils:

    @classmethod
    def toEdgeTensor(cls, output: np_ndarray) -> torch_Tensor:
        col = torch_from_numpy(output).long()
        k = col.size(1)
        row = torch_arange(col.size(0), dtype=torch_long).view(-1, 1).repeat(1, k)
        return torch_stack([row.reshape(-1), col.reshape(-1)], dim=0).long()
    
    @classmethod
    def validateIndices(cls, indices: torch_Tensor) -> None:
        assert indices is None or (not indices.is_floating_point() and indices.dim() == 1), f"indices dimensions: {indices.dim()}\nindices type: {indices.dtype}"

    @classmethod
    def validateEdgeIndex(cls, _edge_index: torch_Tensor) -> None:
        assert not _edge_index.is_floating_point(), f"_edge_index should be indices and not floating points.."
        assert _edge_index.dim() == 2, f"_edge_index dimensions: {_edge_index.dim()}"
        assert _edge_index.size(0) == 2, f"_edge_index first dimension: {_edge_index.size(0)}"

    r"""
    Asserts edge index to have a unique count per node.
    This is equal to k and is also returned
    """
    @classmethod
    def validateKNNEdgeIndex(cls, _edge_index: torch_Tensor) -> None:
        cls.validateEdgeIndex(_edge_index)
        unique_node_counts = _edge_index[0].unique(return_counts=True)[1].unique()
        assert unique_node_counts.size(0) == 1
        return unique_node_counts[0]

    @classmethod
    def face2vertexNormals(cls, v: torch_Tensor, fv: torch_Tensor, n: torch_Tensor, fn: torch_Tensor) -> torch_Tensor:
        r"""
        Computes vertex normals based on faces.
        """
        assert v.dim() == 2 and fv.dim() == 2 and n.dim() == 2 and fn.dim() == 2, f"Dimensions:\nv: {v.dim()}\nfv: {fv.dim()}\nn: {n.dim()}\nfn: {fn.dim()}"
        assert v.is_floating_point() and n.is_floating_point() and not fv.is_floating_point() and not fn.is_floating_point(), f"dtypes:\nv: {v.dtype}\nn: {n.dtype}\nfv: {fv.dtype}\nfn: {fn.dtype}"
        assert fv.size(0) == fn.size(0), "For every face there should be references to points and normals."
        assert v.size(1) == n.size(1), "v and n should have the same number of euclidian dimensions"
        
        vn = torch_zeros_like(v, device=v.device)
        vn.index_add_(0, fv.view(-1), n[fn].view(-1, 3))
        return t_nn_f_normalize(vn, dim=-1)
    
    @classmethod
    def maskToSlicedTorchData(cls, mask: torch_Tensor) -> torch_Tensor:
        assert mask.dim() == 2, "mask must have 2 dimensions"
        assert mask.dtype == torch_bool, "Mask must contain booleans"
        assert mask is not None, "Mask cannot be None"

        N1, N2 = mask.size(0), mask.size(1)
        idxs = mask.view(-1).nonzero().squeeze(1)
        data = idxs.remainder(N2)
        slices = torch_zeros(N1 + 1, dtype=torch_long, device=mask.device)
        u_idx, count = idxs.div_(N2, rounding_mode="floor").unique(return_counts=True)
        slices[u_idx + 1] = count
        slices = slices.cumsum_(dim=0)
        return SlicedTorchData(data, slices)
    
    @classmethod
    def ChamferDistance(cls, pos0: torch_Tensor, pos1: torch_Tensor) -> float:
        assert pos0.dim() == 2
        assert pos1.dim() == 2
        assert pos0.size(1) == 3
        assert pos1.size(1) == 3

        knn0 = tg_nn_pool_knn(pos0, pos1, 1)
        knn1 = tg_nn_pool_knn(pos1, pos0, 1)
        chamfer0 = (pos0[knn0[1]] - pos1[knn0[0]]).square().sum(dim=1)
        chamfer1 = (pos1[knn1[1]] - pos0[knn1[0]]).square().sum(dim=1)
        
        return torch_cat([chamfer0, chamfer1], dim=0)
    
    @classmethod
    def HausdorffDistance(cls, pos0: torch_Tensor, pos1: torch_Tensor) -> float:
        assert pos0.dim() == 2
        assert pos1.dim() == 2
        assert pos0.size(1) == 3
        assert pos1.size(1) == 3

        knn0 = tg_nn_pool_knn(pos0, pos1, 1)
        knn1 = tg_nn_pool_knn(pos1, pos0, 1)
        chamfer0 = (pos0[knn0[1]] - pos1[knn0[0]]).norm(dim=1)
        chamfer1 = (pos1[knn1[1]] - pos0[knn1[0]]).norm(dim=1)
        
        return torch_cat([chamfer0, chamfer1], dim=0)
    
    @classmethod
    def PaperDistance(cls, gt: torch_Tensor, noisy: torch_Tensor) -> float:
        r'''
            Normalize Average Hausdorff Distance according to the paper
        '''
        assert gt.dim() == 2
        assert noisy.dim() == 2
        assert gt.size(1) == 3
        assert noisy.size(1) == 3

        bounding_box_diagonal = (gt.max(dim=0).values - gt.min(dim=0).values).norm(dim=0)
        knn = tg_nn_pool_knn(gt, noisy, 1)
        hausdorff = (gt[knn[1]] - noisy[knn[0]]).norm(dim=1) / bounding_box_diagonal
        
        return hausdorff

    @classmethod
    def averageEdgeLength(cls, pos: torch_Tensor, edge_index: torch_Tensor):
        return (pos[edge_index[1]] - pos[edge_index[0]]).norm(dim=1).mean(dim=0)
    
    @classmethod
    def pointcloudRadius(cls, pos: torch_Tensor):
        return (pos - pos.mean(dim=0, keepdim=True)).norm(dim=1).max(dim=0).values

    @classmethod
    # https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
    # starts: 1D torch tensor of indices of size n representing the starts of the ranges.
    # ends: 1D torch tensor of indices of size n representing the ends of the ranges.
    # return: 1D torch tensor of indices with indices that are within one of the given ranges.
    # WARNING: The list can be unsorted and contain duplicates.
    def rangeBoundariesToIndices(cls, starts: torch_Tensor, ends: torch_Tensor) -> torch_Tensor:
        assert starts.dtype == torch_long
        l = ends - starts
        nonsense_ids = l <= 0
        if nonsense_ids.any():
            # Lengths of ranges that are zero are nonsense and should be ignored.
            # Ranges with a negative length have the start and end reversed.
            #   This should be fixed before calling this function for efficiency.
            warn(f"Nonsensible ranges are given and will be ignored! (IDs: {nonsense_ids.nonzero().flatten()}, Range lengths: {l[nonsense_ids]})")
            starts = starts[~nonsense_ids]
            ends = ends[~nonsense_ids]
            l = ends - starts
        clens = l.cumsum_(dim=0)
        ids = torch_ones(clens[-1],dtype=torch_long)
        ids[0] = starts[0]
        ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
        return ids.cumsum(dim=0)
    
class DeprecatedUtils:

    '''
        DEPRECATED DEFINITIONS
    '''

    # v: (num_vertices, num_axes) -> Euclidian position
    # f: (num_faces, num_corners) -> vertex_index
    # n: (num_faces, num_axes) -> Euclidian position
    @classmethod
    def vertexNormalsFromFaceNormals(cls, v: np_ndarray, f: np_ndarray, n: np_ndarray) -> np_ndarray:
        warn("This method is slow and therefore deprecated! Use igl.per_vertex_normals instead!")
        # vta: ((3*num_faces,), (num_vertices+1)) -> (neighbor_face_index, cumulative_vertex_degree)
        vta = igl_vertex_triangle_adjacency(f, len(v))
        # vi: (num_vertices,) -> all vertex indices
        vi = np_arange(len(v))
        # v_degree: (num_vertices,) -> vertex_degree
        v_degree = vta[1][vi+1] - vta[1][vi]
        # unique_degrees: (num_unique_vertex_degrees,) -> vertex_degree
        # vi_di: (num_vertices,) -> unique_vertex_degree_index
        unique_degrees, vi_di = np_unique(v_degree, return_inverse=True)
        vertex_normals = np_zeros((len(v), 3))
        # di: degree_i
        # degree: vertex_degree
        for di, unique_degree in enumerate(unique_degrees):
            # v_with_degree: (num_vertices_with_degree,) -> vertex_index
            v_with_degree = np_argwhere(vi_di == di).reshape(-1)
            # vta_to_index: (num_faces_of_vertices_with_degree,) -> vta_index
            vta_to_index = cls.rangeBoundariesToIndices(vta[1][v_with_degree].reshape(-1), vta[1][v_with_degree+1].reshape(-1))
            # faces_of_v_with_degree: (num_vertices_with_degree, unique_degree) -> face_index
            faces_of_v_with_degree = vta[0][vta_to_index].reshape(-1, unique_degree)
            # normals_of_faces: (num_vertices_with_degree, unique_degree, num_axes) -> Euclidian position
            normals_of_faces = n[faces_of_v_with_degree]
            # normalized_summed_normals: (num_vertices_with_degree, num_axes) -> Euclidian position
            normalized_summed_normals = sklearn_preprocessing_normalize(normals_of_faces.sum(axis=1))
            vertex_normals[v_with_degree] = normalized_summed_normals
        return vertex_normals
    
    # v: (num_vertices, num_axes) -> Euclidian position
    # f: (num_faces, num_corners) -> vertex_index
    @classmethod
    def normalsOfFaces(cls, f: np_ndarray, v: np_ndarray) -> np_ndarray:
        warn("This method is slow and therefore deprecated! Use igl.per_face_normals instead!")
        # fv: (num_faces, num_corners, num_axes) -> Euclidian position
        fv = v[f]
        # crosses: (num_faces, num_axes) -> Euclidian position
        normals = sklearn_preprocessing_normalize(np_cross(fv[:, 1, :] - fv[:, 0, :], fv[:, 2, :] - fv[:, 1, :]))
        return normals
