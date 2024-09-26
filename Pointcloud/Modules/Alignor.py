from Pointcloud.Modules.Utils import GeneralUtils, SlicedTorchData

from torch import (
    arange as torch_arange,
    cross as torch_cross,
    einsum as torch_einsum,
    long as torch_long,
    logical_and as torch_logical_and,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch.linalg import (
    det as torch_linalg_det,
    eigh as torch_linalg_eigh
)
from torch.nn.functional import (
    normalize as torch_nn_functional_normalize
)
from torch_geometric.data import (
    Data as tg_data_Data
)
from typing import Tuple as typing_Tuple

def NVT(bindex: torch_Tensor, weight: torch_Tensor, vector: torch_Tensor):
    _device = weight.device
    # (P, 3, 3)
    outer = vector[..., None] * vector[:, None]
    # (N, P, 3, 3)
    Tj = weight[..., None, None] * outer
    # (I, 3, 3)
    return torch_zeros((bindex.max(dim=0).values + 1, 3, 3), dtype=Tj.dtype, device=_device) \
                .index_add_(dim=0, index=bindex, source=Tj)

class Alignor:

    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ["pos", "n"])
        self.graph = graph

    def getMDTransformation(self, selection: SlicedTorchData):
        SIGMA_1 = 3

        _g = self.graph
        _device = _g.edge_index.device
        _data = selection.data
        bindex = selection._batchIndices()[0]
        N = _g.num_nodes

        _n = _g.n
        _pos = _g.pos
        _mass = _g.mass
        ci = _pos[bindex]
        cj = _pos[_data]
        dv = cj - ci
        scale_factors = 1 / torch_zeros(N, dtype=dv.dtype, device=_device) \
                                .scatter_reduce_(dim=0, index=bindex, src=dv.norm(dim=1), reduce="amax")
        scale_factors_i = scale_factors[bindex]
        dv_scaled = dv * scale_factors_i[:, None]
        nj = _n[_data]
        # (P, 3)
        wj = wj = torch_nn_functional_normalize(torch_cross(torch_cross(dv_scaled, nj, dim=1), dv_scaled, dim=1), p=2, dim=1)
        # (P, 3)
        njprime = 2 * (nj * wj).sum(dim=1)[:, None] * wj - nj
        # Big problem. Pointclouds don't have areas. What do we do about this?!?!
        # Reasoning: If the area of the normal vector is small, it doesn't influence the final voting tensor.
        # Should we use neighbourhood density to have a scaling factor or not?
        # (P,)
        areas = _mass[_data] * scale_factors_i ** 2
        # (P,)
        maxArea = torch_zeros(N, dtype=areas.dtype, device=_device) \
                        .scatter_reduce_(dim=0, index=bindex, src=areas, reduce="amax")[bindex]
        # (P,)
        ddcs = dv_scaled.norm(dim=1)
        # (P,)
        muj = (areas / maxArea)*(-ddcs*SIGMA_1).exp_()
        # (I, 3, 3)
        Ti = NVT(bindex, muj, njprime)
        # ((I, 3), (I, 3, 3))
        eigh = torch_linalg_eigh(Ti)
        return _pos, scale_factors, eigh
    
    @classmethod
    def getMDFeatures(cls, eigval: torch_Tensor) -> torch_Tensor:
        assert eigval.dim() == 2
        assert eigval.size(1) == 3
        assert eigval.is_floating_point()

        N = eigval.size(0)
        eigval_f = eigval.sort(dim=1, descending=True).values
        flat = torch_logical_and(eigval_f[:, 1] < 0.01, eigval_f[:, 2] < 0.001)
        edge = torch_logical_and(eigval_f[:, 1] > 0.01, eigval_f[:, 2] < 0.1)
        corner = eigval_f[:, 2] > 0.1
        char = torch_zeros(N, dtype=torch_long, device=eigval.device)
        char[flat] = 1
        char[edge] = 2
        char[corner] = 3
        return char
    
    def getRInv(self, eigh: typing_Tuple[torch_Tensor, torch_Tensor], indices: torch_Tensor = None) -> torch_Tensor:
        assert eigh[0].dim() == 2 and eigh[1].dim() == 3
        assert eigh[0].size(1) == 3 and eigh[1].size(1) == 3 and eigh[1].size(2) == 3
        assert eigh[0].size(0) == eigh[1].size(0)
        assert eigh[0].is_floating_point() and eigh[1].is_floating_point()
        if indices is not None:
            assert indices.dim() == 1
            assert indices.size(0) == eigh[0].size(0), f"indices.size(): {indices.size()}\n eigh[0].size(): {eigh[0].size()}"
            assert not indices.is_floating_point()

        _g = self.graph
        eigval, eigvec = eigh
        _device = eigval.device
        indices = indices if indices is not None else torch_arange(_g.num_nodes, dtype=torch_long, device=_device)
        N = indices.size(0)
        # (N, 3)
        _n = _g.n[indices]
        # (N, 3)
        eigval_order = eigval.argsort(dim=-1, descending=True)
        # (N, 3, 3)
        eigvec_T = eigvec.transpose(dim0=1, dim1=2)
        # (N, 3, 3)
        R = eigvec_T[
            torch_arange(N, dtype=torch_long, device=_device)[:, None, None],
            eigval_order[..., None],
            torch_arange(3, dtype=torch_long, device=_device)[None, None]
        ]
        R[(R[:, 0, :] * _n).sum(dim=1) < 0] *= -1
        R[torch_linalg_det(R) < 0, 2] *= -1
        # (N, 3, 3)
        R_inv = R.transpose_(dim0=1, dim1=2)
        return R_inv

    @classmethod
    def applyRInv(cls, R_inv: torch_Tensor, other: torch_Tensor) -> torch_Tensor:
        len_other_shape = other.dim()
        if R_inv.dim() == 3 and R_inv.size(0) == other.size(0):
            if len_other_shape == 2:
                return torch_einsum("ni,nij->nj", other, R_inv)
        else:
            raise ValueError(f"Input arrays were not of the correct shape")
        
    def getVUFilteredNormals(self, selection: SlicedTorchData, rho: float = 0.9, tau: float = 0.3, d: float = 3):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        _g = self.graph
        
        GeneralUtils.validateAttributes(_g, ["pos", "n"])

        _data = selection.data
        _device = _data.device
        _dtype_ints = _data.dtype
        _n = _g.n
        I = len(selection)
        bindex = selection._batchIndices()[0]
        ni = _n[bindex]
        nj = _n[_data]
        wij = ((ni * nj).sum(dim=1).clamp_(-1, 1).acos_() <= rho).to(_dtype_ints)
        summed_wij = torch_zeros((I,), dtype=_dtype_ints, device=_device) \
            .index_add_(dim=0, index=bindex, source=wij)
        Ti = NVT(bindex, wij, nj) / summed_wij[:, None, None]
        eigval, eigvec = torch_linalg_eigh(Ti)
        ordered_eigval = eigval.sort(dim=1, descending=True)
        batch_idx = torch_arange(I, device=_device)[:, None]
        ordered_eigvec = eigvec[batch_idx, :, ordered_eigval.indices].transpose(1, 2)
        new_eigval = (ordered_eigval.values > tau).to(int)
        new_n = d*_n + ((new_eigval * (ordered_eigvec * _n[:, None]).sum(dim=2))[..., None] * ordered_eigvec).sum(dim=1)
        return new_n / new_n.norm(dim=1, keepdim=True)

    def getVUDecomposition(self, selection: SlicedTorchData, _n: torch_Tensor, rho: float = 0.9):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        _g = self.graph
        _data = selection.data
        _device = _data.device
        _dtype_ints = _data.dtype
        I = len(selection)
        bindex = selection._batchIndices()[0]
        _v = _g.pos
        _dtype_floats = _v.dtype

        vj = _v[_data]
        ni = _n[bindex]
        nj = _n[_data]
        wij = ((ni * nj).sum(dim=1).clamp_(-1, 1).acos_() <= rho).to(_dtype_ints)
        wvij = wij[..., None] * vj
        summed_wij = torch_zeros((I,), dtype=_dtype_ints, device=_device) \
            .index_add_(dim=0, index=bindex, source=wij)
        summed_wvij = torch_zeros((I, 3), dtype=_dtype_floats, device=_device) \
            .index_add_(dim=0, index=bindex, source=wvij)
        v_center = summed_wvij / summed_wij[:, None]
        dv = vj - v_center[bindex]
        summed_wdv = NVT(bindex, wij, dv)
        Ci = summed_wdv / summed_wij[:, None, None]
        eigval, eigvec = torch_linalg_eigh(Ci)
        ordered_eigval = eigval.sort(dim=1, descending=True)
        batch_idx = torch_arange(I, device=_device)[:, None]
        ordered_eigvec = eigvec[batch_idx, :, ordered_eigval.indices].transpose(1, 2)
        return ordered_eigval.values, ordered_eigvec
    
    def getVUFeatures(self, eigval: torch_Tensor, mean_graph_edge_length: float, k: int = 6) -> torch_Tensor:
        tau = 16.0 / k * mean_graph_edge_length ** 2
        return (eigval < tau).sum(dim=1) % 3
    