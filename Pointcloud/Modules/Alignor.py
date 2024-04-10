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

class Alignor:

    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ["pos", "n"])
        self.graph = graph


    def getNormalVotingTensorTransforms(self, selection: SlicedTorchData):
        SIGMA_1 = 3

        _g = self.graph
        _device = _g.edge_index.device
        _data = selection.data
        # To Numpy convertions
        N = _g.num_nodes
        P = _data.size(0)
        # (N, 3)
        _n = _g.n
        # (N, 3)
        ci = _g.pos
        # (N, 1)
        _mass = _g.mass
        bindex = selection._batchIndices()

        # (P, 3)
        cj = ci[_data]
        # (P, 3)
        cjci = cj - ci[bindex]
        # (P,)
        distances = cjci.norm(dim=1)
        # (N,)
        scale_factors = 1 / torch_zeros(N, dtype=cjci.dtype, device=_device) \
                                .scatter_reduce_(dim=0, index=bindex, src=distances, reduce="amax")
        # (P,)
        scale_factors_ = scale_factors[bindex]
        # (P, 3)
        dcs = cjci * scale_factors_[:, None]
        # (P, 3)
        nj = _n[_data]
        # (P, 3)
        wj = wj = torch_nn_functional_normalize(torch_cross(torch_cross(dcs, nj, dim=1), dcs, dim=1), p=2, dim=1)
        # (P, 3)
        njprime = 2 * (nj * wj).sum(dim=1)[:, None] * wj - nj
        # Big problem. Pointclouds don't have areas. What do we do about this?!?!
        # Reasoning: If the area of the normal vector is small, it doesn't influence the final voting tensor.
        # Should we use neighbourhood density to have a scaling factor or not?
        # (P,)
        areas = _mass[_data] * scale_factors_ ** 2
        # (P,)
        maxArea = torch_zeros(N, dtype=areas.dtype, device=_device) \
                        .scatter_reduce_(dim=0, index=bindex, src=areas, reduce="amax")[bindex]
        # (P,)
        ddcs = dcs.norm(dim=1)
        # (P,)
        muj = (areas / maxArea)*(-ddcs*SIGMA_1).exp_()
        # (P, 3, 3)
        outer = njprime[..., None] * njprime[:, None]
        # (N, P, 3, 3)
        Tj = muj[..., None, None] * outer
        # (N, 3, 3)
        Ti = torch_zeros((N, 3, 3), dtype=Tj.dtype, device=_device) \
                    .scatter_reduce_(
                        dim=0,
                        index=bindex[:, None, None].expand(P, 3, 3),
                        src=Tj,
                        reduce="sum"
                    )
        # ((N, 3), (N, 3, 3))
        eigh = torch_linalg_eigh(Ti)
        return ci, scale_factors, eigh
    
    @classmethod
    def getGroups(cls, eigval: torch_Tensor) -> torch_Tensor:
        assert eigval.dim == 2
        assert eigval.size(1) == 3
        assert eigval.is_floating_point()

        N = eigval.size(0)
        eigval_f = eigval.sort(dim=1, descending=True)
        flat = torch_logical_and(eigval_f[:, 1] < 0.01, eigval_f[:, 2] < 0.001)
        edge = torch_logical_and(eigval_f[:, 1] > 0.01, eigval_f[:, 2] < 0.1)
        corner = eigval_f[:, 2] > 0.1
        char = torch_zeros(N, dtype=torch_long, device=eigval.device)
        char[flat] = 1
        char[edge] = 2
        char[corner] = 3
        return char
    
    def getRInv(self, eigh: typing_Tuple[torch_Tensor, torch_Tensor], selection: SlicedTorchData, indices: torch_Tensor = None) -> torch_Tensor:
        assert eigh[0].dim() == 2 and eigh[1].dim() == 3
        assert eigh[0].size(1) == 3 and eigh[1].size(1) == 3 and eigh[1].size(2) == 3
        assert eigh[0].size(0) == eigh[1].size(0)
        assert eigh[0].is_floating_point() and eigh[1].is_floating_point()
        if indices is not None:
            assert indices.dim() == 1
            assert indices.size(0) == eigh[0].size(0)
            assert indices.is_floating_point()

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
        R = R = eigvec_T[
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