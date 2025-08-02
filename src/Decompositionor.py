from .Selector import Selection
from .Utils import GeneralUtils

from dataclasses import dataclass
from torch import (
    arange as torch_arange,
    cat as torch_cat,
    cross as torch_cross,
    logical_and as torch_logical_and,
    stack as torch_stack,
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

@dataclass
class Decomposition:
    eigval: torch_Tensor
    eigvec: torch_Tensor

    def __post_init__(self):
        _eigval = self.eigval
        _eigvec = self.eigvec
        self._assertEigval(_eigval)
        self._assertEigvec(_eigvec)
        assert _eigval.size(0) == _eigvec.size(0)

    def __len__(self):
        return self.eigval.size(0)
    
    def __setattr__(self, name: str, value: torch_Tensor):
        if name == "data":
            self._assertData(value)
        elif name == "slices":
            self._assertSlices(value)
        object.__setattr__(self, name, value)

    def _assertEigval(self, _eigval: torch_Tensor):
        assert _eigval.dim() == 2
        assert _eigval.size(1) == 3
        assert _eigval.is_floating_point()

    def _assertEigvec(self, _eigvec: torch_Tensor):
        assert _eigvec.dim() == 3
        assert _eigvec.size(1) == 3 and _eigvec.size(2) == 3
        assert _eigvec.is_floating_point()

    def getNVTFeatures(self):
        features = self.eigval
        lambda1, lambda2, lambda3 = features[:, 2], features[:, 1], features[:, 0]
        linearity = (lambda2 - lambda3) / lambda1
        planarity = (lambda1 - lambda2) / lambda1
        sphericity = lambda3 / lambda1
        return planarity, linearity, sphericity
    
    def getClasses(self, scale: float = 0.2):
        planarity, linearity, sphericity = self.getNVTFeatures()
        features = torch_stack([planarity, linearity, sphericity], dim=1)
        features[:, 0] *= scale
        return features.argmax(dim=1)
    
    def getMDFeatures(self) -> torch_Tensor:
        _eigval = self.eigval
        N = _eigval.size(0)
        eigval_f = _eigval.sort(dim=1, descending=True).values
        flat = torch_logical_and(eigval_f[:, 1] < 0.01, eigval_f[:, 2] < 0.001)
        edge = torch_logical_and(eigval_f[:, 1] > 0.01, eigval_f[:, 2] < 0.1)
        corner = eigval_f[:, 2] > 0.1
        char = torch_zeros(N, dtype=int, device=_eigval.device)
        char[flat] = 1
        char[edge] = 2
        char[corner] = 3
        return char
    
    def getVUFeatures(self, tau: float) -> torch_Tensor:
        return (self.eigval < tau).sum(dim=1) % 3
    
    def getBetterVUFeatures(self, mean_graph_edge_length: float, k: int = 6) -> torch_Tensor:
        eigval = self.eigval
        tau = 16.0 / k * mean_graph_edge_length ** 2
        return (eigval < tau).sum(dim=1) % 3
        
    def getVUSmoothedNormals(self, n: torch_Tensor, tau: float = 0.3, d: float = 3):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        _device = n.device
        eigval, eigvec = self.eigval, self.eigvec
        N = eigval.size(0)
        ordered_eigval = eigval.sort(dim=1, descending=True)
        batch_idx = torch_arange(N, device=_device)[:, None]
        ordered_eigvec = eigvec[batch_idx, :, ordered_eigval.indices].transpose(1, 2)
        new_eigval = (ordered_eigval.values > tau).to(int)
        new_n = d*n + ((new_eigval * (ordered_eigvec * n[:, None]).sum(dim=2))[..., None] * ordered_eigvec).sum(dim=1)
        return new_n / new_n.norm(dim=1, keepdim=True)
    
    def getRInv(self, n: torch_Tensor, indices: torch_Tensor = None) -> torch_Tensor:
        eigval, eigvec = self.eigval, self.eigvec
        _device = eigval.device
        indices = indices if indices is not None else torch_arange(eigval.size(0), dtype=int, device=_device)
        N = indices.size(0)
        # (N, 3)
        eigval_order = eigval.argsort(dim=-1, descending=True)
        # (N, 3, 3)
        eigvec_T = eigvec.transpose(dim0=1, dim1=2)
        # (N, 3, 3)
        R = eigvec_T[
            torch_arange(N, dtype=int, device=_device)[:, None, None],
            eigval_order[..., None],
            torch_arange(3, dtype=int, device=_device)[None, None]
        ]
        R[(R[:, 0, :] * n).sum(dim=1) < 0] *= -1
        R[torch_linalg_det(R) < 0, 2] *= -1
        # (N, 3, 3)
        R_inv = R.transpose_(dim0=1, dim1=2)
        return R_inv


class Decompositionor():

    def __init__(self, graph: tg_data_Data):
        GeneralUtils.validateAttributes(graph, ["pos"])
        self.graph = graph

    def getMDTransformation(self, selection: Selection, _n: torch_Tensor, _mass: torch_Tensor):
        SIGMA_1 = 3
        _graph = self.graph
        i, j = selection.getEdgeIndex()
        bi = selection.getBatchIndex()

        _pos = _graph.pos
        vi = _pos[i]
        vj = _pos[j]
        dv = vj - vi
        scale_factors = 1.0 / selection.scatter(dv.norm(dim=1), "max")[0]
        scale_factors_i = scale_factors[bi]
        dv_scaled = dv * scale_factors_i[:, None]
        nj = _n[j]
        # (P, 3)
        wj = wj = torch_nn_functional_normalize(torch_cross(torch_cross(dv_scaled, nj, dim=1), dv_scaled, dim=1), p=2, dim=1)
        # (P, 3)
        njprime = 2 * (nj * wj).sum(dim=1)[:, None] * wj - nj
        # Big problem. Pointclouds don't have areas. What do we do about this?!?!
        # Reasoning: If the area of the normal vector is small, it doesn't influence the final voting tensor.
        # Should we use neighbourhood density to have a scaling factor or not?
        # (P,)
        areas = _mass[j] * scale_factors_i ** 2
        # (P,)
        maxArea = selection.scatter(areas, "max")[0][bi]
        # (P,)
        ddcs = dv_scaled.norm(dim=1)
        # (P,)
        muj = (areas / maxArea)*(-ddcs*SIGMA_1).exp_()
        Tj = muj[:, None, None] * njprime[:, None] * njprime[..., None]
        # (I, 3, 3)
        Ti = selection.scatter(Tj, "add")
        # ((I, 3), (I, 3, 3))
        eigh = torch_linalg_eigh(Ti)
        return Decomposition(*eigh), scale_factors
    
    def getNormalFilteredPVT(self, selection: Selection, _n: torch_Tensor, rho: float = 0.9):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        _g = self.graph
        bi, j = selection.getEdgeIndex()
        _v = _g.pos
        _dtype_ints = j.dtype

        vj = _v[j]
        ni = _n[bi]
        nj = _n[j]
        wij = ((ni * nj).sum(dim=1, keepdim=True).clamp_(-1, 1).acos_() <= rho).to(_dtype_ints)
        summed_wij = selection.scatter(wij, "add")
        # Make all weights 1 if all weights are zero
        err_mask = summed_wij == 0
        err_neighbors = bi[None] == err_mask.flatten().nonzero().flatten()[:, None]
        wij[err_neighbors.sum(dim=0).nonzero().flatten()] = 1
        summed_wij[err_mask] = err_neighbors.sum(dim=1)
        # Continue normal pipeline
        wvij = wij * vj
        summed_wvij = selection.scatter(wvij, "add")
        v_center = summed_wvij / summed_wij
        dv = vj - v_center[bi]
        T = wij[..., None] * dv[:, None] * dv[..., None]
        summed_wdv = selection.scatter(T, "add")
        Ci = summed_wdv / summed_wij[..., None]
        err = (summed_wij == 0).flatten().nonzero().flatten()
        if err.size(0) > 0:
            err_n = _n[err]
            err_v = _v[err]
            s1 = torch_cross(err_n, err_v)[:, None]
            s2 = torch_cross(err_n, s1[:, 0])[:, None]
            samples = torch_cat([s1, -s1, s2, -s2], dim=1)
            Ci[err] = (samples[:, :, None] * samples[..., None]).sum(dim=1)
            # print(f"{err} indices corrected for NormalFilteredPVT method")
        eigh = torch_linalg_eigh(Ci)
        return Decomposition(*eigh)
    
    def getBetterFilteredPVT(self, selection: Selection, _n: torch_Tensor, rho: float = 0.9):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        _g = self.graph
        bi, j = selection.getEdgeIndex()
        _v = _g.pos
        _dtype_ints = j.dtype

        vj = _v[j]
        dv = vj - _v[bi]
        nj = _n[j]
        wij = ((nj * torch_nn_functional_normalize(dv, dim=1)).sum(dim=1, keepdim=True).clamp_(-1, 1).abs_().acos_() > rho).to(_dtype_ints)
        wvij = wij * vj
        summed_wij = selection.scatter(wij, "add")
        summed_wvij = selection.scatter(wvij, "add")
        v_center = summed_wvij / summed_wij
        dv = vj - v_center[bi]
        T = wij[..., None] * dv[:, None] * dv[..., None]
        summed_wdv = selection.scatter(T, "add")
        Ci = summed_wdv / summed_wij[..., None]
        eigh = torch_linalg_eigh(Ci)
        return Decomposition(*eigh)
    
    def getPVT(self, selection: Selection) -> torch_Tensor:
        bi, j = selection.getEdgeIndex()
        vj = self.graph.pos[j]
        v_center = selection.scatter(vj, "mean")
        dvjc = vj - v_center[bi]
        dvjc_o = dvjc[:, None] * dvjc[..., None]
        vt_dvjc =  selection.scatter(dvjc_o, "add")
        return Decomposition(*torch_linalg_eigh(vt_dvjc))

    def getNVT(self, selection: Selection, _n: torch_Tensor, rho: float = 0.9):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        i, j = selection.getEdgeIndex()
        nj = _n[j]
        Tj = nj[:, None] * nj[..., None]
        Ti = selection.scatter(Tj, "mean")
        return Decomposition(*torch_linalg_eigh(Ti))
    
    def getNormalFilteredNVT(self, selection: Selection, _n: torch_Tensor, rho: float = 0.9):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        i, j = selection.getEdgeIndex()
        ni = _n[i]
        nj = _n[j]
        wij = ((ni * nj).sum(dim=1, keepdim=True).clamp_(-1, 1).acos_() <= rho).to(j.dtype)
        summed_wij = selection.scatter(wij, "add")
        Tj = wij[..., None] * nj[:, None] * nj[..., None]
        Ti = selection.scatter(Tj, "add") / summed_wij[..., None]
        err = (summed_wij == 0).flatten().nonzero().flatten()
        err_n = _n[err]
        Ti[err] = err_n[:, None] * err_n[..., None]
        return Decomposition(*torch_linalg_eigh(Ti))

    def getBetterFilteredNVT(self, selection: Selection, _n: torch_Tensor, rho: float = 0.9):
        '''
            rho is the local binary neighbourhood threshold.
            tau is proportional to the applied noise.
            d is the dampening factor.
        '''
        _v = self.graph.pos
        i, j = selection.getEdgeIndex()
        vi = _v[i]
        vj = _v[j]
        dv = vj - vi
        nj = _n[j]
        wij = ((torch_nn_functional_normalize(dv, dim=1) * nj).sum(dim=1, keepdim=True).clamp_(-1, 1).abs_().acos_() > rho).to(j.dtype)
        summed_wij = selection.scatter(wij, "add")
        # Make all weights 1 if all weights are zero
        err_mask = summed_wij == 0
        err_neighbors = i[None] == err_mask.flatten().nonzero().flatten()[:, None]
        wij[err_neighbors.sum(dim=0).nonzero().flatten()] = 1
        summed_wij[err_mask] = err_neighbors.sum(dim=1)
        # Continue normal pipeline
        Tj = wij[..., None] * nj[:, None] * nj[..., None]
        Ti = selection.scatter(Tj, "add") / summed_wij[..., None]
        return Decomposition(*torch_linalg_eigh(Ti))