from .Utils import SlicedTorchData, TorchUtils

from torch import (
    einsum as torch_einsum,
    eye as torch_eye,
    float as torch_float,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch.linalg import (
    inv_ex as torch_linalg_inv_ex
)
from torch_geometric.data import (
    Data as tg_data_Data
)

class Denoiser:
    def __init__(self, graph: tg_data_Data):
        assert hasattr(graph, "pos") and graph.pos is not None
        assert graph.pos.dim() == 2
        assert graph.pos.size(1) == 3

        self.graph = graph

    def corner_step(self, indices: torch_Tensor, neighbourhood: SlicedTorchData, n: torch_Tensor, d: float):
        _graph = self.graph
        _pos = _graph.pos
        _pos_i = _pos[indices]

        TorchUtils.validateIndices(indices)
        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        i, j = neighbourhood._batchIndices(indices)
        _device = j.device
        N = indices.size(0)

        vj = _pos[j]
        nj = n[j]
        nj_outer_nj = nj[..., None] * nj[:, None, :]
        summed_outer = torch_zeros(
            (N, 3, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=nj_outer_nj.to(torch_float)
        )
        nj_nj_vj = torch_einsum("nij,nj->ni", nj_outer_nj, vj)
        summed_njnjvj = torch_zeros(
            (N, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=nj_nj_vj.to(torch_float)
        )
        inv_summed_outer, info = torch_linalg_inv_ex(summed_outer)
        mask_inversable = info == 0
        tics = _pos_i.clone()
        tics[mask_inversable] = torch_einsum("nij,nj->ni", inv_summed_outer[mask_inversable], summed_njnjvj[mask_inversable])
        mask_threshold = (tics - _pos_i).norm(dim=1) < d
        tics[~mask_threshold] = _pos_i[~mask_threshold]
        return tics

    def edge_step(self, indices: torch_Tensor, neighbourhood: SlicedTorchData, n: torch_Tensor, edge_vectors: torch_Tensor, d: float):
        _graph = self.graph
        _pos = _graph.pos
        _pos_i = _pos[indices]

        TorchUtils.validateIndices(indices)
        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        i, j = neighbourhood._batchIndices(indices)
        _device = j.device
        N = indices.size(0)

        yi1 = edge_vectors[indices[i]]
        vi = _pos[indices[i]]
        vj = _pos[j]
        nj = n[j]

        vj_pi = vj - ((vj - vi)*yi1).sum(dim=1, keepdim=True)*yi1
        nj_pi = nj - (nj*yi1).sum(dim=1, keepdim=True)*yi1

        nj_X_nj = nj_pi[..., None] * nj_pi[:, None]
        yi1_X_yi1 = yi1[..., None] * yi1[:, None]
        scatter_input = nj_X_nj + yi1_X_yi1
        normalization = torch_zeros(
            (N, 3, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=scatter_input.to(torch_float)
        )
        nj_X_nj_vj = torch_einsum("nij,nj->ni", nj_X_nj, vj_pi)
        yi1_X_yi1_vi = torch_einsum("nij,nj->ni", yi1_X_yi1, vi)
        scatter_input = nj_X_nj_vj + yi1_X_yi1_vi
        minimization = torch_zeros(
            (N, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=scatter_input.to(torch_float)
        )
        inv_normalization, info = torch_linalg_inv_ex(normalization)
        mask_inversable = info == 0
        tics = _pos_i.clone()
        tics[mask_inversable] = torch_einsum("nij,nj->ni", inv_normalization[mask_inversable], minimization[mask_inversable])
        mask_threshold = (tics - _pos_i).norm(dim=1) < d
        tics[~mask_threshold] = _pos_i[~mask_threshold]
        return tics

    def flat_step(self, indices: torch_Tensor, neighbourhood: SlicedTorchData, n: torch_Tensor, d: float, alpha: float = 0.1):
        _pos = self.graph.pos
        _pos_i = _pos[indices]

        TorchUtils.validateIndices(indices)
        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        i, j = neighbourhood._batchIndices(indices)
        _device = j.device
        N = indices.size(0)

        vi = _pos[indices[i]]
        vj = _pos[j]
        ni = n[indices[i]]
        nj = n[j]
        dist = vj - vi

        center = vj.mean(dim=0)
        delta = (vj - center).norm(dim=1).max(dim=0).values
        # similarity = (-16 * (ni - nj).square().sum(dim=1) / delta ** 2).exp()
        similarity = (-16 * (1 - (ni * nj).sum(dim=1).square()) / delta ** 2).exp()
        closeness = (-4 * dist.square().sum(dim=1) / delta ** 2).exp()
        Wij = similarity * closeness
        dot = (nj * dist).sum(dim=1)
        src = Wij[:, None] * dot[:, None] * ni
        summed = torch_zeros(
            (N, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=src
        )
        Wij_summed = torch_zeros(
            (N,),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=Wij
        )
        di = summed / Wij_summed[:, None] * alpha
        mask_threshold = di.norm(dim=1) <= d
        di[~mask_threshold, :] = 0
        return _pos_i + di

    def feature_step(self, indices: torch_Tensor, neighbourhood: SlicedTorchData, n: torch_Tensor):
        _graph = self.graph
        _pos = _graph.pos
        _pos_i = _pos[indices]

        TorchUtils.validateIndices(indices)
        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        _slices = neighbourhood.slices
        i, j = neighbourhood._batchIndices(indices)
        _device = j.device
        N = indices.size(0)

        vj = _pos[j]
        n_o = n[:, None] * n[..., None]
        
        I = torch_eye(3, dtype=torch_float, device=_device)
        ni_o = n_o[indices]
        nj_o = n_o[j]
        summed_nj_o = torch_zeros(
            (N, 3, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=nj_o.to(torch_float)
        )
        cardinality = (_slices[1:] - _slices[:-1])[indices]
        nj_o_vj = torch_einsum("nij,nj->ni", nj_o, vj)
        summed_nj_o_vj = torch_zeros(
            (N, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=nj_o_vj.to(torch_float)
        )
        n_o_v = torch_einsum("nij,nj->ni", ni_o, _pos_i)
        summed_vj = torch_zeros(
            (N, 3),
            dtype=torch_float,
            device=_device
        ).index_add_(
            dim=0,
            index=i,
            source=vj.to(torch_float)
        )
        n_o_summed_vj = torch_einsum("nij,nj->ni", ni_o, summed_vj)
        w0, w1, w2 = 1.5, 1, 1
        A0 = I[None] + ni_o
        A1 = summed_nj_o
        A2 = cardinality[:, None, None] * ni_o
        A = w0*A0 + w1*A1 + w2*A2
        b0 = _pos_i + n_o_v
        b1 = n_o_summed_vj
        b2 = summed_nj_o_vj
        b = w0*b0 + w1*b1 + w2*b2
        A_inv, info = torch_linalg_inv_ex(A)
        mask_inversable = info == 0
        optimal_pos = _pos_i.clone()
        new_pos = torch_einsum("nij,nj->ni", A_inv, b)
        optimal_pos[mask_inversable] = optimal_pos[mask_inversable] + (new_pos[mask_inversable] - optimal_pos[mask_inversable])
        return optimal_pos, mask_inversable