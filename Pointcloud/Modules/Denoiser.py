from .Utils import SlicedTorchData, TorchUtils
from .Selector import Selection

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

    def corner_step(self, selection: Selection, n: torch_Tensor, d: float, alpha: float = 0.1):
        _graph = self.graph
        _pos = _graph.pos
        _pos_i = _pos[selection.i]

        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        i, j = selection.getEdgeIndex()

        vj = _pos[j]
        nj = n[j]
        nj_outer_nj = nj[..., None] * nj[:, None, :]
        summed_outer = selection.scatter(nj_outer_nj, "add")
        nj_nj_vj = torch_einsum("nij,nj->ni", nj_outer_nj, vj)
        summed_njnjvj = selection.scatter(nj_nj_vj, "add")
        inv_summed_outer, info = torch_linalg_inv_ex(summed_outer)
        mask_inversable = info == 0
        tics = _pos_i.clone()
        tics[mask_inversable] = torch_einsum("nij,nj->ni", inv_summed_outer[mask_inversable], summed_njnjvj[mask_inversable])
        di = (tics - _pos_i) * alpha
        mask_threshold = di.norm(dim=1) < d
        tics[~mask_threshold] = _pos_i[~mask_threshold]
        tics[mask_threshold] = _pos_i[mask_threshold] + di[mask_threshold]
        return tics

    def edge_step(self, selection: Selection, n: torch_Tensor, edge_vectors: torch_Tensor, d: float, alpha: float = 0.1):
        _graph = self.graph
        _pos = _graph.pos
        _pos_i = _pos[selection.i]

        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        i, j = selection.getEdgeIndex()

        yi1 = edge_vectors[i]
        vi = _pos[i]
        vj = _pos[j]
        nj = n[j]

        vj_pi = vj - ((vj - vi)*yi1).sum(dim=1, keepdim=True)*yi1
        nj_pi = nj - (nj*yi1).sum(dim=1, keepdim=True)*yi1

        nj_X_nj = nj_pi[..., None] * nj_pi[:, None]
        yi1_X_yi1 = yi1[..., None] * yi1[:, None]
        scatter_input = nj_X_nj + yi1_X_yi1
        normalization = selection.scatter(scatter_input, "add")
        nj_X_nj_vj = torch_einsum("nij,nj->ni", nj_X_nj, vj_pi)
        yi1_X_yi1_vi = torch_einsum("nij,nj->ni", yi1_X_yi1, vi)
        scatter_input = nj_X_nj_vj + yi1_X_yi1_vi
        minimization = selection.scatter(scatter_input, "add")
        inv_normalization, info = torch_linalg_inv_ex(normalization)
        mask_inversable = info == 0
        tics = _pos_i.clone()
        tics[mask_inversable] = torch_einsum("nij,nj->ni", inv_normalization[mask_inversable], minimization[mask_inversable])
        di = (tics - _pos_i) * alpha
        mask_threshold = di.norm(dim=1) < d
        tics[~mask_threshold] = _pos_i[~mask_threshold]
        tics[mask_threshold] = _pos_i[mask_threshold] + di[mask_threshold]
        return tics

    def flat_step(self, selection: Selection, n: torch_Tensor, d: float, alpha: float = 0.1):
        _pos = self.graph.pos
        _pos_i = _pos[selection.i]

        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        i, j = selection.getEdgeIndex()

        vi = _pos[i]
        vj = _pos[j]
        ni = n[i]
        nj = n[j]
        dist = vj - vi

        center = vj.mean(dim=0)
        delta = (vj - center).norm(dim=1).max(dim=0).values
        similarity = (-16 * (ni - nj).square().sum(dim=1) / delta ** 2).exp()
        # similarity = (-16 * (1 - (ni * nj).sum(dim=1).square()) / delta ** 2).exp()
        closeness = (-4 * dist.square().sum(dim=1) / delta ** 2).exp()
        Wij = similarity * closeness
        dot = (nj * dist).sum(dim=1)
        src = Wij[:, None] * dot[:, None] * ni
        summed = selection.scatter(src, "add")
        Wij_summed = selection.scatter(Wij, "add")
        di = summed / Wij_summed[:, None] * alpha
        mask_threshold = di.norm(dim=1) <= d
        di[~mask_threshold, :] = 0
        return _pos_i + di

    def new_step(self, selection: Selection, n: torch_Tensor, d: float, alpha: float = 0.1):
        _graph = self.graph
        _pos = _graph.pos
        _device = _pos.device

        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        indices = selection.i
        i, j = selection.getEdgeIndex()
        _slices = selection.slices

        vi = _pos[indices]
        vj = _pos[j]
        n_o = n[:, None] * n[..., None]

        delta_j = (vj - vj.mean(dim=0)).norm(dim=1).max(dim=0).values
        closeness = (-4 * (vj - _pos[i]).square().sum(dim=1) / delta_j ** 2).exp()
        similarity = (-16 * (n[j] - n[i]).square().sum(dim=1) / 4).exp()
        likeliness = (-9 * (n[j] * (vj - _pos[i])).sum(dim=1).square() / delta_j ** 2).exp()
        wij = likeliness

        I = torch_eye(3, dtype=torch_float, device=_device)
        ni_o = n_o[indices]
        nj_o = n_o[j]
        summed_nj_o = selection.scatter(wij[:, None, None] * nj_o, "add")
        cardinality = _slices[1:] - _slices[:-1]
        nj_o_vj = torch_einsum("nij,nj->ni", nj_o, vj)
        summed_nj_o_vj = selection.scatter(wij[:, None] * nj_o_vj, "add")
        n_o_v = torch_einsum("nij,nj->ni", ni_o, vi)
        summed_vj = selection.scatter(wij[:, None] * vj, "add")
        n_o_summed_vj = torch_einsum("nij,nj->ni", ni_o, summed_vj)
        w0, w1, w2 = 1, 1, 1
        A0 = I[None] + ni_o
        A1 = summed_nj_o
        A2 = cardinality[:, None, None] * ni_o
        A = w0*A0 + w1*A1 + w2*A2
        b0 = vi + n_o_v
        b1 = n_o_summed_vj
        b2 = summed_nj_o_vj
        b = w0*b0 + w1*b1 + w2*b2
        A_inv, info = torch_linalg_inv_ex(A)
        mask_inversable = info == 0
        optimal_pos = vi.clone()
        new_pos = torch_einsum("nij,nj->ni", A_inv, b)
        optimal_pos[mask_inversable] = optimal_pos[mask_inversable] + (new_pos[mask_inversable] - optimal_pos[mask_inversable])
        di = (optimal_pos - vi) * alpha
        mask_threshold = di.norm(dim=1) < d
        optimal_pos[~mask_threshold] = vi[~mask_threshold]
        optimal_pos[mask_threshold] = vi[mask_threshold] + di[mask_threshold]
        return optimal_pos

    def feature_step(self, selection: Selection, n: torch_Tensor, d: float, alpha: float = 0.1):
        _graph = self.graph
        _pos = _graph.pos
        _device = _pos.device

        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        indices = selection.i
        i, j = selection.getEdgeIndex()
        _slices = selection.slices

        vi = _pos[indices]
        vj = _pos[j]
        n_o = n[:, None] * n[..., None]
        
        I = torch_eye(3, dtype=torch_float, device=_device)
        ni_o = n_o[indices]
        nj_o = n_o[j]
        summed_nj_o = selection.scatter(nj_o, "add")
        cardinality = _slices[1:] - _slices[:-1]
        nj_o_vj = torch_einsum("nij,nj->ni", nj_o, vj)
        summed_nj_o_vj = selection.scatter(nj_o_vj, "add")
        n_o_v = torch_einsum("nij,nj->ni", ni_o, vi)
        summed_vj = selection.scatter(vj, "add")
        n_o_summed_vj = torch_einsum("nij,nj->ni", ni_o, summed_vj)
        w0, w1, w2 = 1, 1, 1
        A0 = I[None] + ni_o
        A1 = summed_nj_o
        A2 = cardinality[:, None, None] * ni_o
        A = w0*A0 + w1*A1 + w2*A2
        b0 = vi + n_o_v
        b1 = n_o_summed_vj
        b2 = summed_nj_o_vj
        b = w0*b0 + w1*b1 + w2*b2
        A_inv, info = torch_linalg_inv_ex(A)
        mask_inversable = info == 0
        optimal_pos = vi.clone()
        new_pos = torch_einsum("nij,nj->ni", A_inv, b)
        optimal_pos[mask_inversable] = optimal_pos[mask_inversable] + (new_pos[mask_inversable] - optimal_pos[mask_inversable])
        di = (optimal_pos - vi) * alpha
        mask_threshold = di.norm(dim=1) < d
        optimal_pos[~mask_threshold] = vi[~mask_threshold]
        optimal_pos[mask_threshold] = vi[mask_threshold] + di[mask_threshold]
        return optimal_pos
    
    def dummy_step(self, selection: Selection, n: torch_Tensor, d: float, alpha: float = 0.1):
        _graph = self.graph
        _pos = _graph.pos

        assert n.dim() == 2
        assert _pos.size(0) == n.size(0)
        assert n.size(1) == 3

        indices = selection.i

        vi = _pos[indices]
        return vi.clone()