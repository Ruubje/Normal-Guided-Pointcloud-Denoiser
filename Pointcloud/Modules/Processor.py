from .Denoiser import Denoiser
from .Decompositionor import Decompositionor
from .GraphBuilder import GraphBuilder
from .Noise import Noise
from .Object import Pointcloud
from .Selector import Selector
from .Utils import GeneralUtils, SlicedTorchData, TorchUtils

from torch import (
    cat as torch_cat,
    pi as torch_pi,
    Tensor as torch_Tensor
)
from torch_geometric.data import (
    Data as tg_data_Data
)
from torch_geometric.utils import (
    degree as tg_utils_degree,
    subgraph as tg_utils_subgraph
)
from tqdm import tqdm
from typing import Callable as typing_Callable

class Processor:
    def __init__(self, pointcloud: Pointcloud):
        self.pointcloud = pointcloud
        self.graphBuilder = GraphBuilder(pointcloud)
        _graph = self.graphBuilder.graph
        self.graph = _graph
        self.selector = Selector(_graph)
        self.noise = Noise(_graph)
        self.denoiser = Denoiser(_graph)
        self.decompositionor = Decompositionor(_graph)

    def getMDFeatures(self) -> torch_Tensor:
        selection = self.selector.getMDSelection(indices=None)
        decomposition, scale_factors = self.decompositionor.getMDTransformation(selection, self.graph.n, self.graph.mass)
        return decomposition.getMDFeatures()
    
    def getMDPatches(self, indices: torch_Tensor = None):
        _alignor = self.alignor

        N = len(indices) if indices is not None else self.graph.num_nodes
        selection = self.selector.getMDSelection(indices=indices)
        _, sf, eigh = _alignor.getMDTransformation(selection)
        self.RInv = _alignor.getRInv(eigh, indices)
        self.graph_list = [self.getMDPatch(x if indices is None else indices[x], x, selection, self.RInv, sf) for x in tqdm(range(N), desc="Creating patches in list")]
        return self.graph_list, self.RInv
    
    def getMDPatch(self, global_index: int, index: int, selection: SlicedTorchData, R_inv: torch_Tensor, scale_factors: torch_Tensor) -> tg_data_Data:
        GeneralUtils.validateAttributes(self, ["graph"])
        # (Pi,)
        p = selection[index]
        scale_factor = scale_factors[index]
        _g = self.graph
        _edge_index = _g.edge_index
        # (Pi, 3)
        nj = _g.n[p]
        # (3, 3)
        R_inv_i = R_inv[index]
        # (Pi, 3)
        nj_R_inv = nj @ R_inv_i
        # (3,)
        gt = _g.gt_n[global_index] if hasattr(_g, "gt_n") else _g.n[global_index]
        # (Pi, 3)
        _pos = _g.pos[p]
        # (Pi, 3)
        c = (_pos - _pos.mean(dim=0)) * scale_factor @ R_inv_i
        # (Pi, 3)
        n = nj_R_inv
        # (Pi, 1)
        a = (_g.mass[p] * scale_factor)[:, None]
        # (Pi, 1)
        d = tg_utils_degree(_edge_index[0])[p, None]
        # (Pi, 8)
        x = torch_cat((c, n, a, d), dim=1)
        # (2, E)
        edge_index = tg_utils_subgraph(p, _edge_index, relabel_nodes=True)[0]
        # (1, 3)
        y = gt[None] @ R_inv_i
        return tg_data_Data(x=x, edge_index=edge_index, y=y)
    
    def getVUDecomposition(self):
        _graph = self.graph
        _graph.edge_index = self.graphBuilder.getKNNEdgeIndex(6)
        _pos = _graph.pos
        _edge_index = _graph.edge_index
        _selector = self.selector
        _decompositionor = self.decompositionor
        mean_graph_edge_length = TorchUtils.averageEdgeLength(_pos, _edge_index)
        r = 2 * mean_graph_edge_length
        selection = _selector.getPointsInRangeSelection(r)
        decompositionNVT = _decompositionor.getNormalFilteredNVT(selection, _graph.n, rho=0.95)
        filtered_normals = decompositionNVT.getVUSmoothedNormals(_graph.n, tau=0.3, d=3)
        decompositionPVT = _decompositionor.getNormalFilteredPVT(
            selection,
            filtered_normals,
            rho=0.95
        )
        return decompositionPVT
    
    def getMartinFeatureDecomposition(self, r: float, rho: float = 0.9):
        _n = self.graph.n
        selection = self.selector.getPointsInRangeSelection(r)
        nvt = self.decompositionor.getNormalFilteredNVT(selection, _n, rho)
        filtered_normals = nvt.getVUSmoothedNormals(_n)
        decomposition = self.decompositionor.getNormalFilteredPVT(selection, filtered_normals, rho)
        return decomposition, filtered_normals
    
    def getMyFeatureDecomposition(self, N: int = 2 ** 4, angle: float = None):
        angle = angle if angle is not None else torch_pi * 5 / 12
        n = self.graph.n
        selection = self.selector.getKNNSelection(N)
        nvt = self.decompositionor.getBetterFilteredNVT(selection, n, angle)
        filtered_normals = nvt.getVUSmoothedNormals(n)
        decomposition = self.decompositionor.getBetterFilteredNVT(selection, filtered_normals, angle)
        return decomposition, filtered_normals

    def denoise(self):
        l = TorchUtils.averageEdgeLength(self.graph.pos, self.selector.getKNNSelection(6).getEdgeIndex())
        d = 2 * l
        alphas = [1, 0.2, 1]
        for _ in range(2):
            decomposition, f_n = self.getMyFeatureDecomposition()
            classes = decomposition.getClasses()
            selection = self.selector.getKNNSelection(8)
            for key in range(3):
                indices = (classes == key).nonzero().flatten()
                if indices.size(0) == 0:
                    continue
                elif key == 0:
                    new_pos = self.denoiser.flat_step(selection.filter(indices), f_n, d, alphas[key])
                elif key == 1:
                    edge_vectors = decomposition.eigvec[..., 0]
                    new_pos = self.denoiser.edge_step(selection.filter(indices), f_n, edge_vectors, d, alphas[key])
                else:
                    new_pos = self.denoiser.feature_step(selection.filter(indices), f_n, d, alphas[key])
                self.graph.pos[indices] = new_pos
            self.graph.n = f_n

    def denoiseUntilMinimumError(self, gt_pos: torch_Tensor, strategy: dict, k: int = 7, alpha: list = [0.02, 0.02, 0.1], d: float = 200, error_funcs: list[typing_Callable] = [TorchUtils.PaperDistance]):
        _graph = self.graph
        _denoiser = self.denoiser
        _selector = self.selector

        noisy_graph_pos = _graph.pos.clone()
        noisy_graph_n = _graph.n.clone()

        # Do iterations
        i = 0
        previous_pos = noisy_graph_pos
        current_pos = previous_pos
        previous_error = [error_func(gt_pos, _graph.pos) + 200 for error_func in error_funcs]
        current_error = [error_func(gt_pos, _graph.pos) for error_func in error_funcs]
        # Initialize tqdm without a total
        pbar = tqdm(desc="Vertex updating")
        while current_error[0].mean(dim=0) < previous_error[0].mean(dim=0):
            decomposition, f_n = self.getMyFeatureDecomposition()
            edge_vectors = decomposition.eigvec[..., 0]
            classes = decomposition.getClasses()
            selection = _selector.getKNNSelection(k)
            for key, func in strategy.items():
                indices = (classes == key).nonzero().squeeze_()
                if indices.size(0) == 0:
                    continue
                if func == _denoiser.edge_step:
                    new_pos = func(selection.filter(indices), f_n, edge_vectors, d, alpha[key])
                else:
                    new_pos = func(selection.filter(indices), f_n, d, alpha[key])
                _graph.pos[indices] = new_pos
            error = [error_func(gt_pos, _graph.pos) for error_func in error_funcs]
            previous_error = current_error
            current_error = error
            previous_pos = current_pos
            current_pos = _graph.pos
            self.graph.n = f_n
            i += 1
            pbar.update(1)
            pbar.set_postfix({"Comparing Error": float(error[0].mean(dim=0))})

        print(f"Stopped cause new error was {current_error[0].mean(dim=0):.2E} compaired to previous error {previous_error[0].mean(dim=0):.2E}")
        pbar.close()
        _graph.pos = noisy_graph_pos
        _graph.n = noisy_graph_n
        return previous_pos, previous_error, i - 1
    
    def preprocessPointcloud(self, k: int = 12, noise_level: float = 0.3):
        _graph = self.graph
        _graphBuilder = self.graphBuilder

        _graph.edge_index = _graphBuilder.getKNNEdgeIndex(k)
        _graphBuilder.setAndFlipNormals(flip=False)

        _pos = _graph.pos
        _edge_index = _graph.edge_index

        l = TorchUtils.averageEdgeLength(_pos, _edge_index)
        self.noise.generateNoise(noise_level, l, keepNormals=False)
        _graphBuilder.setAndFlipNormals(flip=True)