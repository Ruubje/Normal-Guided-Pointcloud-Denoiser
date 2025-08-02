from .Object import Pointcloud
from .Selector import Selection
from .Utils import GeneralUtils, TorchUtils, SlicedTorchData

# from deltaconv.geometry import (
#     estimate_basis as dc_geo_estimate_basis
# )
from math import (
    cos as math_cos,
    pi as math_pi
)
from numpy import (
    vstack as np_vstack,
)
from robust_laplacian import point_cloud_laplacian as robust_pointcloud_laplacian
from torch import (
    arange as torch_arange,
    argmax as torch_argmax,
    argsort as torch_argsort,
    float as torch_float,
    from_numpy as torch_from_numpy,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch.linalg import (
    eigh as torch_linalg_eigh
)
from torch_cluster import (
    knn_graph as tc_knn_graph
)
from torch_scatter import(
    scatter_mean as torch_scatter_mean
)
from torch_geometric.data import Data as tg_data_Data
from torch_geometric.utils import (
    to_undirected as tg_utils_to_undirected
)
from tqdm import tqdm

class GraphBuilder:
    r"""
    Creates a graph from a pointcloud.
    This step is done before Patch selection, alignment and input generation.
    """

    def __init__(self, pointcloud: Pointcloud):
        GeneralUtils.validateAttributes(pointcloud, ["v"])
        self.device = pointcloud.v.device
        self.pointcloud = pointcloud
        self.graph = tg_data_Data(pos=pointcloud.v)
        if pointcloud.hasNormals():
            self.graph.n = pointcloud.n

    def setTriangleGraphWithFlippedNormals(self) -> tg_data_Data:
        _graph = self.graph
        _graph.edge_index, _graph.mass = self.getLaplacianEdgeIndex()
        self.setAndFlipNormals()
        return self.graph
    
    def getKNNEdgeIndex(self, k: int = 12) -> None:
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ["pos"])
        return tc_knn_graph(_graph.pos, k, flow="target_to_source")

    def getLaplacianEdgeIndex(self) -> None:
        _device = self.device
        L, M = robust_pointcloud_laplacian(self.pointcloud.v.cpu().numpy())
        Lcoo = L.tocoo()
        return torch_from_numpy(np_vstack((Lcoo.row, Lcoo.col))).long().to(_device),\
                                                    torch_from_numpy(M.data).float().to(_device)

    def getRobustLaplacianFaces(self) -> torch_Tensor:
        edge_index = self.getLaplacianEdgeIndex()[0]
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        return TorchUtils.edge_to_faces(edge_index)

    def setAndFlipNormals(self, flip: bool = True) -> None:
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ["edge_index"])
        self.setPVTNormals(_graph.edge_index)
        if flip:
            self.flipNormals()
    
    def getDeltaconvCoordinates(self, knn_edge_index: torch_Tensor = None, k: int = 12) -> None:
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ['pos'])
        if knn_edge_index is None:
            knn_edge_index = tc_knn_graph(_graph.pos, k=k)
        return dc_geo_estimate_basis(_graph.pos, knn_edge_index, k=k)

    def setNormalsDeltaconv(self, knn_edge_index: torch_Tensor = None, k: int = 12) -> None:
        DeltaConvCoordinates = self.getDeltaconvCoordinates(knn_edge_index, k)
        self.graph.n = DeltaConvCoordinates[0]

    def setPVTNormals(self, edge_index: torch_Tensor) -> None:
        eigvec = self.getPVTDecompositionWithKNN(edge_index)
        self.graph.n = eigvec[..., 0]

    def getPVTDecompositionWithKNN(self, edge_index: torch_Tensor) -> None:
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ['pos'])
        k = TorchUtils.validateKNNEdgeIndex(edge_index)
        _pos = _graph.pos
        
        i, j = edge_index[0].view(-1, k), edge_index[1].view(-1, k)
        vj = _pos[j]
        v_center = vj.mean(dim=1)
        dvjc = vj - v_center[i]
        vt_dvjc = (dvjc[:, :, None] * dvjc[..., None]).sum(dim=1)
        _, eigvec = torch_linalg_eigh(vt_dvjc)
        return eigvec
    
    def getPVTDecomposition(self, selection: Selection) -> None:
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ['pos'])
        _pos = _graph.pos
        N = _graph.num_nodes
        
        i, j = selection.getEdgeIndex()
        vj = _pos[j]
        v_center = torch_scatter_mean(src=vj, index=i, dim=0)
        dvjc = vj - v_center[i]
        dvjc_o = dvjc[:, None] * dvjc[..., None]
        vt_dvjc =  torch_zeros((N, 3, 3), dtype=dvjc_o.dtype, device=dvjc_o.device) \
            .index_add_(dim=0, index=i, source=dvjc_o)
        _, eigvec = torch_linalg_eigh(vt_dvjc)
        return eigvec
    
    def flipNormals(self) -> None:
        self.calculateEdgeCost()
        mst, _ = self.calculateUndirectedMST()
        self.flipNormalsWithMST(mst)

    def calculateEdgeCost(self) -> None:
        r"""
        Computes the cost of the edges based on the edge_index and normal vectors.
        Cost function is 1 - n_i * n_j.

        Args:
            graph (Torch Geometric Data): Graph on which to compute cost.
        """
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ["edge_index", "n"])
        normals = _graph.n[_graph.edge_index]
        _graph.edge_attr = 1 - (normals[0] * normals[1]).sum(dim=-1).abs_()

    def calculateUndirectedMST(self) -> torch_Tensor:
        r"""
        Computes and returns the indices of the edges that point to the edges
        that are contained in the Minimal Spanning Tree (MST).

        Args:
            graph (Torch Geometric Data): Graph on which to build MST.
        """
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ['pos', 'edge_index', 'edge_attr'])

        _device = _graph.pos.device
        N = _graph.num_nodes
        E = _graph.num_edges
        _edge_index = _graph.edge_index
        _edge_attr = _graph.edge_attr

        sorted_cost = torch_argsort(_edge_attr, dim=0, descending=False)
        included_edges = torch_zeros(E, dtype=bool, device=_device)
        groups = torch_arange(N, device=_device)
        for index in tqdm(sorted_cost, desc="Creating MST.."):
            _edge = _edge_index[:, index]
            if groups[_edge[0]] != groups[_edge[1]]:
                included_edges[index] = True
                _edge_groups = groups[_edge]
                groups[groups == _edge_groups[0]] = _edge_groups[1]
        edge_indices = included_edges.nonzero().squeeze()
        return tg_utils_to_undirected(edge_index=_edge_index[:, edge_indices], edge_attr=_edge_attr[edge_indices], num_nodes=N)

    def flipNormalsWithMST(self, mst_edge_index: torch_Tensor) -> None:
        r"""
        Flips normals in the graph based on the MST.

        Args:
            graph (Torch Geometric Data): Graph that contains normals that need to be flipped.
            mst (Torch Tensor (E,)): Indices that point to the edges that are included in the Minimal Spanning Tree.
        """
        _graph = self.graph
        GeneralUtils.validateAttributes(_graph, ['pos', 'n'])

        _flipThreshold = math_cos(7./12.*math_pi) # Between 0 and 60 degrees

        device = _graph.pos.device
        _n = _graph.n
        def dfs_from_node(src, visited):
            visited[src] = True

            # Get neighbors of the current node
            neighbors = mst_edge_index[1, mst_edge_index[0] == src]

            # Recursive DFS on unvisited neighbors
            for dest in neighbors:
                if not visited[dest]:
                    if (_n[src] * _n[dest]).sum(dim=0) < _flipThreshold:
                        _n[dest] *= -1
                    dfs_from_node(dest, visited)

        N = _graph.num_nodes
        visited = torch_zeros(N, dtype=bool, device=device)
        start_node = torch_argmax(_graph.pos[:, 2])
        if _n[start_node, 2] < 0:
            _n[start_node] *= -1
        dfs_from_node(start_node, visited)
