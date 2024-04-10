from .Object import Pointcloud
from .Utils import GeneralUtils

from deltaconv.geometry import (
    estimate_basis as dc_geo_estimate_basis
)
from igl import (
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency,
    per_vertex_normals as igl_per_vertex_normals,
    per_face_normals as igl_per_face_normals,
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency
)
from math import (
    cos as math_cos,
    pi as math_pi
)
from numpy import (
    any as np_any,
    arange as np_arange,
    argwhere as np_argwhere,
    cross as np_cross,
    logical_not as np_logical_not,
    nan_to_num as np_nan_to_num,
    ndarray as np_ndarray,
    ones as np_ones,
    unique as np_unique,
    vstack as np_vstack,
    zeros as np_zeros
)
from robust_laplacian import point_cloud_laplacian as robust_pointcloud_laplacian
from sklearn.preprocessing import normalize as sklearn_preprocessing_normalize
from torch import (
    arange as torch_arange,
    argmax as torch_argmax,
    argsort as torch_argsort,
    from_numpy as torch_from_numpy,
    long as torch_long,
    stack as torch_stack,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch_cluster import (
    knn_graph as tc_knn_graph
)
from torch_geometric.data import Data as tg_data_Data
from torch_geometric.utils import (
    to_undirected as tg_utils_to_undirected
)
from tqdm import tqdm
from typing import Any as typing_Any
from warnings import warn

class GraphBuilder:
    r"""
    Creates a graph from a pointcloud.
    This step is done before Patch selection, alignment and input generation.
    """

    @classmethod
    def calculateRobustLaplacian(cls, pointcloud: Pointcloud) -> torch_Tensor:
        L, M = robust_pointcloud_laplacian(pointcloud.v.numpy())
        Lcoo = L.tocoo()
        return torch_from_numpy(np_vstack((Lcoo.row, Lcoo.col))).long(), torch_from_numpy(M.data)

    @classmethod
    def calculateEdgeCost(cls, graph: tg_data_Data) -> None:
        r"""
        Computes the cost of the edges based on the edge_index and normal vectors.
        Cost function is 1 - n_i * n_j.

        Args:
            graph (Torch Geometric Data): Graph on which to compute cost.
        """
        GeneralUtils.validateAttributes(graph, ["edge_index", "n"])
        normals = graph.n[graph.edge_index]
        graph.edge_attr = 1 - (normals[0] * normals[1]).sum(dim=-1).abs_()

    @classmethod
    def calculateUndirectedMST(cls, graph: tg_data_Data) -> torch_Tensor:
        r"""
        Computes and returns the indices of the edges that point to the edges
        that are contained in the Minimal Spanning Tree (MST).

        Args:
            graph (Torch Geometric Data): Graph on which to build MST.
        """
        GeneralUtils.validateAttributes(graph, ['pos', 'edge_index', 'edge_attr'])

        device = graph.pos.device
        N = graph.num_nodes
        E = graph.num_edges
        _edge_index = graph.edge_index
        _edge_attr = graph.edge_attr

        sorted_cost = torch_argsort(_edge_attr, dim=0, descending=False)
        included_edges = torch_zeros(E, dtype=bool, device=device)
        groups = torch_arange(N, device=device)
        for index in tqdm(sorted_cost, desc="Creating MST.."):
            _edge = _edge_index[:, index]
            if groups[_edge[0]] != groups[_edge[1]]:
                included_edges[index] = True
                _edge_groups = groups[_edge]
                groups[groups == _edge_groups[0]] = _edge_groups[1]
        edge_indices = included_edges.nonzero().squeeze()
        return tg_utils_to_undirected(edge_index=_edge_index[:, edge_indices], edge_attr=_edge_attr[edge_indices], num_nodes=N)

    @classmethod
    def flipNormals(cls, graph: tg_data_Data, mst_edge_index: torch_Tensor) -> None:
        r"""
        Flips normals in the graph based on the MST.

        Args:
            graph (Torch Geometric Data): Graph that contains normals that need to be flipped.
            mst (Torch Tensor (E,)): Indices that point to the edges that are included in the Minimal Spanning Tree.
        """
        GeneralUtils.validateAttributes(graph, ['pos', 'n'])

        _flipThreshold = math_cos(7./12.*math_pi) # Between 0 and 60 degrees

        device = graph.pos.device
        _n = graph.n
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

        N = graph.num_nodes
        visited = torch_zeros(N, dtype=bool, device=device)
        start_node = torch_argmax(graph.pos[:, 2])
        if _n[start_node, 2] < 0:
            _n[start_node] *= -1
        dfs_from_node(start_node, visited)

    @classmethod
    def flipNormalsFacade(cls, graph: tg_data_Data) -> None:
        cls.calculateEdgeCost(graph)
        mst, _ = cls.calculateUndirectedMST(graph)
        cls.flipNormals(graph, mst)

    @classmethod
    def pointcloudToGraph(cls, pointcloud: Pointcloud) -> tg_data_Data:
        edges, mass = cls.calculateRobustLaplacian(pointcloud)
        graph = tg_data_Data(pos=pointcloud.v, edge_index=edges, mass=mass)
        if pointcloud.hasNormals():
            graph.n = pointcloud.n
        return graph
    
    @classmethod
    def generateNormalsDeltaconv(cls, graph: tg_data_Data, edge_index: torch_Tensor = None) -> None:
        GeneralUtils.validateAttributes(graph, ['pos'])
        if edge_index is None:
            edge_index = tc_knn_graph(graph.pos, k=12)
        graph.n, _, _ = dc_geo_estimate_basis(graph.pos, edge_index, k=12)
