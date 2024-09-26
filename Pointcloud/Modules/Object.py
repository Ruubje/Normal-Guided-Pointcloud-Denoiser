#https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

from .Utils import TorchUtils

from copy import deepcopy
from igl import (
    barycenter as igl_barycenter,
    doublearea as igl_doublearea,
    read_obj as igl_read_obj,
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency,
    per_vertex_normals as igl_per_vertex_normals,
    per_face_normals as igl_per_face_normals,
    triangle_triangle_adjacency as igl_triangle_triangle_adjacency,
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency
)
from numpy import (
    arange as np_arange,
    array as np_array,
    c_ as np_c_,
    nan_to_num as np_nan_to_num,
    repeat as np_repeat,
    sum as np_sum,
    vstack as np_vstack,
    zeros as np_zeros
)
from pathlib import Path
#https://github.com/nmwsharp/robust-laplacians-py
from robust_laplacian import point_cloud_laplacian
from scipy.spatial import KDTree as scipy_spatial_KDTree
from torch import (
    float as torch_float,
    from_numpy as torch_from_numpy,
    long as torch_long,
    tensor as torch_tensor,
    Tensor as torch_Tensor
)
from torch_geometric.data import Data as tg_data_Data
from torch_geometric.transforms import SamplePoints as tg_transforms_SamplePoints

class Pointcloud:

    def __init__(self, v: torch_Tensor, n: torch_Tensor = None) -> None:
        assert v.is_floating_point()
        assert v.dim() == 2
        assert v.size(1) == 3
        if n is not None:
            assert n.is_floating_point()
            assert n.dim() == 2
            assert n.size(1) == 3
            assert v.size(0) == n.size(0)

        self.v = v
        self.n = n

    def saveObj(self, file_path: str) -> None:
        f = open(file_path, "x")
        f.write("# File made by Ruben Band\n")
        for i in self.v:
            line = "v " + " ".join([str(x) for x in i.tolist()]) + "\n"
            f.write(line)
        if self.n is not None:
            for i in self.n:
                line = "vn " + " ".join([str(x) for x in i.tolist()]) + "\n"
                f.write(line)
        f.close()
        self.file_path = file_path

    @classmethod
    def loadObj(cls, file_path: str, device='cpu') -> Pointcloud:
        r"""
        We only implement loading, since saving doesn't work with this library...
        """
        path = Path(file_path)
        assert path.is_file()
        assert path.suffix == ".obj"

        v, _, n, fv, _, fn = igl_read_obj(file_path)
        v, n, fv, fn = torch_tensor(v, dtype=torch_float, device=device), torch_tensor(n, dtype=torch_float, device=device), torch_tensor(fv, dtype=torch_long, device=device), torch_tensor(fn, dtype=torch_long, device=device)
        if n.size(0) > 0 and fn.size(0) > 0:
            pointcloud = Pointcloud(v, TorchUtils.face2vertexNormals(v, fv, n, fn))
        elif n.size(0) > 0 and v.size(0) == n.size(0):
            pointcloud = Pointcloud(v, n) 
        else:
            pointcloud = Pointcloud(v)
        pointcloud.file_path = file_path
        return pointcloud
    
    @classmethod
    def sampleObj(cls, file_path: str, num_points: int, device='cpu') -> Pointcloud:
        r"""
        Load an obj file and sample points from the mesh.
        """
        path = Path(file_path)
        assert path.is_file()
        assert path.suffix == ".obj"

        v, _, n, fv, _, fn = igl_read_obj(file_path)
        v, n, fv, fn = torch_tensor(v, dtype=torch_float, device=device), torch_tensor(n, dtype=torch_float, device=device), torch_tensor(fv, dtype=torch_long, device=device), torch_tensor(fn, dtype=torch_long, device=device)

        assert v.size(1) == 3
        assert fv.size(1) == 3

        data = tg_data_Data(pos=v, face=fv.T)
        new_pointcloud = tg_transforms_SamplePoints(
            num=num_points,
            include_normals=True
        )(data)
        pointcloud = Pointcloud(new_pointcloud.pos, new_pointcloud.normal)
        pointcloud.file_path = file_path
        return pointcloud
    
    def hasNormals(self):
        return self.n is not None
    
    def hasFilePath(self):
        return self.file_path is not None

'''
    This file contains classes that represent .obj files.
    A mesh is a pointcloud connected with triangles and therefore
    it extends the pointcloud (and sometimes overrides functions)
'''

class FileObject:

    '''
        Init stuff
    '''

    def __init__(self, file_path, read_file=True):
        if not isinstance(file_path, str):
            raise ValueError(f"file_path: ({file_path}) should be a string.")
        self.file_path = Path(file_path)
        if self.file_path.suffix != ".obj" or not self.file_path.exists() or not self.file_path.is_file():
            raise FileNotFoundError(f"File {file_path} is not a .obj file or is not found.")
        if read_file:
            return self.readFile()
     
    def readFile(self):
        file_path = self.file_path
        if file_path.is_file():
            data = igl_read_obj(str(file_path))
            self.gt = data[0]
            return data

class FilePointcloud(FileObject):

    DEFAULT_NEIGHBOURHOOD_MODE = 1

    '''
        Init Stuff
    '''

    def __init__(self, file_path, read_file=True):
        super().__init__(file_path, read_file=False)
        self.graph_vertices_match = False
        if read_file:
            self.readFile()

    def readFile(self, mode=DEFAULT_NEIGHBOURHOOD_MODE, calculate_meta=True):
        data = super().readFile()
        _data0 = data[0]
        _data3 = data[3]
        self.f = _data3
        self.vta = igl_vertex_triangle_adjacency(_data3, _data0.shape[0])
        self.setVertices(_data0)
        if calculate_meta:
            self.setGraph(mode=mode)
            self.g.y = torch_from_numpy(deepcopy(self.vn))
        return data
    
    '''
        Object stuff
    '''
    
    def setVertices(self, v, mode=DEFAULT_NEIGHBOURHOOD_MODE, calculate_meta=True):
        self.graph_vertices_match = False
        self.v = v
        self.kdtree = scipy_spatial_KDTree(v)
        self.calculateNormals()
        self.calculateAreas()
        if calculate_meta:
            self.setGraph(mode=mode)

    def calculateNormals(self):
        self.vn = np_nan_to_num(igl_per_vertex_normals(self.v, self.f), copy=False, nan=0)
        self.fn = igl_per_face_normals(self.v, self.f, FileMesh.DEGENERATE_NORMAL_PLACEHOLDER)

    def calculateAreas(self):
        if self.vta is None:
            raise AttributeError("Attribure 'vta' not found.")
        _vta = self.vta
        _vta1 = _vta[1]
        self.fa = igl_doublearea(self.v, self.f) / 2.0
        self.va = np_array([np_sum(self.fa[_vta[0][_vta1[vi]:_vta1[vi+1]]]) / 3 for vi in np_arange(len(self.v))])
    
    def getNormals(self):
        return self.vn

    def getAreas(self):
        return self.va
    
    '''
        Graph stuff
    '''

    def setGraph(self, mode=DEFAULT_NEIGHBOURHOOD_MODE):
        old_y = None
        if hasattr(self, 'g') and hasattr(self.g, 'y'):
            old_y = self.g.y
        self.g = self.toGraph(mode=mode)
        self.g.y = old_y
        self.graph_vertices_match = True

    def toNodes(self):
        return torch_from_numpy(self.v)
    
    # Create a graph from the pointcloud
    # mode: 0 is knn and 1 is 
    def toEdges(self, mode=DEFAULT_NEIGHBOURHOOD_MODE):
        if mode == 0:
            KNN_K_HARDCODED = 12
            # k=12, because the mean + var degree of the armadillo & fandisk are 8 & 10 respectively.
            # This means that knn for k=12 will cover most of the details that the robust laplacian will catch as well!
            return TorchUtils.toEdgeTensor(self.kdtree.query(self.v, k=KNN_K_HARDCODED+1)[1][:, 1:])
        elif mode == 1:
            L, _ = point_cloud_laplacian(self.v)
            Lcoo = L.tocoo()
            return torch_from_numpy(np_vstack((Lcoo.row, Lcoo.col))).long()
        else:
            raise ValueError(f"mode {mode} is undefined within this method.")
    
    def toGraph(self, mode=DEFAULT_NEIGHBOURHOOD_MODE):
        return tg_data_Data(edge_index=self.toEdges(mode=mode), pos=self.toNodes(), n=torch_from_numpy(self.getNormals()), a=torch_from_numpy(self.getAreas()))

class FileMesh(FilePointcloud):

    DEGENERATE_NORMAL_PLACEHOLDER = np_zeros(3)

    '''
        Init Stuff
    '''

    def __init__(self, file_path, read_file=True):
        super().__init__(file_path, read_file=False)
        if read_file:
            self.readFile()

    def readFile(self, calculate_meta=True):
        data = super().readFile(calculate_meta=False)
        if calculate_meta:
            self.calculateNormals()
            self.setGraph()
            self.g.y = torch_from_numpy(deepcopy(self.fn))
        return data
    
    '''
        Object stuff
    '''

    def setVertices(self, v, calculate_meta=True):
        super().setVertices(v, calculate_meta=False)
        if calculate_meta:
            self.toGraph()

    def getNormals(self):
        return self.fn
    
    def getAreas(self):
        return self.fa

    '''
        Graph stuff
    '''

    def setGraph(self):
        old_y = None
        if hasattr(self, 'g') and hasattr(self.g, 'y'):
            old_y = self.g.y
        self.g = self.toGraph()
        self.g.y = old_y
        self.graph_vertices_match = True

    def toNodes(self):
        return torch_from_numpy(igl_barycenter(self.v, self.f))
    
    def toEdges(self):
        num_f = self.f.shape[0]
        tta = igl_triangle_triangle_adjacency(self.f)[0]
        return torch_from_numpy(np_c_[np_repeat(np_arange(num_f)[:, None], 3, axis=1).flatten(), tta.flatten()].T).long()
    
    def toGraph(self):
        return tg_data_Data(edge_index=self.toEdges(), pos=self.toNodes(), n=torch_from_numpy(self.getNormals()), a=torch_from_numpy(self.getAreas()))