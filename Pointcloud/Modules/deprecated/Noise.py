from .Object import FilePointcloud

from copy import deepcopy
from igl import per_vertex_normals as igl_per_vertex_normals, read_obj as igl_read_obj, write_obj as igl_write_obj
from numpy import arange as np_arange, average as np_average, einsum as np_einsum, load as np_load, ones as np_ones, nan_to_num as np_nan_to_num, repeat as np_repeat, save as np_save, vstack as np_vstack
from numpy.linalg import norm as np_linalg_norm
from numpy.random import choice as np_random_choice, normal as np_random_normal

'''
    The purpose of this class is to create noise on Objects.
    Noise is not edge or face dependant and therefore it is only implemented for pointclouds (and therefore also meshes).
'''
class Noise:
    
    NOISE_DIR = "Noise"
    NOISE_ID = 0

    def __init__(self, pointcloud):
        if not isinstance(pointcloud, FilePointcloud):
            raise ValueError(f"pointcloud does not have the type Pointcloud.\nPointcloud type: {type(pointcloud)}")

        self.object = pointcloud
        self.noise_dir = pointcloud.file_path.parent / Noise.NOISE_DIR
        self.noise_dir.mkdir(parents=True, exist_ok=True)
        Noise.NOISE_ID = len([f for f in self.noise_dir.iterdir()])

    # Generates noise for the given object.
    # noise_level is a number representing the intensity of the noise.
    # noise_type is 0 for Gaussian and 1 for Impulsive noise.
    # noise_direction is 0 for Vertex Normal direction and 1 for a Random direction.
    def generateNoise(self, noise_level, noise_type=0, noise_direction=0):
        in_range = lambda input, start, end: (input - (end - start)*0.5)**2
        if not isinstance(noise_level, (int, float)) or in_range(noise_level, 0, 1) > in_range(0, 0, 1):
            raise ValueError(f"noise_level is {noise_level}, but should be a positive number!")
        if not isinstance(noise_type, int) or in_range(noise_type, 0, 1) > in_range(0, 0, 1):
            raise ValueError(f"noise_type is {noise_type}, but should be a number between 0 and 1!")
        if not isinstance(noise_direction, int) or in_range(noise_direction, 0, 1) > in_range(0, 0, 1):
            raise ValueError(f"noise_direction is {noise_direction}, but should be a number between 0 and 1!")

        self.noise_level = noise_level
        self.noise_type = noise_type
        self.noise_direction = noise_direction
        
        _object = self.object
        _g = _object.g
        _pos = _g.pos
        _gt_shape = _object.gt.shape
        _edge_index = _g.edge_index
        avg_edge_length = np_average(np_linalg_norm(_pos[_edge_index[1]] - _pos[_edge_index[0]], axis=1))
        standard_deviation = avg_edge_length * noise_level
        random_numbers = np_random_normal(0, standard_deviation, _gt_shape)
        random_offset = _object.vn if noise_direction == 0 else random_numbers
        random_offset = random_numbers[:, 0][:, None]*_object.vn if noise_direction == 0 else random_numbers
        if noise_type == 1:
            _random_indices = np_random_choice(np_arange(_gt_shape[0]), size=int(_gt_shape[0]*(1 - noise_level)), replace=False)
            random_offset[_random_indices] = 0

        _object.setVertices(_object.gt + random_offset)
    
    def resetNoise(self):
        self.noise_level = None
        self.noise_type = None
        self.noise_direction = None

        _object = self.object
        _object.setVertices(_object.gt)
    
    def saveNoise(self):
        if self.noise_level is None or not isinstance(self.noise_level, (int, float)) or self.noise_level == 0:
            raise ValueError(f"No noise has been set, therefore saving is useless.")
        
        _object = self.object
        vertices_to_save = _object.v
        filename = f"{Noise.NOISE_ID}_{self.noise_type}_{self.noise_direction}_{self.noise_level}_{_object.file_path.stem}.npy"
        Noise.NOISE_ID += 1
        np_save(self.noise_dir / filename, vertices_to_save)
        return filename
    
    def loadNoise(self, filename):
        file = self.noise_dir / filename
        if file.exists() and file.is_file() and file.suffix == ".npy":
            data = np_load(file)
            self.object.setVertices(data)
