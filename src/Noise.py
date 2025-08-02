from pathlib import Path

from torch import (
    float as torch_float,
    full as torch_full,
    load as torch_load,
    normal as torch_normal,
    randperm as torch_randperm,
    save as torch_save,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch_geometric.data import (
    Data as tg_data_Data
)
from typing import (
    Union as typing_Union
)

'''
    The purpose of this class is to create noise on Objects.
    Noise is not edge or face dependant and therefore it is only implemented for pointclouds (and therefore also meshes).
'''
class Noise:

    def __init__(self, graph: tg_data_Data):
        self.graph = graph

    # Generates noise for the given object.
    # noise_level is a number representing the intensity of the noise.
    # noise_type is 0 for Gaussian and 1 for Impulsive noise.
    # noise_direction is 0 for Vertex Normal direction and 1 for a Random direction.
    def generateNoise(self, noise_level: typing_Union[int, float], mean_edge_length: float, noise_type: int=0, noise_direction: int=0, keepNormals: bool = False):
        in_range = lambda input, start, end: (input - (end - start)*0.5)**2
        if in_range(noise_level, 0, 1) > in_range(0, 0, 1):
            raise ValueError(f"noise_level is {noise_level}, but should be a positive number!")
        if in_range(noise_type, 0, 1) > in_range(0, 0, 1):
            raise ValueError(f"noise_type is {noise_type}, but should be a number between 0 and 1!")
        if in_range(noise_direction, 0, 1) > in_range(0, 0, 1):
            raise ValueError(f"noise_direction is {noise_direction}, but should be a number between 0 and 1!")

        self.noise_level = noise_level
        self.noise_type = noise_type
        self.noise_direction = noise_direction
        
        _graph = self.graph
        _gt, _ = self.getGT()
        _num_nodes = _graph.num_nodes
        _num_nodes_size = (_num_nodes, 3)
        _device = _gt.device

        standard_deviation = mean_edge_length * noise_level
        random_numbers = torch_normal(torch_zeros(_num_nodes_size, device=_device, dtype=torch_float), torch_full(_num_nodes_size, standard_deviation, device=_device, dtype=torch_float))
        random_offset = random_numbers if noise_direction == 1 else _graph.n * random_numbers[:, 0, None]
        if noise_type == 1:
            _random_indices = torch_randperm(_num_nodes)[:int(_num_nodes*(1 - noise_level))]
            random_offset[_random_indices] = 0

        self.setNoise(_gt + random_offset, keepNormals)
    
    def getGT(self):
        _graph = self.graph
        _pos = _graph.gt if hasattr(_graph, "gt") else _graph.pos
        _normal = _graph.gt_n if hasattr(_graph, "gt_n") else _graph.n
        return _pos, _normal

    def setNoise(self, noise: torch_Tensor, keepNormals: bool = False):
        _graph = self.graph

        # Save ground truth if non exists yet
        _graph.gt, _graph.gt_n = self.getGT()
        
        # Apply noise
        _graph.pos = noise

        # Remove normals, because they do not match with positions anymore
        if not keepNormals:
            delattr(_graph, "n")

    def resetNoise(self):
        _object = self.graph
        if hasattr(_object, "gt"):
            _object.pos = _object.gt
            self.noise_level = None
            self.noise_type = None
            self.noise_direction = None
        else:
            raise ValueError("Can't reset noise if noise has never been applied")
    
    def saveNoise(self, noise_dir: str):
        if self.noise_level is None or not isinstance(self.noise_level, (int, float)) or self.noise_level == 0:
            raise ValueError(f"No noise has been set, therefore saving is useless.")
        file = Path(noise_dir)
        if file.exists() and file.is_file():
            raise ValueError()
        _object = self.graph
        vertices_to_save = _object.v
        noise_dir.mkdir(parents=True, exist_ok=True)
        noise_id = len([f for f in noise_dir.iterdir()])
        filename = f"{self.noise_type}_{self.noise_direction}_{self.noise_level}_{noise_id}.pt"
        torch_save(vertices_to_save, noise_dir / filename)
        return filename
    
    def loadNoise(self, file_path: str):
        file = Path(file_path)
        if file.exists() and file.is_file() and file.suffix == ".pt":
            data = torch_load(file)
            self.graph.setVertices(data)
