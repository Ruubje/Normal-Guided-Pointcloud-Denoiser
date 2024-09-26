from . import Config as config
from .Object import Pointcloud
from .Processor import Processor

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from torch import (
    load as torch_load,
    randperm as torch_randperm,
    save as torch_save
)
from torch.utils.data import get_worker_info as torch_utils_data_get_worker_info
from torch_geometric.data import (
    Batch as tg_data_Batch,
    InMemoryDataset as tg_data_InMemoryDataset
)
from typing import (
    Union as typing_Union,
    Tuple as typing_Tuple
)
from warnings import warn as warnings_warn

@dataclass
class NoiseLevels:
    gaussian: list[float]
    impulsive: list[float]

class FileDataset(tg_data_InMemoryDataset):

    DEFAULT_NOISE_LEVELS = NoiseLevels(
        gaussian = config.GAUSSIAN_NOISE_LEVELS,
        impulsive = config.IMPULSIVE_NOISE_LEVELS
    )
    DEFAULT_SPLIT = config.SPLIT
    EXTENSION = ".pt"
    CLASSES = "_classes"
    GAUSSIAN = "_gaussian_"
    IMPULSIVE = "_impulsive_"
    FEATURE = "_feature"
    NON_FEATURE = "_nonfeature"

    def __init__(self, root: str,
                split_name: str,
                dataset_idx: int = 0,
                noise_levels: NoiseLevels=DEFAULT_NOISE_LEVELS,
                split_distribution: typing_Tuple[float, float, float] = DEFAULT_SPLIT,
                pre_add_objects: list[Pointcloud] = None,
                transform=None,
                pre_transform=None):
        if transform is not None or pre_transform is not None:
            warnings_warn("transform or pre_transform given. Methods for these are not implemented!")
        assert dataset_idx >= 0 and dataset_idx <= 2
        
        self.objects = pre_add_objects
        self.noise_levels = noise_levels
        super(FileDataset, self).__init__(root, transform, pre_transform)
        _processed_paths = self.processed_paths
        feature_graphs = []
        nonfeature_graphs = []
        for path in [x for x in _processed_paths if not x.endswith(self.CLASSES + self.EXTENSION)]:
            # Load on cpu to prevent later multiprocess dataloader errors
            store = torch_load(path, map_location="cpu")
            graph_list = tg_data_Batch.to_data_list(store)
            if path.endswith(self.NON_FEATURE + self.EXTENSION):
                nonfeature_graphs += graph_list
            else:
                feature_graphs += graph_list
        temp = (tg_data_Batch.from_data_list(feature_graphs), tg_data_Batch.from_data_list(nonfeature_graphs))
        try:
            self.loadSplitIndices(split_name)
        except:
            self.generateSplitIndices(data=temp, split=split_distribution)
            self.saveSplitIndices(split_name)
        
        _splitIndices = self.splitIndices
        self.data, self.slices = self.collate(temp[0][_splitIndices[dataset_idx]] +
                                              temp[1][_splitIndices[dataset_idx + 3]])
    
    def assertSplitIndices(self):
        if not hasattr(self, "splitIndices"):
            raise ValueError("splitIndices not found. A split has not been created yet.")

    def getSplit(self):
        self.assertSplitIndices()
        return tuple(self.splitIndices.size())

    def generateSplitIndices(self, data: typing_Tuple[tg_data_Batch, tg_data_Batch], split: typing_Tuple[float, float, float]=DEFAULT_SPLIT):
        if not sum(split) == 1:
            raise ValueError("Sum of train, validation and test splits should be 1.")
        
        data0 = data[0]
        data1 = data[1]
        _device = data0[0].x.device
        num_features = len(data0)
        num_nonfeatures = len(data1)
        split0 = split[0]
        split1 = split[1]
        features_split0 = int(num_features * split0)
        features_split1 = features_split0 + int(num_features * split1)
        nonfeatures_split0 = int(num_nonfeatures * split0)
        nonfeatures_split1 = nonfeatures_split0 + int(num_nonfeatures * split1)
        features_perm = torch_randperm(num_features, device=_device)
        nonfeatures_perm = torch_randperm(num_nonfeatures, device=_device)
        features_train_indices = features_perm[:features_split0]
        features_val_indices = features_perm[features_split0:features_split1]
        features_test_indices = features_perm[features_split1:]
        nonfeatures_train_indices = nonfeatures_perm[:nonfeatures_split0]
        nonfeatures_val_indices = nonfeatures_perm[nonfeatures_split0:nonfeatures_split1]
        nonfeatures_test_indices = nonfeatures_perm[nonfeatures_split1:]
        self.splitIndices = (features_train_indices, features_val_indices, features_test_indices, nonfeatures_train_indices, nonfeatures_val_indices, nonfeatures_test_indices)
    
    def saveSplitIndices(self, name: str, overwrite: bool = False):
        self.assertSplitIndices()
        file_path = Path(self.processed_dir) / (name + ".split")
        if not file_path.exists() or overwrite:
            torch_save(self.splitIndices, file_path)
        else:
            raise ValueError("File already exists and overwrite is set to False.")
        
    def loadSplitIndices(self, name: str):
        file_path = Path(self.processed_dir) / (name + ".split")
        if file_path.exists():
            self.splitIndices = torch_load(file_path, map_location="cpu")
        else:
            raise ValueError("Can't find split file")
    
    @property
    def raw_file_names(self) -> list[str]:
        files = [x.name for x in Path(self.raw_dir).glob("*.obj")]
        if self.objects is not None:
            files_from_objects = [Path(x.file_path).name for x in self.objects]
            files += files_from_objects
        return files


    @property
    def processed_file_names(self):
        _nl = self.noise_levels
        result = []
        for path in self.raw_paths:
            _stem = Path(path).stem
            result.append(_stem + self.CLASSES + self.EXTENSION)
            for level in _nl.impulsive:
                result.append(_stem + self.IMPULSIVE + str(level) + self.FEATURE + self.EXTENSION)
                result.append(_stem + self.IMPULSIVE + str(level) + self.NON_FEATURE + self.EXTENSION)
            for level in _nl.gaussian:
                result.append(_stem + self.GAUSSIAN + str(level) + self.FEATURE + self.EXTENSION)
                result.append(_stem + self.GAUSSIAN + str(level) + self.NON_FEATURE + self.EXTENSION)
        return result

    def download(self):
        if self.objects is not None:
            for pointcloud in self.objects:
                pointcloud.saveObj(Path(self.raw_dir) / Path(pointcloud.file_path).name)

    def process(self):
        _device = config.PROCESS_ACCELERATOR
        _raw_paths = self.raw_paths
        _nl = self.noise_levels
        _pdir = Path(self.processed_dir)
        for path in _raw_paths:
            _stem = Path(path).stem
            pointcloud = Pointcloud.loadObj(path, _device)
            preprocessor = Processor(pointcloud)
            _noise = preprocessor.noise
            classes_file = _pdir / (_stem + self.CLASSES + self.EXTENSION)
            if not classes_file.exists():
                classes = preprocessor.getMDFeatures()
                torch_save(classes, classes_file)
            else:
                classes = torch_load(classes_file, map_location=_device)
            groups = (classes == 2).logical_or(classes == 3)
            feature_idx = groups.nonzero().view(-1)
            nonfeature_idx = groups.logical_not().nonzero().view(-1)
            feature_idx_n = feature_idx.size(0)
            nonfeature_idx_n = nonfeature_idx.size(0)
            group_sizes = getGroupSizes(feature_idx_n, nonfeature_idx_n)
            indices = (
                torch_randperm(feature_idx_n)[:group_sizes[0]],
                torch_randperm(nonfeature_idx_n)[:group_sizes[1]],
            )
            for level in tqdm(_nl.gaussian, desc=f"Preprocessing gaussian noise for {_stem}"):
                for i in range(len(indices)):
                    group = self.FEATURE if i == 0 else self.NON_FEATURE
                    file_location = _pdir / (_stem + self.GAUSSIAN + str(level) + group + self.EXTENSION)
                    if not file_location.exists():
                        _noise.generateNoise(level, 0, 0)
                        preprocessor.graphBuilder.setAndFlipNormals()
                        data_list, _ = preprocessor.getMDPatches(indices[i])
                        store = tg_data_Batch.from_data_list(data_list)
                        torch_save(store, str(file_location))
            for level in tqdm(_nl.impulsive, desc=f"Preprocessing gaussian noise for {_stem}"):
                for i in range(len(indices)):
                    group = self.FEATURE if i == 0 else self.NON_FEATURE
                    file_location = _pdir / (_stem + self.IMPULSIVE + str(level) + group + self.EXTENSION)
                    if not file_location.exists():
                        _noise.generateNoise(level, 1, 0)
                        preprocessor.graphBuilder.setAndFlipNormals()
                        data_list, _ = preprocessor.getMDPatches(indices[i])
                        store = tg_data_Batch.from_data_list(data_list)
                        torch_save(store, str(file_location))

def getGroupSizes(features: int, non_features: int, r: float = 1.5) -> typing_Union[int, int]:
    ratio = float(features) / float(non_features)
    if ratio > r:
        return (int(non_features * r), non_features)
    else:
        return (features, int(features / r))

class SimpleDataset(tg_data_InMemoryDataset):

    DEFAULT_NOISE_LEVELS = NoiseLevels(
        gaussian = config.GAUSSIAN_NOISE_LEVELS,
        impulsive = config.IMPULSIVE_NOISE_LEVELS
    )
    DEFAULT_SPLIT = config.SPLIT
    EXTENSION = ".pt"
    CLASSES = "_classes"
    GAUSSIAN = "_gaussian_"
    IMPULSIVE = "_impulsive_"

    def __init__(self, root: str,
                dataset_idx: int = 0,
                noise_levels: NoiseLevels=DEFAULT_NOISE_LEVELS,
                split_distribution: typing_Tuple[float, float, float] = DEFAULT_SPLIT,
                transform=None,
                pre_transform=None):
        if transform is not None or pre_transform is not None:
            warnings_warn("transform or pre_transform given. Methods for these are not implemented!")
        assert dataset_idx >= 0 and dataset_idx <= 2
        
        self.split_distribution = split_distribution
        self.noise_levels = noise_levels
        super(SimpleDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[dataset_idx])
    
    @property
    def raw_file_names(self) -> list[str]:
        return [x.name for x in Path(self.raw_dir).glob("*.obj")]


    @property
    def processed_file_names(self):
        return ["train" + self.EXTENSION, "val" + self.EXTENSION, "test" + self.EXTENSION]

    def download(self):
        pass

    def process(self):
        _device = config.PROCESS_ACCELERATOR
        _raw_paths = self.raw_paths
        _nl = self.noise_levels
        patches = []
        for path in _raw_paths:
            _stem = Path(path).stem
            pointcloud = Pointcloud.loadObj(path, _device)
            processor = Processor(pointcloud)
            _noise = processor.noise
            _gb = processor.graphBuilder
            for level in tqdm(_nl.gaussian, desc=f"Preprocessing gaussian noise for {_stem}"):
                _noise.generateNoise(level, 0, 0)
                _gb.setAndFlipNormals()
                data_list, _ = processor.getMDPatches()
                patches += data_list
            for level in tqdm(_nl.impulsive, desc=f"Preprocessing impulsive noise for {_stem}"):
                _noise.generateNoise(level, 1, 0)
                _gb.setAndFlipNormals()
                data_list, _ = processor.getMDPatches()
                patches += data_list
        N = len(patches)
        random_split = torch_randperm(N, device=_device)
        sd = self.split_distribution
        first = int(N*sd[0])
        second = first + int(N*sd[1])
        train_idx = random_split[:first]
        val_idx = random_split[first:second]
        test_idx = random_split[second:]
        pps = self.processed_paths
        self.save([patches[x].cpu() for x in train_idx], pps[0])
        self.save([patches[x].cpu() for x in val_idx], pps[1])
        self.save([patches[x].cpu() for x in test_idx], pps[2])
            
# Example usage:
# dataset = PatchDataset(root='./data')

# Apply transformations to the data if desired
# if dataset.transform is not None:
#     dataset.transform = T.Compose([
#         T.RandomRotate(30, resample=False),
#         T.RandomTranslate(0.1)
#     ])

# Put the dataset in a data loader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Loop over the dataloader
# for batch in dataloader:
#     # Access the data for each graph in the batch
#     x, edge_index = batch.x, batch.edge_index
#     # Do something with the data (e.g., train your GNN)
#     print(x, edge_index)
