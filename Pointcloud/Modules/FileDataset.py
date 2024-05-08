from . import Config as config
from .Noise import Noise
from .Object import Pointcloud
from .Preprocessor import Preprocessor
from .Utils import GeneralUtils

from dataclasses import dataclass
from gc import collect as gc_collect
from pathlib import Path
from shutil import copy2 as shutil_copy2
from tqdm import tqdm
from torch import (
    cuda as torch_cuda,
    load as torch_load,
    randperm as torch_randperm,
    save as torch_save
)
from torch_geometric.data import (
    Batch as tg_data_Batch,
    DataLoader as tg_data_DataLoader,
    InMemoryDataset as tg_data_InMemoryDataset
)
from torch_geometric.loader import (
    DataLoader as tg_loader_DataLoader
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
        gaussian = [0.1, 0.2, 0.3],
        impulsive = [0.1, 0.2, 0.3]
    )
    DEFAULT_SPLIT = (0.6, 0.2, 0.2)
    EXTENSION = ".pt"
    CLASSES = "_classes"
    GAUSSIAN = "_gaussian_"
    IMPULSIVE = "_impulsive_"
    FEATURE = "_feature"
    NON_FEATURE = "_nonfeature"

    def __init__(self, root: str,
                pre_add_objects: list[Pointcloud] = None,
                noise_levels: NoiseLevels=DEFAULT_NOISE_LEVELS,
                split: typing_Tuple[float, float, float] = DEFAULT_SPLIT,
                split_name: str = None,
                transform=None,
                pre_transform=None):
        if transform is not None or pre_transform is not None:
            warnings_warn("transform or pre_transform given. Methods for these are not implemented!")
        
        self.objects = pre_add_objects
        self.noise_levels = noise_levels
        super(FileDataset, self).__init__(root, transform, pre_transform)
        _processed_paths = self.processed_paths
        feature_graphs = []
        nonfeature_graphs = []
        for path in [x for x in _processed_paths if not x.endswith(self.CLASSES + self.EXTENSION)]:
            store = torch_load(path)
            graph_list = tg_data_Batch.to_data_list(store)
            if path.endswith(self.NON_FEATURE + self.EXTENSION):
                nonfeature_graphs += graph_list
            else:
                feature_graphs += graph_list
        self.data = (tg_data_Batch.from_data_list(feature_graphs), tg_data_Batch.from_data_list(nonfeature_graphs))
        try:
            self.loadSplitIndices(split_name)
        except:
            self.generateSplitIndices(split=split)
            self.saveSplitIndices(split_name)
        self.createDatasets()
    
    def assertSplitIndices(self):
        if not hasattr(self, "splitIndices"):
            raise ValueError("splitIndices not found. A split has not been created yet.")

    def getSplit(self):
        self.assertSplitIndices()
        return tuple(self.splitIndices.size())

    def generateSplitIndices(self, split: typing_Tuple[float, float, float]=DEFAULT_SPLIT):
        if not sum(split) == 1:
            raise ValueError("Sum of train, validation and test splits should be 1.")
        
        _data = self._data
        data0 = _data[0]
        data1 = _data[1]
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
            self.splitIndices = torch_load(file_path)
        else:
            raise ValueError("Can't find split file")
    
    def createDatasets(self):
        _data = self._data
        data0 = _data[0]
        data1 = _data[1]
        _splitIndices = self.splitIndices
        self.train_ds = tg_data_Batch.from_data_list(data0[_splitIndices[0]] + data1[_splitIndices[3]])
        self.val_ds = tg_data_Batch.from_data_list(data0[_splitIndices[1]] + data1[_splitIndices[4]])
        self.test_ds = tg_data_Batch.from_data_list(data0[_splitIndices[2]] + data1[_splitIndices[5]])
        print(f"train bs: {self.train_ds.batch_size}\nval bs: {self.val_ds.batch_size}\ntest bs: {self.test_ds.batch_size}")
    
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
        if self.object is not None:
            for pointcloud in self.objects:
                pointcloud.saveObj(Path(self.raw_dir) + Path(pointcloud.file_path).name)

    def process(self):
        _device = config.ACCELERATOR
        _raw_paths = self.raw_paths
        _nl = self.noise_levels
        _pdir = Path(self.processed_dir)
        data_list = []
        for path in _raw_paths:
            _stem = Path(path).stem
            pointcloud = Pointcloud.loadObj(path, _device)
            preprocessor = Preprocessor(pointcloud)
            noise = Noise(preprocessor.graph)
            classes_file = _pdir / (_stem + self.CLASSES + self.EXTENSION)
            if not classes_file.exists():
                classes = preprocessor.getClasses()
                torch_save(classes, classes_file)
            else:
                classes = torch_load(classes_file)
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
                        noise.generateNoise(level, 0, 0)
                        preprocessor.graphBuilder.generateNormals()
                        data_list, _ = preprocessor.getPatches(indices[i])
                        store = tg_data_Batch.from_data_list(data_list)
                        torch_save(store, str(file_location))
            for level in tqdm(_nl.impulsive, desc=f"Preprocessing gaussian noise for {_stem}"):
                for i in range(len(indices)):
                    group = self.FEATURE if i == 0 else self.NON_FEATURE
                    file_location = _pdir / (_stem + self.IMPULSIVE + str(level) + group + self.EXTENSION)
                    if not file_location.exists():
                        noise.generateNoise(level, 1, 0)
                        preprocessor.graphBuilder.generateNormals()
                        data_list, _ = preprocessor.getPatches(indices[i])
                        store = tg_data_Batch.from_data_list(data_list)
                        torch_save(store, str(file_location))
        torch_cuda.empty_cache()
        gc_collect()
    
    def len(self) -> int:
        return len(self._data)
    
    def get(self, idx: int):
        return self._data[idx]

    def train_dataloader(self, batch_size, num_workers):
        return tg_loader_DataLoader(
            dataset=self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self, batch_size, num_workers):
        return tg_loader_DataLoader(
            dataset=self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self, batch_size, num_workers):
        return tg_loader_DataLoader(
            dataset=self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
        )

def getGroupSizes(features: int, non_features: int, r: float = 1.5) -> typing_Union[int, int]:
    ratio = float(features) / float(non_features)
    if ratio > r:
        return (int(non_features * r), non_features)
    else:
        return (features, int(features / r))

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
