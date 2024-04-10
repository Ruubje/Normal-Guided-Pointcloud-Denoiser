from .Noise import Noise
from .Object import FilePointcloud
from .Preprocessor import Preprocessor

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2 as shutil_copy2
from torch import (
    load as torch_load,
    randperm as torch_randperm,
    save as torch_save,
)
from torch_geometric.data import (
    Batch as tg_data_Batch,
    DataLoader as tg_data_DataLoader,
    InMemoryDataset as tg_data_InMemoryDataset
)
from torch_geometric.loader import (
    DataLoader as tg_loader_DataLoader
)
from typing import Union as typing_Union
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

    def __init__(self, root: str, objects: list[FilePointcloud] = None, noise_levels: NoiseLevels=DEFAULT_NOISE_LEVELS, split: typing_Union[float, float, float]=DEFAULT_SPLIT, transform=None, pre_transform=None):
        if objects is not None and len(objects) == 0:
            raise ValueError("Cannot create an empty dataset.")
        if not sum(split) == 1:
            raise ValueError("Sum of train, validation and test splits should be 1.")
        if transform is not None or pre_transform is not None:
            warnings_warn("transform or pre_transform given. Methods for these are not implemented!")
        
        self.objects = objects
        self.noise_levels = noise_levels
        self.split = split
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
        # TODO create train/val/test split from group splitted data
        data0 = self.data[0]
        data1 = self.data[1]
        num_features = len(data0)
        num_nonfeatures = len(data1)
        split0 = split[0]
        split1 = split[1]
        features_split0 = int(num_features * split0)
        features_split1 = features_split0 + int(num_features * split1)
        nonfeatures_split0 = int(num_nonfeatures * split0)
        nonfeatures_split1 = features_split0 + int(num_nonfeatures * split1)
        features_perm = torch_randperm(num_features)
        nonfeatures_perm = torch_randperm(num_nonfeatures)
        features_train_indices = features_perm[:features_split0]
        features_val_indices = features_perm[features_split0:features_split1]
        features_test_indices = features_perm[features_split1:]
        nonfeatures_train_indices = nonfeatures_perm[:nonfeatures_split0]
        nonfeatures_val_indices = nonfeatures_perm[nonfeatures_split0:nonfeatures_split1]
        nonfeatures_test_indices = nonfeatures_perm[nonfeatures_split1:]
        self.train_ds = tg_data_Batch.from_data_list(data0[features_train_indices] + data1[nonfeatures_train_indices])
        self.val_ds = tg_data_Batch.from_data_list(data0[features_val_indices] + data1[nonfeatures_val_indices])
        self.test_ds = tg_data_Batch.from_data_list(data0[features_test_indices] + data1[nonfeatures_test_indices])
    
    @property
    def raw_file_names(self) -> list[str]:
        if self.objects is None:
            return [x.name for x in Path(self.raw_dir).glob("*.obj")]
        else:
            return [Path(x.file_path).name for x in self.objects]

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
        for path in [x.file_path for x in self.objects]:
            shutil_copy2(path, self.raw_dir)

    def process(self):
        _raw_paths = self.raw_paths
        _nl = self.noise_levels
        _pdir = Path(self.processed_dir)
        data_list = []
        for path in _raw_paths:
            _stem = Path(path).stem
            pointcloud = FilePointcloud(path)
            preprocessor = Preprocessor(pointcloud)
            noise = Noise(pointcloud)
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
            # print(f"Amount of nodes to be grouped: {groups.size(0)}\nAmount of features: {feature_idx_n}\nAmount of non-features: {nonfeature_idx.size(0)}\nAmount of indices returned: {indices.size(0)}")
            for level in _nl.gaussian:
                for i in range(len(indices)):
                    group = self.FEATURE if i == 0 else self.NON_FEATURE
                    file_location = _pdir / (_stem + self.GAUSSIAN + str(level) + group + self.EXTENSION)
                    if not file_location.exists():
                        noise.generateNoise(level, 0, 0)
                        data_list = preprocessor.getGraphs(indices[i])
                        store = tg_data_Batch.from_data_list(data_list)
                        torch_save(store, str(file_location))
            for level in _nl.impulsive:
                for i in range(len(indices)):
                    group = self.FEATURE if i == 0 else self.NON_FEATURE
                    file_location = _pdir / (_stem + self.IMPULSIVE + str(level) + group + self.EXTENSION)
                    if not file_location.exists():
                        noise.generateNoise(level, 1, 0)
                        data_list = preprocessor.getGraphs(indices[i])
                        store = tg_data_Batch.from_data_list(data_list)
                        torch_save(store, str(file_location))
    
    def len(self) -> int:
        return len(self.data)
    
    def get(self, idx: int):
        return self.data[idx]

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
