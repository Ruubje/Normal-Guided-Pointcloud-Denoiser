# Created by Yuefan Shen, Jhonve https://github.com/Jhonve/GCN-Denoiser
# Altered by Ruben Band

import os

import numpy as np
import scipy.io as sio
import h5py

import torch.utils.data as Tdata

from pathlib import Path
import random

# The idea from the classes have become the following:
# 1. You create a dataset and tell it locations of folders to use in the dataset.
# 2. Then you tell it to generate a list of file paths that are within the folders.
# 3. Then we split this list into training / validation lists.
# 4. Then we load these lists into a FileDataset to me used when training or validating.
# Seperate save files can be made for saving the dataset and saving the way it is split.
# When loading a dataset (list of file paths), the paths should be checked for correctness.
# When loading split settings, it should be checked if the split settings and the dataset agree on having the same size.
class FileDataset(Tdata.Dataset):
    def __init__(self, data_path, batch_size, num_workers, num_neighbors, is_train=True):
        super(FileDataset).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_neighbors = num_neighbors

        self.data_path = data_path
        self.SIZE = data_path.shape[0]

        self.is_train = is_train

    def __len__(self):
        return self.SIZE

    # Transforms an adjacency matrix and features vectors (representing a patch as graph) to input for a network.
    # Mainly pads the data, such that all inputs have the same shape.
    @classmethod
    def file2input(cls, input_matrix, input_features, num_neighbors=64):
        num_faces = input_matrix.shape[0]
        if(num_faces >= num_neighbors):
            input_matrix = input_matrix[0:num_neighbors, 0:num_neighbors]
            input_features = input_features[0:num_neighbors]
        else:
            # Matrix Padding
            input_matrix = np.pad(input_matrix, ((0, num_neighbors - num_faces), (0, num_neighbors - num_faces)), \
                'constant', constant_values = (0, 0))
            input_features = np.pad(input_features, ((0, num_neighbors - num_faces), (0, 0)), \
                'constant', constant_values = (0, 0))

        input_indices = []
        for i in range(num_neighbors):
            temp_idx = np.array((input_matrix[i] == 1).nonzero()).reshape(-1)
            temp_idx = list(temp_idx)
            if(len(temp_idx) == 0):
                temp_idx = [num_neighbors - 1, num_neighbors - 1, num_neighbors - 1]
            elif(len(temp_idx) == 1):
                temp_idx.append(temp_idx[0])
                temp_idx.append(temp_idx[0])
            elif(len(temp_idx) == 2):
                temp_idx.append(temp_idx[1])

            input_indices.append(temp_idx)

        input_indices = np.array(input_indices)

        inputs = np.c_[input_features, input_indices]
        return inputs

    # Load a single patch file and return its content.
    def loadMAT(self, data_path):
        source_data = sio.loadmat(data_path)
        input_matrix = source_data["MAT"]
        input_matrix = np.array(input_matrix)
        
        input_features = source_data["FEA"]
        input_features = np.array(input_features)
        input_features = input_features.T
        
        gt_norm = source_data["GT"]
        gt_norm = np.array(gt_norm).reshape(-1).astype(np.float32)

        inputs = FileDataset.file2input(input_matrix, input_features, self.num_neighbors)

        return inputs, gt_norm

    # This is used by the DataLoader. If the DataLoader is indexed, this function tells to actually load a file, read it and return it's content.
    def __getitem__(self, index):
        inputs, gt_norm = self.loadMAT(self.data_path[index])
        return inputs, gt_norm

    # Returns a Torch DataLoader object, which can be iterated and used for training or testing purposes.
    def getDataloader(self):
        return Tdata.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=self.is_train, num_workers=self.num_workers, pin_memory=True, drop_last=True)

# This class is used for the forward pass.
# It contains a numpy array of network inputs, instead of files
class PatchDataset(Tdata.Dataset):
    def __init__(self, data_patch, data_rotations, max_batch_size, num_workers=8):
        if type(data_patch) != np.ndarray or len(data_patch.shape) != 3:
            raise ValueError(f"data_patch is not a valid 3-dimensional ndarray! Currently it is {data_patch}")
        if type(data_rotations) != np.ndarray or len(data_rotations.shape) != 3:
            raise ValueError(f"data_rotations is not a valid 3-dimensional ndarray! Currently it is {data_rotations}")
        if data_patch.shape[0] != data_rotations.shape[0]:
            raise ValueError(f"data_patch and data_rotations do not have the same amount of patches that they represent! Currently it is:\npatches: {data_patch.shape[0]}\nrotations: {data_rotations.shape[0]}")
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError(f"max_batch_size should be an integer bigger than zero. Currently it is {max_batch_size}")

        super(PatchDataset).__init__()
        self.batch_size = self.getPreferredBatchSize(len(data_patch), max_batch_size)
        self.num_workers = num_workers

        self.data_patch = data_patch
        self.data_rotations = data_rotations
        self.SIZE = data_patch.shape[0]

    def __len__(self):
        return self.SIZE

    def __getitem__(self, index):
        inputs = self.data_patch[index]
        return inputs
    
    # Calculates the biggest possible batch size, such that
    #   the batch size is a factor of the number of inputs n and
    #   the batch size is not bigger than the maximum batch size m.
    @classmethod
    def getPreferredBatchSize(cls, n, m):
        if not isinstance(n, int) or not isinstance(m, int):
            raise ValueError(f"n and m should be integers to calculate preferred batch size.\nm = {m}\nn = {n}")
        
        r = np.arange(1, int(n**0.5)+1)
        f = n % r == 0
        factors = np.append(r[f], np.flip(n / r[f]))
        result = factors[factors < m]
        return int(np.max(result))

    # Returns a Torch DataLoader object, which can be iterated and used for training or testing purposes.
    def getDataloader(self):
        return Tdata.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=True)

# This class handles selecting folders, splitting datasets and generating FileDatasets.
class DatasetManager():
    def __init__(self, batch_size=256, num_workers=8, num_neighbors=64):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_neighbors = num_neighbors

        self.folder_list = set()
        self.data_path = None
        self.split = None # First element contains dataset size and other elements are indices to validation data points.

        self.is_loaded = False
        self.is_generated = False
        
    # Set a new list of file paths as the new dataset.
    def setDataset(self, dataset):
        if not (type(dataset) == np.ndarray):
            raise ValueError("Only ndarrays and lists can be set as datasets.")
        if not dataset.dtype.type == np.str_:
            raise ValueError("ndarray contains element that is not a string and therefore not a path to a file.")
        if not all([os.path.isfile(x) for x in dataset]):
            raise ValueError("There exists a string in the list, which is not a correct file path.")
        if not all([x.endswith(".mat") for x in dataset]):
            raise ValueError("There exists a string that has a path towards a file that is not a .mat file")
        self.data_path = np.array(dataset)
        self.SIZE = len(dataset)
        self.is_loaded = True
    
    # Check if the folder path exists and is a directory. If so, add it to the list of folders from which a dataset will be created.
    def addFolder(self, folder_path):
        if not type(folder_path) == str:
            raise ValueError("data_folder_path should be a string.")
        if not os.path.isdir(folder_path):
            raise ValueError("data_folder_path should be a path to a directory.")
        self.folder_list.add(folder_path)
        self.is_generated = False

    # Given a folder path, load a folder into the MatrixDataset.
    def generateDatasetFromFolders(self, maxFilesPerFolder=-1):
        dataset = []
        for folder_path in self.folder_list:
            file_paths = [folder_path + "/" + x for x in os.listdir(folder_path) if x.endswith(".mat")]
            if len(file_paths) == 0:
                raise Exception(f"Folder [{folder_path}] is a directory, which doesn't contain a .mat file! (And therefore is not a dataset)")
            elif maxFilesPerFolder > -1 and len(file_paths) > maxFilesPerFolder:
                file_paths = random.sample(file_paths, maxFilesPerFolder)
            dataset.extend(file_paths)
        self.setDataset(np.array(dataset))
        self.is_generated = True

    # Save the current list of data paths to a file to be loaded in later.
    def saveDataset(self, target_path):
        if not type(target_path) == str:
            raise ValueError("target_path is not a string..")
        if not target_path.endswith(".h5"):
            raise ValueError("target_path should end with .h5 to save the database with the correct file extension.")
        if os.path.isfile(target_path):
            raise ValueError("Attempting to write over an existing database.")
        if not self.is_loaded:
            raise Exception("Cannot save a dataset when no dataset is loaded.")

        files_path = self.data_path
        
        # If target path is not a valid path, this will throw an error, which is fine!
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(target_path, 'w') as data_file:
            data = data_file.create_dataset("data_path", files_path.shape, dtype=h5py.special_dtype(vlen=str))
            data[:] = files_path
            data_file.close()
        
    # Load a dataset from a h5 file, which contains file references.
    def loadDataset(self, dataset_path):
        # Check correctness of parameter
        if not type(dataset_path) == str:
            raise ValueError("dataset_path should be a string.")
        if not os.path.isfile(dataset_path):
            raise ValueError("dataset_path should be a path to a file.")
        if not dataset_path.endswith(".h5"):
            raise ValueError("dataset_path should be a path to a .h5 file.")
        
        # Read file
        data_path = h5py.File(dataset_path, 'r')
        data_path = np.array(data_path["data_path"], dtype=np.str_)
        self.setDataset(data_path)
    
    # Split the current dataset by generating indices to include into the training and validation set.
    def splitData(self, val_percentile):
        if not self.is_loaded:
            raise Exception("Cannot split data if there is not dataset loaded!")
        if type(val_percentile) != int and type(val_percentile) != float or val_percentile**2 > 1:
            raise ValueError(f"val_percentile is {val_percentile}, which is not a number between 0 and 1.")

        num_data = self.data_path.shape[0]
        num_batches = int(num_data / self.batch_size)
        num_val_batches = int(num_batches * val_percentile)
        num_train_batches = num_batches - num_val_batches
        num_val_data = num_val_batches * self.batch_size
        # num_train_data = num_train_batches * self.batch_size

        if num_train_batches == 0:
            raise ValueError("Splitting failed: Number of training batches is zero. Batch size is too high [{self.batch_size}] or dataset is too small! [{num_data}]")
        if num_val_batches == 0:
            raise ValueError(f"Splitting failed: Number of validation batches is zero. Batch size is too high [{self.batch_size}] or dataset is too small! [{num_data}]")
        
        self.split = np.insert(random.sample(range(num_data), num_val_data), 0, num_data)
    
    def saveSplit(self, target_path="."):
        if not (type(target_path) == str):
            raise ValueError("Target path to store the split is not a string.")
        if os.path.isfile(target_path):
            raise ValueError("Target path contains a path to an existing file.")
        if not (target_path.endswith(".npy")):
            raise ValueError("Target path does not end with .npy and therefore is not a valid path to save file.")
        
        # If target path is not a valid path, this will throw an error, which is fine!
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        
        np.save(target_path, self.split)
    
    def loadSplit(self, target_path):
        if not self.is_loaded:
            raise Exception("Cannot load a split if there is no dataset loaded!")
        if not (type(target_path) == str):
            raise ValueError("Target path to load the split is not a string.")
        if not (os.path.isfile(target_path)):
            raise ValueError("Target path contains a path to an existing file.")
        if not (target_path.endswith(".npy")):
            raise ValueError("Target path does not end with .npy and therefore is not a valid path to save file.")
        
        loaded = np.load(target_path)

        if len(loaded.shape) != 1 or loaded.dtype != np.int_:
            raise Exception("Loaded split array is not one dimensional or doesn't contain integers exclusively.")
        if loaded[0] != self.data_path.shape[0]:
            raise Exception(f"Loaded split array is made for a dataset of size {loaded[0]}, but current dataset has size {self.data_path.shape[0]}")
        if loaded[0] != np.max(loaded):
            raise Exception("Loaded split array does not have the right structure. If the first integer is not the highest integer, the first integer is not the size of the dataset.")
        
        self.split = loaded

    def getTrainingSet(self):
        if not self.is_loaded:
            raise Exception("Cannot get training set if no dataset is loaded.")
        if self.split is None:
            raise Exception("No split is present, therefore no training set can be returned.")
        
        train_index = np.delete(np.arange(self.data_path.shape[0]), np.delete(self.split, 0))
        train_path = self.data_path[train_index]

        dataset = FileDataset(train_path, self.batch_size, self.num_workers, self.num_neighbors, is_train=True)
        return dataset.getDataloader()

    def getValidationSet(self):
        if not self.is_loaded:
            raise Exception("Cannot get validation set if no dataset is loaded.")
        if self.split is None:
            raise Exception("No split is present, therefore no validation set can be returned.")
        
        val_index = np.delete(self.split, 0)
        val_path = self.data_path[val_index]

        dataset = FileDataset(val_path, self.batch_size, self.num_workers, self.num_neighbors, is_train=False)
        return dataset.getDataloader()