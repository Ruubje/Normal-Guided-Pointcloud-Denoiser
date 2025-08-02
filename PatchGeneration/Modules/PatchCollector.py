import numpy as np
from .Mesh import *
import time
from .Network.DataUtils import *
import igl

# PatchCollector is a class that creates and keeps track of patches from a mesh.
class PatchCollector:

    PATCH_DIR_NAME = "Patches"
    
    # Collectors are initialized with the mesh they are working on.
    # mesh is a Mesh object that represents the mesh model that we are working on.
    # k is an optional parameter that increases that patch size manually.
    def __init__(self, mesh, k=4):
        if not isinstance(mesh, Mesh) or not isinstance(k, int):
            raise ValueError("mesh should be a Mesh and k should be an int")
        
        self.k = k
        self.mesh = mesh
        self.obj_path = None

    def check_obj(self, obj_path=None):
        obj_path = self.obj_path if obj_path is None else obj_path
        if obj_path is None or not os.path.isfile(obj_path) or not obj_path.endswith(".obj"):
            raise Exception("obj_path has not been set to a save location yet or the path does not point to an existing .obj file.")

    def getDir(self):
        self.check_obj()
        parent_dir = Path(self.obj_path).parent
        patches_dir = str(parent_dir).replace('\\', '/') + "/" + PatchCollector.PATCH_DIR_NAME
        if not os.path.isdir(patches_dir):
            raise Exception(f"Directory where patches should be stored does not exist: {patches_dir}")
        return patches_dir
    
    def getFiles(self):
        dir = self.getDir()
        
        obj_name = Path(self.obj_path).stem
        filesnames_in_dir = os.listdir(dir)
        files_in_dir = np.array([dir + "/" + x for x in filesnames_in_dir if Path(x).stem[:x.rfind('_')] == obj_name and x.endswith(".mat")])

        return files_in_dir
    
    def getPatchIndices(self):
        files = self.getFiles()
        face_indices_in_dir = np.array([int(x[(x.rfind('_')+1):-4]) for x in files])
        return face_indices_in_dir, files

    def setSavePath(self, obj_path):
        self.check_obj(obj_path)
        self.obj_path = obj_path

    def saveMesh(self):
        self.check_obj()
        igl.write_obj(self.obj_file, self.mesh.getVertices(), self.mesh.f)

    # Reading an obj file directly into a PatchCollector object.
    # obj_file is a String representing the path towards an Object (.obj) file.
    # Returns a new PatchCollector object
    @classmethod
    def loadMesh(cls, obj_path, k=4):
        mesh = Mesh.readFile(obj_path)
        pc = PatchCollector(mesh, k)
        pc.obj_path = obj_path
        return pc
    
    # Sets the gt attribute of the mesh from which patches are collected.
    # path is an optional argument, where you can set the path to an .obj object which contains the ground truth.
    def setGT(self, path=None):
        if not (path is None) and (not isinstance(path, str) or not path.endswith(".obj") or not os.path.isfile(path)):
            raise ValueError("path must either be None or a path to a .obj object.")
        
        path = path if not (path is None) else self.getGT()
        self.check_obj(path)
        gt_mesh = Mesh.readFile(path)
        self.mesh.gt = gt_mesh.getFaceNormals()
    
    # Based on the current obj_path, the ground truth of the noisy mesh should be stored in an .obj file in the parent directory and have the same name without patch index.
    # Example directory:
    # Example.obj
    # Noise --
    #        | Example_1.obj
    def getGT(self):
        self.check_obj()
        parent_dir = str(Path(self.obj_path).parent.parent).replace('\\', '/')
        obj_name = Path(self.obj_path).stem
        gt = parent_dir + "/" + obj_name[:obj_name.rfind('_')] + ".obj"
        print(parent_dir, obj_name, gt)
        return gt
    
    # This method loops over List<(Index, Patch)> and converts all patches individually to a graph representation and stores them using Patch.save()
    # patches is a list of tuple pairs representing the index of the face and the corresponding Patch.
    def savePatches(self, patches):
        if not hasattr(patches, '__iter__') or not all((isinstance(x, tuple), isinstance(x[0], int), isinstance(x[1], Patch)) for x in patches):
            raise ValueError("patches is not iterable or there exists an element in iterable that is not a Patch.")
        
        self.check_obj()
        parent_dir = Path(self.obj_path).parent
        obj_basename = os.path.basename(Path(self.obj_path))[:-4]
        patches_dir = str(parent_dir).replace('\\', '/') + "/" + PatchCollector.PATCH_DIR_NAME
        Path(patches_dir).mkdir(parents=True, exist_ok=True)

        for patch_tuple in patches:
            file_path = patches_dir + "/" + f"{obj_basename}_{patch_tuple[0]}.mat"
            patch_tuple[1].save(file_path)
    
    # This method looks for patch files to be loaded in a FileDataset based on the given face indices.
    # From this FileDataset, the files can actually be loaded in the program to a graph representation.
    # WARNING: Converting graph representations of patches to Patch objects is not supported.
    def loadPatches(self, faces, batch_size, num_workers, num_neighbors):
        if not hasattr(faces, "__iter__") or not all(isinstance(x, (int, np.integer)) for x in faces):
            raise ValueError("faces is not iterable or contains an element that is not an integer.")

        face_indices_in_dir, files_in_dir = self.getPatchIndices()
        missing_indices = np.setdiff1d(faces, face_indices_in_dir)

        if missing_indices.shape[0] > 0:
            raise Exception(f"Patches for the following face indices have not been found: {missing_indices}")
        
        files_to_index = np.any(face_indices_in_dir[:, None] == np.array(faces)[None], axis=1)
        files = np.array(files_in_dir[files_to_index])

        return FileDataset(files, batch_size, num_workers, num_neighbors)

    def collectMissing(self, patches, timeout=-1):
        current_indices, _ = self.getPatchIndices()
        missing_indices = np.setdiff1d(patches, current_indices)
        missing_patches = self.collectPatches(missing_indices, timeout)
        return missing_patches

    # Collecting all patches from the mesh. First selecting and storing all patches and then aligning them
    # timeout is a time in seconds. If the timeout time is reached the method stops collecting and starts aligning the collected patches.
    # Returns the aligned patches (array of Patch objects) (within the time limit).
    def collectPatches(self, patches, timeout=-1):
        if not type(patches) == np.ndarray or not len(patches.shape) == 1 or not np.all(patches < len(self.mesh.f)):
            raise ValueError(f"Patches should be an numpy array with 1 dimension and able to index faces of the set mesh. Currently it is: {patches}")
        
        start_time = time.time()
        numberOfFaces = len(patches)
        print("Start selecting patches")
        selectedPatches = []
        for i in range(numberOfFaces):
            patch_tuple = (patches[i], self.mesh.selectPaperPatch(patches[i], self.k))
            selectedPatches.append(patch_tuple)
            msg = "Patch " + str(i+1) + "/" + str(numberOfFaces) + " selected!"
            time_since_start = int(time.time() - start_time)
            if timeout > -1:
                msg = f"[Timeout: {time_since_start}/{timeout}] " + msg
            print(msg)
            if time_since_start >= timeout and timeout > -1:
                break
        numberOfSelectedPatches = len(selectedPatches)
        for i, patch in enumerate(selectedPatches):
            patch[1].alignPatch()
            msg = "Patch " + str(i+1) + "/" + str(numberOfSelectedPatches) + " aligned!"
            print(msg)
        return selectedPatches
    
    # Collects all patches and transforms them into input for a DGCNN.
    def collectNetworkInput(self, max_batch_size, timeout=-1):
        patches = self.collectPatches(np.arange(len(self.mesh.f)), timeout)
        fileformat = np.array([FileDataset.file2input(*patch[1].toGraph(), 64) for patch in patches])
        rotations = np.array([p[1].lastRotationApplied for p in patches])
        pd = PatchDataset(fileformat, rotations, max_batch_size)
        return pd

class NoiseGenerator:
    NOISE_DIR_PATH = "Noise"

    def __init__(self, obj_path):
        self.mesh = Mesh.readFile(obj_path)
        self.obj_path = obj_path
        self.noise_dir = str(Path(obj_path).parent).replace('\\', '/') + "/" + NoiseGenerator.NOISE_DIR_PATH
        os.makedirs(self.noise_dir, exist_ok=True)

    def saveNoisyMeshes(self, meshes, mode=0):
        if not hasattr(meshes, "__iter__") or not all(isinstance(x, Mesh) for x in meshes):
            raise ValueError("noise_levels should be an iterable containing meshes.")

        save_paths = [f"{self.noise_dir}/{os.path.basename(self.obj_path)[:-4]}_{int(mesh.noise_factor * 10)}.obj" for mesh in meshes]
        for t in zip(meshes, save_paths):
            t[0].writeFile(t[1], mode)
        
        return save_paths

    def generateNoise(self, noise_factors):
        if not hasattr(noise_factors, "__iter__") or not all(isinstance(x, float) for x in noise_factors):
            raise ValueError("noise_levels should be an iterable containing integers.")
        
        return [self.mesh.applyGaussianNoise(factor) for factor in noise_factors]