import igl
import polyscope as ps
import numpy as np
import meshplot as mp
import copy
import scipy.io as sio
import os
import errno

from .RotationMatrix import RotationMatrix as rm

# This class contains methods to manipulate a mesh with vertices and faces.
class Mesh:
    # Initialize the mesh object.
    # v is a list of vertex positions.
    # n is a list of normal vectors.
    # f is a list of indices connecting vertices to create a face.
    # fn is a list of indices connecting vertices to create a face normal???
    # f2f is a map created by igl showing which face is neighbouring which face.
    #   It is only initialized if neighbour look ups are needed.
    def __init__(self, v, f, noise_factor = 0, f2f = None, vta = None, gt=None):
        self.v = v
        self.f = f
        self.noise_factor = noise_factor
        self.f2f = f2f if not (f2f is None) else igl.triangle_triangle_adjacency(f)[0]
        self.vta = vta if not (vta is None) else igl.vertex_triangle_adjacency(f, len(v))
        self.gt = gt

    """
    Class methods
    Contains functionality for when there is no mesh information yet or when there are multiple meshes.
    """

    # View multiple meshes in one visualization as point clouds.
    # *meshes are multiple meshes that need to be shown in one plot.
    # Shows a plot using Meshplot
    @classmethod
    def mpViewPCs(cls, *meshes):
        vertices = None
        color = None
        for i, mesh in enumerate(meshes):
            vertices = mesh.getVertices() if i==0 else np.append(vertices, mesh.getVertices(), axis=0)
            tiledIndex = np.tile(i, (mesh.getVertices().shape[0], 1))
            color = tiledIndex if i==0 else np.append(color, tiledIndex, axis=0)
        return mp.plot(vertices, c=color, shading={"point_size": np.max(np.linalg.norm(vertices))/1000})
    
    # View multiple meshes in one visualization as meshes.
    # *meshes are multiple meshes that need to be shown in one plot.
    # Shows a plot using Meshplot
    @classmethod
    def mpViewMeshes(cls, *meshes):
        vertices = None
        faces = None
        color = None
        for i, mesh in enumerate(meshes):
            vertices = mesh.getVertices() if i==0 else np.append(vertices, mesh.getVertices(), axis=0)
            faces = mesh.f if i==0 else np.append(faces, mesh.f + np.tile(vertices.shape[0] - mesh.getVertices().shape[0], mesh.f.shape), axis=0)
            tiledIndex = np.tile(i, (mesh.getVertices().shape[0], 1))
            color = tiledIndex if i==0 else np.append(color, tiledIndex, axis=0)
        return mp.plot(vertices, faces, c=color, shading={"wireframe": True})

    # Reads an object file and initializes a new Mesh object with it.
    # obj_file is the file path to open.
    # Returns a new Mesh object containing vertex, normal and face information
    @classmethod
    def readFile(cls, obj_file):
        if not type(obj_file) == str:
            raise ValueError("obj_file (first argument) must be a string representing the path towards the object file.")
        if not obj_file.endswith(".obj"):
            raise ValueError("obj_file (first argument) must be a path towards and object file ending with '.obj'")
        if not os.path.exists(obj_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), obj_file)
        v, _, _, f, _, _ = igl.read_obj(obj_file)
        return Mesh(v, f)

    """
    Write methods that change files.
    """

    # obj_file is that path to save the file.
    # modes: If file location already exists:
    #   {0: Raise Exception 1: Skip 2: Overwrite}
    def writeFile(self, obj_file, mode=0):
        if not type(obj_file) == str:
            raise ValueError("obj_file (first argument) must be a string representing the path towards the object file.")
        if not obj_file.endswith(".obj"):
            raise ValueError("obj_file (first argument) must be a path towards and object file ending with '.obj'")
        if not isinstance(mode, int) or ((mode - 1) / 2)**2 > 1:
            raise ValueError("mode should be the integer 0, 1 or 2")
        if os.path.exists(obj_file):
            if mode == 0:
                raise ValueError(f"obj_file (first argument) points to an existing file: {obj_file}")
            if mode == 1:
                print(f"Warning! Saving mesh to {obj_file} has been skipped, because file already existed.")
        else:
            igl.write_obj(obj_file, self.getVertices(), self.f)

    """
    Get methods
    Function to get information from the mesh.
    """

    # Returns the vertices of the mesh.
    def getVertices(self):
        return self.v
    
    # Getter method for the face normals.
    # If they are not calculated yet, they are calculated upon calling this method.
    # Returns array of face normals.
    def getFaceNormals(self):
        faceVertices = self.getVertices()[self.f]
        crosses = np.cross(faceVertices[:, 1, :] - faceVertices[:, 0, :], faceVertices[:, 2, :] - faceVertices[:, 1, :])
        faceNormals = crosses / np.linalg.norm(crosses, axis=1)[:, None]
        return faceNormals
    
    # Getter method for getting the IGL vertex-triangle adjacency.
    # Sets the vta attribute.
    # Returns the adjacency matrix.
    def getVertexTriangleAdjacency(self):
            return self.vta

    # Calculates the center of the mesh by averaging all vertex positions.
    # Returns the center of the mesh.
    def getPCCenter(self):
        num_vertices = self.getVertices().shape[0]
        center = np.sum(self.getVertices(), axis=0)/num_vertices
        return center

    # Calculates the size of the mesh.
    # The size is defined as the vertex with the maximum distance from the center of the mesh.
    # Returns 3x1 array, center of the mesh.
    def getPCSize(self):
        center = self.getPCCenter()
        return np.max(np.linalg.norm(self.getVertices() - np.tile(center, (self.getVertices().shape[0], 1)), axis=1))
    
    # Calculates the bounding size in which the Mesh perfectly fits.
    # Returns 3x1 array containing the size of the bounding box.
    def getPCBoundingBox(self):
        return np.max(self.getVertices(), axis=0) - np.min(self.getVertices(), axis=0)

    # Calculates the areas of the given faces.
    # faces is an array with indices to faces from which the areas need to be calculated.
    # Returns #facesx1 array with the areas corresponding to the face indices.
    def getAreas(self, faces):
        triangles = self.getVertices()[self.f[faces.astype(int)]]
        As = triangles[...,1,:] - triangles[...,0,:]
        Bs = triangles[...,2,:] - triangles[...,0,:]
        cross = np.cross(As, Bs, axis=-1)
        areas = 0.5*np.linalg.norm(cross, axis=-1)
        return areas

    # Retrieve an array of face indices that contain vertices that are within range.
    # This method is described in the paper.
    # center is a 3x1 array representing the center of the sphere.
    # range is a number representing the radius of the sphere.
    # Returns an 1D array containing face indices of all faces that contain a vertex that is within the sphere.
    def getFacesInRange(self, center, range):
        translated = self.getVertices() - center
        dist = np.linalg.norm(translated, axis=1)
        vm = dist < range
        vertices = np.arange(vm.shape[0])[vm]
        faces = self.getTrianglesOfVertices(vertices)
        return faces

    # This definition is introduced to replace getFacesInRange.
    # Even though getFacesInRange is technically the truth, it's very slow..
    # Instead of adding all faces in range to the set of faces that gets returned,
    #    Only faces that also attach to the given face get returned.
    # Therefore, the center argument is replaced by a face_index.
    # The center of the face_index is used for range calculations.
    def getNeighbourhoodInRange(self, center_face_index, range):
        v = self.getVertices()
        center = v[self.f[center_face_index]]
        faces_in_range = np.array([center_face_index])

        faces_to_check = np.array([center_face_index])
        while len(faces_to_check) > 0:
            adjacent_faces = self.f2f[faces_to_check]
            # Remove faces that are already checked / in range
            unique_adjacent_faces = np.setdiff1d(adjacent_faces, faces_in_range)
            # Remove faces that are out of range
            dist = np.linalg.norm(v[self.f[unique_adjacent_faces]] - center, axis=2)
            faces_to_append = np.delete(unique_adjacent_faces, np.where(np.any(dist > range, axis=1)))
            faces_in_range = np.append(faces_in_range, faces_to_append)
            faces_to_check = faces_to_append

        return faces_in_range
    
    # Gets the neighbouring faces as described by the paper.
    # face_index is the index of the face which is considered.
    # ring is a number (n) representing the size of the neighbourhood that needs to be returned.
    # Returns an array containing all faces within the n-ring neighbourhood.
    def getNeighbourhood(self, face_index, ring):
        if ring < 1:
            return np.array([])
        
        adjecancy = self.f2f

        result = np.array(face_index)
        for _ in range(ring):
            nb = adjecancy[result]
            result = np.union1d(result, nb)

        return np.array(list(result))
    
    # Input is a 2d array
    # Output is the unique operator, but with original array shape and filled with nan.
    def rowwise_unique(x):
        unique = np.unique(x)
        correct_size = np.empty(x.shape)
        correct_size.fill(np.nan)
        correct_size[:len(unique)] = unique
        return correct_size

    def getNeighbourhoods(self, fis, ring):
        # (face_index, edge_neighbor_index) -> face_index
        f2f = self.f2f
        l = len(fis)

        # (face_index, ring_neighbor_index) -> face_index
        result = np.array(fis).reshape((l, 1))
        for _ in range(ring):
            # (face_index, found_neighbors_index) -> face_index
            nb = f2f[result.astype(int)].reshape((l, -1))
            concat = np.concatenate((result, nb), axis=1)
            unique = np.apply_along_axis(self.rowwise_unique, 1, concat)
            result = unique[:, ~np.all(np.isnan(unique), axis=0)]

        return result
    
    # Get the triangles which contain the given vertex index.
    # v is the vertex index.
    # Returns an array of face indices which contain the vertex v.
    def getTrianglesOfVertex(self, v):
        vta = self.getVertexTriangleAdjacency()
        triangles = vta[0][vta[1][v]:vta[1][v+1]]
        return triangles
    
    # Get the triangles that are connected to the given vertex indices.
    # vs is an array with vertex indices for which neighbouring triangles need to be found.
    # Returns an array of face indices that are connected to one of the given vertices.
    def getTrianglesOfVertices(self, vs):
        vta = self.getVertexTriangleAdjacency()
        start = vta[1][vs]
        stop = vta[1][vs+1]
        ranges = np.frompyfunc(np.arange, 2, 1)(start, stop)
        uniqueFaces = set()
        for range in ranges:
            uniqueFaces.update(vta[0][range])
        triangles = np.sort(np.array(list(uniqueFaces)))
        return triangles
    
    # Get the average edge length of the mesh.
    def getAverageEdgeLength(self):
        return igl.avg_edge_length(self.getVertices(), self.f)

    # Gets the new index of a face after patch selection of a given old face index.
    # old_fi is the old face index.
    # imv is the old to new vertex map that is used to calculate the new face index.
    # Returns the new face index
    def getNewFaceIndex(self, old_fi, nf, imv):
        old_vertices_of_face = self.f[old_fi]
        new_vertices_of_face = imv[old_vertices_of_face]
        mask_where_new_vertices_are_equal_to_new_vertices_of_face = np.equal(nf, new_vertices_of_face)
        mask_where_all_vertices_are_equal = np.all(mask_where_new_vertices_are_equal_to_new_vertices_of_face, axis=1)
        resulting_index = np.arange(len(nf))[mask_where_all_vertices_are_equal]
        return -1 if len(resulting_index) == 0 else resulting_index
    
    # TODO Implement this function! (No need because it's older version though)
    def getAngularError(self, other_mesh):
        pass
        if not isinstance(other_mesh, Mesh) or other_mesh.f.shape == self.f.shape:
            raise ValueError("The angular error can only be calculated between this Mesh and another Mesh with the same amount of faces.")
        
        my_normals = self.getFaceNormals()
        other_normals = other_mesh.getFaceNormals()


    """
    Select faces by mask (array with trues and falses) or id (face ids to keep)
    The methods will return a new mesh object to keep the current data in tact to use for other purposes
    The methods will throw away faces and vertices that are not included in the face selection
    """

    # Creates a new Mesh object containing only information about the selected face indices.
    # faceIds is an array containing face indices to include in the new Mesh.
    # Returns newMesh is the new Mesh object
    #         imv is a map that points the old vertex index to the new vertex index.
    def createPatch(self, faceIds, old_fi):
        nv, nf, imv, _ = igl.remove_unreferenced(self.getVertices(), self.f[faceIds])
        new_fi = self.getNewFaceIndex(old_fi, nf, imv)
        newMesh = Patch(self, new_fi, nv, nf)
        newMesh.noise_factor = self.noise_factor
        newMesh.gt = self.gt[faceIds]
        return newMesh
    
    # This method returns a Mesh object that contains the patch corresponding to the face.
    # fi is the face index from which we want to create a patch.
    # Returns a new Mesh object containing the selected patch.
    def selectPaperPatch(self, fi, k):
        nh = self.getNeighbourhood(fi, 2)
        avg_tworing_area = np.sum(self.getAreas(nh)) / nh.shape[0]
        r = k * avg_tworing_area ** 0.5
        center = np.average(self.getVertices()[self.f[fi]], axis=0)
        faces_in_range = self.getFacesInRange(center, r)
        patch = self.createPatch(faces_in_range, fi)
        return patch

    # Creates a copy of the current mesh. Can be used for debugging purposes.
    # Returns a new Mesh object that is a copy of the current object.
    def copy(self):
        v_copy = copy.deepcopy(self.v)
        f_copy = copy.deepcopy(self.f)
        noise_factor_copy = self.noise_factor # This is not numpy and therefore doesn't need to be deepcopied!
        f2f_copy = copy.deepcopy(self.f2f)
        vta_copy = copy.deepcopy(self.vta)
        gt_copy = copy.deepcopy(self.gt)
        return Mesh(v_copy, f_copy, noise_factor_copy, f2f_copy, vta_copy, gt_copy)
    
    # Applying Gaussian noise on the current Mesh and saving it in a seperate attribute.
    # Returns the noisy vertices and normals that just have been created.
    def applyGaussianNoise(self, factor=0.1):
        v = self.getVertices()
        random_sample = np.random.normal(size=(v.shape[0], 3))
        random_direction = random_sample / np.linalg.norm(random_sample, axis=1)[:, None]
        stdev = self.getAverageEdgeLength()*factor
        random_gaussian_sample = np.random.normal(0, stdev, (v.shape[0], 1))
        noise = random_direction * np.tile(random_gaussian_sample, (1, 3))
        new_mesh = self.copy()
        new_mesh.v += noise
        new_mesh.noise_factor = factor
        new_mesh.gt = self.getFaceNormals()
        return new_mesh

    """
    Transformation methods
    These have functionality that transform the mesh.
    They return themselves, so that multiple transformations can be done in one line of code.
    """

    # Changes the model by translating / moving it in a certain direction.
    # translation is an 3x1 array representing the relative distance to move the mesh.
    # Returns the object itself
    def translate(self, translation):
        tiled_translation = np.tile(translation, (self.getVertices().shape[0], 1))
        self.v += tiled_translation
        return self

    # Resizes the mesh to the given size. Size is defined as the vertex with the maximum distance to the center of the mesh.
    # size is the target size to scale to.
    # Returns the object itself.
    def resize(self, size):
        center = self.getPCCenter()
        self.translate(-center)
        current_size = np.max(np.linalg.norm(self.getVertices(), axis=1))
        scale = size / current_size
        self.v *= scale
        self.translate(center)
        return self

    # Rotates the mesh with a rotation matrix. This is basicly a matrix multiplication with all vertices.
    # rotationMatrix is a 3x3 array representing a rotation matrix.
    # Returns the object itself.
    def rotate(self, rotationMatrix):
        v = self.getVertices()
        center = self.getPCCenter()
        self.translate(-center)
        rotated_v = np.dot(rotationMatrix, v.T).T
        self.v = rotated_v
        self.translate(center)
        return self
    
    # Update the vertices of the mesh such that the normals of the faces align with the given normals.
    # n is an array with normals for every face.
    # k is the number of times the algorithm is executed.
    # This method is tested in a notebook!
    def updateVertices(self, n, k=15):
        # Vertices
        v = self.getVertices()
        # Faces of mesh
        f = self.f
        # Vertex Triangle Adjacency
        vta = self.getVertexTriangleAdjacency()
        # Repeat algorithm k times
        for _ in range(k):
            _V = len(v)
            # (vi,)
            vis = np.arange(_V)
            starts = np.delete(vta[1], -1)
            stops = np.delete(vta[1], 0)
            degrees = stops - starts
            MAX_DEGREE = np.max(degrees)
            # 1D array of 1D array ranges
            ranges = np.frompyfunc(np.arange, 2, 1)(starts, stops)
            # ranges (vi, fm) where fm means max degree of all vertices
            ranges_padded = np.stack([np.pad(r, (0, MAX_DEGREE - len(r)), 'constant', constant_values=(-1, -1)) for r in ranges])
            mask_is_padded = ranges_padded == -1
            # faces: (vi, fm)
            faces = vta[0][ranges_padded]
            # vks = (vi, fm, vj)
            vks = f[faces]
            # dvs = (vi, fm, vj, ax)
            dvs = v[vks] - np.tile(v[vis][:, None, None, :], (1, MAX_DEGREE, 3, 1)) # One of these is gonna be vi - vi = 0
            # nj = (vi, fm, ax)
            nj = n[faces]
            # dot = (vi, fm, vj) = (vi, fm, empty, ax) * (vi, fm, vj, ax)
            dot = np.sum(np.tile(nj[:, :, None, :], (1, 1, 3, 1)) * dvs, axis=3)
            # elements are the elements to be summed for this face.
            # elements = (vi, fm, vj, ax) = (vi, fm, vj, empty) * (vi, fm, empty, ax)
            elements = dot[..., None] * nj[:, :, None]
            # Set padded entries back to 0, such that they don't influence vertex positions
            elements[mask_is_padded] = 0
            # Sum over fm and vj, S = (vi, ax)
            S = np.sum(np.sum(elements, axis=1), axis=1)
            # Divide sum by number of neighbouring faces and add the resulting offset to the vertex.
            # Scaled (vi, ax) = S (vi, ax) / degrees (vi,)
            scaled_S = S / (3 * np.tile(degrees[:, None], (1, 3)))
            v += scaled_S

    """
    Show methods
    Shows visualizations of the mesh.
    """

    # Visualizes the mesh as a point cloud using meshplot.
    def mpShowPC(self):
        return mp.plot(self.getVertices(), shading={"point_size": np.max(np.linalg.norm(self.v))/1000})
    
    # Visualizes the mesh a mesh using meshplot.
    def mpShowMesh(self):
        return mp.plot(self.getVertices(), self.f, shading={"wireframe": True})
    
    # Visualize the vertices as a point cloud using polyscope.
    def psViewPC(self):
        ps.init()
        ps.register_point_cloud("main", self.getVertices())
        ps.show()

    # Visualize the vertices, edges and faces as a mesh using polyscope.
    def psViewMesh(self):
        ps.init()
        ps.register_surface_mesh("main", self.getVertices(), self.f)
        ps.show()

# A patch is a part of a Mesh that is selected to do further calculations on.
class Patch(Mesh):

    # Initialize the patch.
    # pi is the patch index, representing an index pointing to a face from which the patch was created.
    def __init__(self, parent, pi, *args):
        super().__init__(*args)
        self.parent = parent
        self.pi = pi
        self.lastRotationApplied = None
    
    # Creates a copy of the current patch. Can be used for debugging purposes.
    # Returns a new Patch object that is a copy of the current object.
    def copy(self):
        parent_copy = self.parent # This variable is a reference and should therefore not be copied!
        pi_copy = self.pi # This variable is a reference and should therefore not be copied!
        lr_copy = copy.deepcopy(self.lastRotationApplied)
        v_copy = copy.deepcopy(self.v)
        f_copy = copy.deepcopy(self.f)
        noise_factor_copy = self.noise_factor # This is not numpy and therefore doesn't need to be deepcopied!
        f2f_copy = copy.deepcopy(self.f2f)
        vta_copy = copy.deepcopy(self.vta)
        gt_copy = copy.deepcopy(self.gt)
        new_patch = Patch(parent_copy, pi_copy, v_copy, f_copy, noise_factor_copy, f2f_copy, vta_copy, gt_copy)
        new_patch.lastRotationApplied = lr_copy
        return new_patch
    
    # Transforms / Aligns the patch as described in the paper.
    def alignPatch(self):
        center = self.getPCCenter()
        size = self.getPCSize()
        if not np.allclose(np.linalg.norm(center), 0):
            self.translate(-center)
        if not np.allclose(size, 1):
            self.resize(1)
        rotationMatrix = self.getPaperRotationMatrix().matrix
        self.rotate(rotationMatrix)
        self.lastRotationApplied = rotationMatrix
        # WARNING, Apparently rotation twice seems to give more accuracy.
        # For now this is fine, but these lines should later be removed, because the rotation should be applied once.
        # WARNING, I've removed the double use of this method. The method still needs to be improved later, but for now, I need a single rotation matrix, so the second rotation matrix is removed.
        # rotationMatrix = self.getPaperRotationMatrix().matrix
        # self.rotate(rotationMatrix)
    
    # Get the rotation matrix for this patch. The algorithm for defining the matrix is described in the paper.
    # Returns a 3x3 array containing the rotation matrix.
    def getPaperRotationMatrix(self):
        rotation = rm(self)
        return rotation
    
    # The patch can be converted to a graph representation with this method.
    # Returns an adjacency matrix and a feature vector per graph node.
    def toGraph(self):
        v = self.getVertices()
        E = igl.triangle_triangle_adjacency(self.f)[0]
        centers = igl.barycenter(v, self.f)
        normals = self.getFaceNormals()
        areas = self.getAreas(np.arange(self.f.shape[0])).reshape(-1, 1)
        neighbors = (E != -1).sum(axis=1).reshape(-1, 1)
        point_positions = v[self.f].reshape(-1, 9)
        V = np.concatenate((centers, normals, areas, neighbors, point_positions), axis=1)
        return E, V

    # Saves the Patch to a file, which can later be imported and used by the GCN.
    # file_path is the path towards the file where the patch needs to be saved.
    def save(self, file_path):
        if not type(file_path) == str:
            raise ValueError("file_path (first argument) must be a string representing the path which to save the patch to.")
        if not file_path.endswith(".mat"):
            raise ValueError("file_path (first argument) must be a path ending with '.mat', to save the patch as a '.mat' file.")
        if self.gt is None:
            raise ValueError("Cannot save the patch as a file, because no ground truth has been set.")

        MAT, FEA = self.toGraph()
        GT = self.gt[self.pi].reshape(3, 1) # The normal of the middle face (ground truth)
        ROT = self.lastRotationApplied

        file_dictionary = {
            "MAT": MAT,
            "FEA": FEA.T,
            "GT": GT,
            "ROT": ROT
        }

        sio.savemat(file_path, file_dictionary)

if __name__ == "__main__":
    myMesh = Mesh.readFile('MyProject/new_saved_fandisk.obj')
    patch = myMesh.getPatch(myMesh.getVertices()[0], 10)
    myMesh.ViewPC(patch.getVertices())