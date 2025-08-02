import igl
import numpy as np

#A class that can handle all method and algorithms that are relevant to rotatino matrices.
class RotationMatrix():
    
    # Get the rotation matrix for this patch. The algorithm for defining the matrix is described in the paper.
    # Returns a 3x3 array containing the rotation matrix.
    def __init__(self, patch):
        # This sigma should be changed in the future maybe! This is the best guess for what sigma should be currently..
        # Proof has been found in the code that sigma should indeed be a third!
        SIGMA = 1./3.
        # Can only execute if attribute pi is set with a face id.
        self.bcs = igl.barycenter(patch.v, patch.f)
        self.ci = self.bcs[patch.pi]
        self.cj = np.delete(self.bcs, patch.pi, axis=0)
        self.dcs = self.cj - self.ci
        self.nj = np.delete(patch.getFaceNormals(), patch.pi, axis=0)
        self.raw_wj = np.cross(np.cross(self.dcs, self.nj, axis=1), self.dcs)
        self.wj = np.nan_to_num(self.raw_wj / np.linalg.norm(self.raw_wj, axis=1, keepdims=True))
        self.njprime = 2 * np.sum(np.multiply(self.nj, self.wj), axis=1)[:, None] * self.wj - self.nj
        self.areas = patch.getAreas(np.delete(np.arange(len(patch.f)), patch.pi, axis=0))
        self.maxArea = np.max(self.areas)
        self.ddcs = np.linalg.norm(self.dcs, axis=1)
        self.muj = (self.areas / self.maxArea)*np.exp(-self.ddcs/SIGMA)
        self.outer = self.njprime[..., None] * self.njprime[:, None]
        self.Tj = self.muj[:, None, None] * self.outer
        self.Ti = np.sum(self.Tj, axis=0)
        self.eig = np.linalg.eigh(self.Ti)
        self.sort = np.flip(np.argsort(self.eig[0]))
        self.matrix = self.eig[1].T[self.sort]
        if np.sum(self.matrix[0] * patch.getFaceNormals()[patch.pi]) < 0:
            self.matrix[0, :] *= -1
        if np.linalg.det(self.matrix) < 0:
            self.matrix[2, :] *= -1
    
    # Produce a random rotation matrix. Specifically used by tests to test the methods.
    # Returns a 3 by 3 matrix representing a random rotation.
    @classmethod
    def getRandomRotationMatrix(cls):
        random_locations = np.random.normal(size=(3, 3))
        normalized = random_locations / np.linalg.norm(random_locations, axis=1)[:, None]
        proj_1_on_0 = np.dot(normalized[1], normalized[0])*normalized[0]
        second_axis = normalized[1] - proj_1_on_0
        normalized_1 = second_axis / np.linalg.norm(second_axis)
        proj_2_on_0 = np.dot(normalized[2], normalized[0])*normalized[0]
        proj_2_on_1 = np.dot(normalized[2], normalized_1)*normalized_1
        third_axis = normalized[2] - proj_2_on_0 - proj_2_on_1
        normalized_2 = third_axis / np.linalg.norm(third_axis)
        rotation = np.stack([normalized[0], normalized_1, normalized_2])
        if np.linalg.det(rotation) < 0:
            rotation[2, :] *= -1
        return rotation