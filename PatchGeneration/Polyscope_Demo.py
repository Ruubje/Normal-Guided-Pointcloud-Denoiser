import polyscope as ps
import pywavefront as pwf
import numpy as np

scene = pwf.Wavefront("models/fandisk.obj", collect_faces=True)

# Initialize polyscope
ps.init()

### Register a point cloud
# `my_points` is a Nx3 numpy array
# ps.register_point_cloud("Armadillo", np.array(scene.vertices))

### Register a mesh
# `verts` is a Nx3 numpy array of vertex positions
# `faces` is a Fx3 array of indices, or a nested list5
ps.register_surface_mesh("Armadillo", np.array(scene.vertices), np.array(list(scene.meshes.values())[0].faces), smooth_shade=True)

# Add a scalar function and a vector function defined on the mesh
# vertex_scalar is a length V numpy array of values
# face_vectors is an Fx3 array of vectors per face
vertex_values = np.array(scene.vertices)[:,1] / np.array(scene.vertices)[:,1].max(axis=0)
face_vectors = np.array(list(scene.meshes.values())[0].faces)
# ps.get_surface_mesh("Armadillo").add_scalar_quantity("my_scalar", 
        # vertex_values, defined_on='vertices', cmap='blues')
ps.get_surface_mesh("Armadillo").add_vector_quantity("my_vector", 
        face_vectors, defined_on='faces', color=(0.2, 0.5, 0.5))

# View the point cloud and mesh we just registered in the 3D UI
ps.show()