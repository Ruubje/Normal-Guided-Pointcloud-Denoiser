import pytest
from Modules.Mesh import *
from Modules.RotationMatrix import *
import numpy as np
import igl

@pytest.fixture
def myMesh():
    v = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0]
    ], dtype=np.float32)
    f = np.array([
        [1, 0, 2],
        [1, 2, 3],
        [3, 2, 4],
        [3, 4, 5],
        [4, 0, 5],
        [0, 1, 5],
        [2, 0, 4],
        [1, 3, 5]
    ])
    return Mesh(v, f)

@pytest.fixture
def myPatch(myMesh):
    parent = myMesh
    PI = 0
    v = myMesh.v
    f = myMesh.f[[0, 1, 5, 6]]
    patch = Patch(parent, PI, v, f)
    return patch

def test_setup_example_mesh(myMesh):
    assert myMesh.v.shape == (6, 3)
    assert myMesh.f.shape == (8, 3)

def test_normals_example_mesh(myMesh):
    normals = myMesh.getFaceNormals()
    barycenters = igl.barycenter(myMesh.v, myMesh.f)
    dot_product = np.sum(normals * barycenters, axis=1)
    assert np.all(dot_product > 0)

def test_createPatch_remove_vertex_5(myMesh):
    OLD_FI = 6
    patch = myMesh.createPatch([0, 1, 2, 6], OLD_FI)
    assert patch.v.shape[0] == 5
    assert patch.pi == 3

def test_createPatch_remove_vertices_3_and_4(myMesh):
    OLD_FI = 5
    patch = myMesh.createPatch([0, 5], OLD_FI)
    assert patch.v.shape[0] == 4
    assert patch.pi == 1

def test_createPatch_remove_vertices_3_and_4_2(myMesh):
    OLD_FI = 4
    patch = myMesh.createPatch([0, 5], OLD_FI)
    assert patch.v.shape[0] == 4
    assert patch.pi == -1

def test_copy_all_values_are_different_objects_or_values(myMesh):
    # Setup attribute values'
    _ = myMesh.getFaceNormals()
    _ = myMesh.getNeighbourhood(0, 1)
    _ = myMesh.getVertexTriangleAdjacency()
    # Copy and assert
    newMesh = myMesh.copy()
    assert not (newMesh is myMesh)
    for attribute, value1 in vars(myMesh).items():
        value2 = vars(newMesh)[attribute]
        assert not (value1 is None) and not (value2 is None)
        assert not (value1 is value2) or (value1 == value2 and type(value1) == int)
    
def test_getPCCenter(myMesh):
    center = myMesh.getPCCenter()
    assert np.all(center == 0.0)
    
def test_getPCSize(myMesh):
    size = myMesh.getPCSize()
    assert size == 1.0

def test_getPCBoundingBox(myMesh):
    size = myMesh.getPCBoundingBox()
    assert np.all(size == np.array([2, 2, 2]))
    
def test_getAreas(myMesh):
    areas = myMesh.getAreas(np.arange(len(myMesh.f)))
    assert areas.shape[0] == myMesh.f.shape[0]
    assert np.all(areas == np.sqrt(3)/2)

def test_getFacesInRange(myMesh):
    center = np.array([2.0, 0.0, 0.0])
    range = 1.5
    faces_in_range = myMesh.getFacesInRange(center, range)
    assert np.all(faces_in_range == np.array([0, 1, 2, 6]))

def test_getNeighbourhood_wrong_ring(myMesh):
    face_index = 0
    ring = -10
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert neighbourhood.shape[0] == 0
    
def test_getNeighbourhood_1_ring(myMesh):
    face_index = 0
    ring = 1
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert np.all(neighbourhood == np.array([0, 1, 5, 6]))

def test_getNeighbourhood_2_ring(myMesh):
    face_index = 0
    ring = 2
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert np.all(neighbourhood == np.array([0, 1, 2, 4, 5, 6, 7]))

def test_getNeighbourhood_3_ring(myMesh):
    face_index = 0
    ring = 3
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert np.all(neighbourhood == np.arange(myMesh.f.shape[0]))

def test_getFaceNormals(myMesh):
    face_normals = myMesh.getFaceNormals()
    sqrt1div3 = np.sqrt(1/3)
    expected_result = np.array([
        [sqrt1div3, sqrt1div3, sqrt1div3],
        [sqrt1div3, sqrt1div3, -sqrt1div3],
        [sqrt1div3, -sqrt1div3, -sqrt1div3],
        [-sqrt1div3, -sqrt1div3, -sqrt1div3],
        [-sqrt1div3, -sqrt1div3, sqrt1div3],
        [-sqrt1div3, sqrt1div3, sqrt1div3],
        [sqrt1div3, -sqrt1div3, sqrt1div3],
        [-sqrt1div3, sqrt1div3, -sqrt1div3]
    ])
    assert np.all(np.square(face_normals - expected_result) < 0.001)

def test_getFaceNormals_Twice(myMesh):
    face_normals = myMesh.getFaceNormals()
    assert  np.allclose(face_normals, myMesh.getFaceNormals())

def test_getVertexTriangleAdjacency(myMesh):
    vta = myMesh.getVertexTriangleAdjacency()
    expected_result_0 = np.array([0, 4, 5, 6, 0, 1, 5, 7, 0, 1, 2, 6, 1, 2, 3, 7, 2, 3, 4, 6, 3, 4, 5, 7], dtype=np.int32)
    expected_result_1 = np.array([0, 4, 8, 12, 16, 20, 24], dtype=np.int32)
    assert np.all(vta[0] == expected_result_0)
    assert np.all(vta[1] == expected_result_1)

def test_getVertexTriangleAdjacency_call_twice(myMesh):
    vta = myMesh.getVertexTriangleAdjacency()
    assert vta is myMesh.getVertexTriangleAdjacency()

def test_getTrianglesOfVertex(myMesh):
    vi = 0
    triangles = myMesh.getTrianglesOfVertex(vi)
    expected_result = np.array([0, 4, 5, 6])
    assert np.all(triangles == expected_result)

def test_getTrianglesOfVertices(myMesh):
    vis = np.array([0, 1])
    triangles = myMesh.getTrianglesOfVertices(vis)
    expected_result = np.array([0, 1, 4, 5, 6, 7])
    assert np.all(triangles == expected_result)

def test_translate(myMesh):
    translation = np.array([0.5, 2.5, 1.3])
    old_v = np.copy(myMesh.getVertices())
    size = myMesh.getPCSize()

    myMesh.translate(translation)

    newVertices = myMesh.getVertices()
    oldVerticesTranslated = old_v + translation

    assert np.allclose(newVertices, oldVerticesTranslated)
    assert np.allclose(size, myMesh.getPCSize())
    # assert False

def test_resize(myMesh):
    new_size = 100
    expected_result = np.array([
        [0, 0, 100],
        [0, 100, 0],
        [100, 0, 0],
        [0, 0, -100],
        [0, -100, 0],
        [-100, 0, 0]
    ], dtype=np.float32)

    myMesh.resize(new_size)

    assert np.all(myMesh.v == expected_result)

def test_rotate(myMesh):
    rotation = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    expected_result = np.array([
        [0, -1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0]
    ], dtype=np.float32)

    myMesh.rotate(rotation)

    assert np.all(myMesh.v == expected_result)

def test_rotate_Normals(myMesh):
    rotation = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    _ = myMesh.getFaceNormals()
    sqrt1div3 = np.sqrt(1/3)
    expected_result = np.array([
        [sqrt1div3, -sqrt1div3, sqrt1div3],
        [sqrt1div3, sqrt1div3, sqrt1div3],
        [sqrt1div3, sqrt1div3, -sqrt1div3],
        [-sqrt1div3, sqrt1div3, -sqrt1div3],
        [-sqrt1div3, -sqrt1div3, -sqrt1div3],
        [-sqrt1div3, -sqrt1div3, sqrt1div3],
        [sqrt1div3, -sqrt1div3, -sqrt1div3],
        [-sqrt1div3, sqrt1div3, sqrt1div3]
    ], dtype=np.float32)

    myMesh.rotate(rotation)

    assert np.allclose(myMesh.getFaceNormals(), expected_result)

def test_myPatch_all_vertices_have_a_face(myPatch):
    assert np.all(np.isin(np.arange(len(myPatch.v)), myPatch.f))

def test_myPatch_patch_is_from_mesh(myPatch, myMesh):
    assert myPatch.parent is myMesh

def test_myPatch_patch_is_1_ring_and_pi_is_center(myPatch):
    assert np.all(myPatch.getNeighbourhood(myPatch.pi, 1) == np.arange(len(myPatch.f)))

def test_myPatch_patch_copy_duplicates_all_attributes(myPatch):
    # Setup attribute values
    _ = myPatch.getFaceNormals()
    _ = myPatch.getNeighbourhood(myPatch.pi, 1)
    _ = myPatch.getVertexTriangleAdjacency()
    myPatch.lastRotationApplied = np.identity(3)
    # Copy and assert
    newPatch = myPatch.copy()
    assert not (myPatch is newPatch)
    for attribute, value1 in vars(myPatch).items():
        value2 = vars(newPatch)[attribute]
        assert not (value1 is None) and not (value2 is None)
        assert not (value1 is value2) or type(value1) == int or type(value1) == Mesh

def test_alignPatch_patch_is_centered(myPatch):
    myPatch.alignPatch()
    ORIGIN = np.array([0, 0, 0])
    assert np.allclose(myPatch.getPCCenter(), ORIGIN)
    
def test_alignPatch_patch_is_unit_size(myPatch):
    myPatch.alignPatch()
    assert np.allclose(myPatch.getPCSize(), 1.0)

def test_alignPatch_twice_has_same_center(myPatch):
    myPatch.alignPatch()
    testPatch = myPatch.copy()
    testPatch.alignPatch()

    center1 = myPatch.getPCCenter()
    center2 = testPatch.getPCCenter()

    assert np.allclose(center1, center2)

def test_alignPatch_twice_has_same_size(myPatch):
    myPatch.alignPatch()
    testPatch = myPatch.copy()
    testPatch.alignPatch()

    size1 = myPatch.getPCSize()
    size2 = testPatch.getPCSize()

    assert np.allclose(size1, size2)
    
def test_alignPatch_twice_has_same_rotation(myPatch):
    myPatch.alignPatch()
    testPatch = myPatch.copy()
    testPatch.alignPatch()

    position1 = myPatch.getPCCenter()
    position2 = testPatch.getPCCenter()

    size1 = myPatch.getPCSize()
    size2 = testPatch.getPCSize()

    rotation1 = myPatch.getPaperRotationMatrix().matrix
    rotation2 = testPatch.getPaperRotationMatrix().matrix

    print(f"Position 1:\n{position1}\nPosition 2:\n{position2}\nSize 1:\n{size1}\nSize 2:\n{size2}\nRotation matrix 1:\n{rotation1}\nRotation matrix 2:\n{rotation2}")
    print(f"Angle1 can be {np.arccos(rotation1[1][1])}, {np.arcsin(rotation1[1][2])}, {-np.arcsin(rotation1[2][1])} or {np.arccos(rotation1[2][2])}")
    print(f"Angle2 can be {np.arccos(rotation2[1][1])}, {np.arcsin(rotation2[1][2])}, {-np.arcsin(rotation2[2][1])} or {np.arccos(rotation2[2][2])}")
    # print(f"{rotation1.dtype}")

    assert np.allclose(rotation1, rotation2)

def test_alignPatch_rotationmatrix_after_alignment_should_be_identity_or_180_degrees_flipped(myPatch):
    myPatch.alignPatch()
    identity = np.identity(3)
    flipped = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_matrix = myPatch.getPaperRotationMatrix()
    assert np.allclose(rotation_matrix.matrix, identity) or np.allclose(rotation_matrix.matrix, flipped)

def test_getPaperRotationMatrix_returns_a_RotationMatrix_object(myPatch):
    rotation_matrix = myPatch.getPaperRotationMatrix()
    assert isinstance(rotation_matrix, RotationMatrix)