import pytest
from Modules.Mesh import *
from Modules.RotationMatrix import *
from Tests.test_Mesh import myMesh, myPatch
import numpy as np

@pytest.fixture
def RotationMatrix(myPatch):
    return myPatch.getPaperRotationMatrix()

def test_RotationMatrix_bcs_are_barycenters(myPatch, RotationMatrix):
    barycenters = np.sum(myPatch.v[myPatch.f], axis=1) / 3
    assert np.allclose(RotationMatrix.bcs, barycenters)

def test_RotationMatrix_ci_check_shape(RotationMatrix):
    TARGET_SHAPE = (3,)
    assert RotationMatrix.ci.shape == TARGET_SHAPE

def test_RotationMatrix_ci_is_center_barycenter(RotationMatrix):
    TARGET_BARYCENTER = np.array([1, 1, 1]) / 3.
    assert np.allclose(RotationMatrix.ci, TARGET_BARYCENTER)

def test_RotationMatrix_cj_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3)
    assert RotationMatrix.cj.shape == TARGET_SHAPE

def test_RotationMatrix_dcs_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3)
    assert RotationMatrix.dcs.shape == TARGET_SHAPE

def test_RotationMatrix_nj_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3)
    assert RotationMatrix.nj.shape == TARGET_SHAPE

def test_RotationMatrix_nj_are_normal_vectors(RotationMatrix):
    assert np.allclose(np.linalg.norm(RotationMatrix.nj, axis=1), 1)

def test_RotationMatrix_raw_wj_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3)
    assert RotationMatrix.raw_wj.shape == TARGET_SHAPE

def test_RotationMatrix_raw_wj_perpendicular_to_dcs(RotationMatrix):
    assert np.allclose(np.sum(RotationMatrix.raw_wj * RotationMatrix.dcs, axis=0), 0)

def test_RotationMatrix_wj_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3)
    assert RotationMatrix.wj.shape == TARGET_SHAPE

def test_RotationMatrix_wj_are_normal_vectors(RotationMatrix):
    assert np.allclose(np.linalg.norm(RotationMatrix.wj, axis=1), 1)

def test_RotationMatrix_njprime_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3)
    assert RotationMatrix.njprime.shape == TARGET_SHAPE

def test_RotationMatrix_njprime_are_normal_vectors(RotationMatrix):
    assert np.allclose(np.linalg.norm(RotationMatrix.njprime, axis=1), 1)

def test_RotationMatrix_njprime_plus_nj_are_parallel_to_wj(RotationMatrix):
    wj = RotationMatrix.wj
    njs = (RotationMatrix.nj + RotationMatrix.njprime)
    assert np.allclose(np.sum(np.multiply(njs, wj), axis=1), np.linalg.norm(wj, axis=1) * np.linalg.norm(njs, axis=1))
    
def test_RotationMatrix_angle_between_njs_is_half_the_angle_between_nj_and_wj(RotationMatrix):
    wj = RotationMatrix.wj
    nj = RotationMatrix.nj
    njprime = RotationMatrix.njprime
    assert np.allclose(np.arccos(np.sum(np.multiply(nj, njprime), axis=1)), np.arccos(np.sum(np.multiply(wj, njprime), axis=1)) * 2)
    
def test_RotationMatrix_areas_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1,)
    assert RotationMatrix.areas.shape == TARGET_SHAPE

def test_RotationMatrix_areas_all_the_same(RotationMatrix):
    EXPECTED_AREA = np.sqrt(3) / 2
    assert np.allclose(RotationMatrix.areas, EXPECTED_AREA)

def test_RotationMatrix_muj_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1,)
    assert RotationMatrix.muj.shape == TARGET_SHAPE

def test_RotationMatrix_outer_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3, 3)
    assert RotationMatrix.outer.shape == TARGET_SHAPE
    
def test_RotationMatrix_outer_product_characteristic(RotationMatrix):
    # Test (nj' ⊗ nj') * nj = nj' * <nj', nj>,
    # where <,> is the inner product and ⊗ is the outer product
    left_side = np.dot(RotationMatrix.outer, RotationMatrix.nj)
    right_side = np.dot(RotationMatrix.njprime, RotationMatrix.nj) * RotationMatrix.njprime
    assert np.allclose(left_side, right_side)

def test_RotationMatrix_outer_product_eigenvalue_characteristic(RotationMatrix):
    # Test if the eigenvalues of the outer products equal the lengths squared of all njprimes
    # Side note: the outer products have more than 1 eigenvalue, but the others equal 0 with eigenvectors in the orthogonal subspace.
    #   We are not interested in those vectors and therefore we find the max eigenvalue.
    eigenvalues = np.max(np.linalg.eigvals(RotationMatrix.outer), axis=1)
    assert np.allclose(eigenvalues, np.square(np.linalg.norm(RotationMatrix.njprime, axis=1)))

def test_RotationMatrix_outer_product_check_eigenvector_is_njprime(RotationMatrix):
    # Test if the eigenvector with the largest eigenvalue of the outer products equal nj'
    for i in range(RotationMatrix.outer.shape[0]):
        eigh = np.linalg.eigh(RotationMatrix.outer[i])
        index_of_largest_eigenvalue = np.argmax(eigh[0])
        assert np.allclose(eigh[1][:, index_of_largest_eigenvalue], RotationMatrix.njprime[i])

def test_RotationMatrix_Tj_check_shape(myPatch, RotationMatrix):
    TARGET_SHAPE = (len(myPatch.f) - 1, 3, 3)
    assert RotationMatrix.Tj.shape == TARGET_SHAPE

def test_RotationMatrix_Tj_check_eigenvector_is_njprime_scaled_with_muj(RotationMatrix):
    for i in range(RotationMatrix.Tj.shape[0]):
        eigh = np.linalg.eigh(RotationMatrix.Tj[i])
        index_of_largest_eigenvalue = np.argmax(eigh[0])
        eigenvector = eigh[1][:, index_of_largest_eigenvalue]
        njprime = RotationMatrix.njprime[i]
        assert np.allclose(eigenvector, njprime) or np.allclose(-eigenvector, njprime)
        assert np.allclose(eigh[0][index_of_largest_eigenvalue], RotationMatrix.muj[i])

def test_Einsum_Notation_Works_As_Intended(N=10):
    A = np.random.rand(N, 3)
    B = np.random.rand(N, 3, 3)

    R = np.zeros((N, 3))
    for i in range(N):
        R[i, :] = B[i, :, :] @ A[i, :]
        
    einR = np.einsum("ij,ikj->ik", A, B)

    assert np.allclose(einR, R)

def test_Noise_Generation_Unbiased(N=100000, threshold=0.8):
    random_sample = np.random.normal(size=(N, 3))
    random_direction = random_sample / np.linalg.norm(random_sample, axis=1)[:, None]

    axes = np.identity(3)
    c = (1./3.)**(1./2.)
    corners = np.array([[c, c, c], [-c, c, c], [-c, -c, c]])

    normal_axes = np.sum(np.einsum("ni,ij->nj", random_direction, axes)>threshold)/N
    diagonal_axes = np.sum(np.einsum("ni,ij->nj", random_direction, corners)>threshold)/N
    assert (normal_axes - diagonal_axes) ** 2 < 0.1
