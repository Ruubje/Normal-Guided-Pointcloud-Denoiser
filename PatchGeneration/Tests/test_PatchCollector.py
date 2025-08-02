import pytest
from Modules.Mesh import *
from Modules.PatchCollector import *
from Modules.RotationMatrix import *
import numpy as np
import igl

@pytest.fixture
def myPatchCollector():
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
    return PatchCollector(Mesh(v, f))

def test_selectPaperPatch(myPatchCollector):
    # tba
    return
