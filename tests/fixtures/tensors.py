"""
PyTest fixtures for the testing structures.
"""

import pytest
import numpy as np
import kkcalc2 as kk


@pytest.fixture
def tensor_isotropic():
    return 1.0


@pytest.fixture
def np_tensor_isotropic():
    return np.float64(1.0)


@pytest.fixture
def tensor_inplane_isotropic():
    return (1.0, 2.0)


@pytest.fixture
def np_tensor_inplane_isotropic():
    return np.array([1.0, 2.0])


@pytest.fixture
def tensor_xyz():
    return (1.0, 2.0, 3.0)


@pytest.fixture
def np_tensor_xyz():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def tensor_full():
    return ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0))


@pytest.fixture
def np_tensor_full():
    return np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


@pytest.fixture
def array_energies():
    return np.array([2450, 2475, 2500, 2550])


@pytest.fixture
def array_tensor_isotropic():
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def array_tensor_inplane_isotropic():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


@pytest.fixture
def array_tensor_xyz():
    return np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )


@pytest.fixture
def array_tensor_full():
    return np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
            [[7.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 9.0]],
            [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
        ]
    )


@pytest.fixture
def kk_asp_isotropic():
    return kk.models.asp_db_complex("S")


@pytest.fixture
def kk_asp_inplane_isotropic(kk_asp_isotropic):
    return (kk_asp_isotropic, kk.models.asp_db_complex("2S"))


@pytest.fixture
def kk_asp_xyz(kk_asp_isotropic):
    return (
        kk_asp_isotropic,
        kk.models.asp_db_complex("2S"),
        kk.models.asp_db_complex("3S"),
    )


@pytest.fixture
def kk_asp_full(kk_asp_xyz):
    return (
        kk_asp_xyz,
        kk_asp_xyz[1:] + kk_asp_xyz[:1],
        kk_asp_xyz[::-1],
    )
