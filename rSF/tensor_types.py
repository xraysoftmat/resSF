"""
Types to support the scattering tensor calculations for materials.
"""

from enum import StrEnum
import numpy as np
import kkcalc2 as kk

# Enumerates


class Alignment(StrEnum):
    """
    Enum for alignment input types.
    """

    ISOTROPIC = "isotropic"
    """A single value for the scattering tensor, representing isotropic scattering in any direction."""
    INPLANE_ISOTROPIC = "inplane_isotropic"
    """A tuple of two values for the scattering tensor, representing isotropic scattering in the plane
    and a different value for the out-of-plane direction."""
    XYZ = "xyz"
    """A tuple of three values for the scattering tensor, representing scattering in the x, y, and z directions."""
    FULL = "full"
    """A full 3x3 matrix for the scattering tensor, representing anisotropic scattering in all directions."""


# Types
tensor_isotropic = float | complex
tensor_inplane_isotropic = tuple[tensor_isotropic, tensor_isotropic]
tensor_xyz = tuple[tensor_isotropic, tensor_isotropic, tensor_isotropic]
tensor_full = tuple[tensor_xyz, tensor_xyz, tensor_xyz]

array_tensor_isotropic = (
    list[tensor_isotropic]
    | np.ndarray[tuple[int], np.dtype[np.floating]]
    | kk.models.asp
    | kk.models.asf
)
array_tensor_inplane_isotropic = (
    list[tensor_inplane_isotropic]
    | np.ndarray[tuple[int, 2], np.dtype[np.floating]]
    | kk.models.asp
    | kk.models.asf
)
array_tensor_xyz = (
    list[tensor_xyz]
    | np.ndarray[tuple[int, 3], np.dtype[np.float64]]
    | kk.models.asp
    | kk.models.asf
)
array_tensor_full = (
    list[tensor_full]
    | np.ndarray[tuple[int, 3, 3], np.dtype[np.float64]]
    | kk.models.asp
    | kk.models.asf
)

tensor_type = (
    tensor_isotropic
    | tensor_inplane_isotropic
    | tensor_xyz
    | tensor_full
    | array_tensor_isotropic
    | array_tensor_inplane_isotropic
    | array_tensor_xyz
    | array_tensor_full
)
