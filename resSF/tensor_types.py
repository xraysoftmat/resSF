"""
Types to support the scattering tensor calculations for materials.
"""

# StdLib
from enum import StrEnum

# Third-party
import kkcalc2 as kk
import numpy as np


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
"""A single value for the scattering tensor, representing isotropic scattering in any direction."""
np_inplane_isotropic = np.ndarray[tuple[2], np.dtype[np.floating | np.complexfloating]]
tensor_inplane_isotropic = (
    tuple[tensor_isotropic, tensor_isotropic] | np_inplane_isotropic
)
"""A tuple of two values for the scattering tensor, representing isotropic scattering in the plane
and a different value for the out-of-plane direction."""
np_xyz = np.ndarray[tuple[3], np.dtype[np.floating | np.complexfloating]]
tensor_xyz = tuple[tensor_isotropic, tensor_isotropic, tensor_isotropic] | np_xyz
"""A tuple of three values for the scattering tensor, representing scattering in the x, y, and z directions."""
np_full = np.ndarray[tuple[3, 3], np.dtype[np.floating | np.complexfloating]]
tensor_full = tuple[tensor_xyz, tensor_xyz, tensor_xyz] | np_full
"""A full 3x3 matrix for the scattering tensor, representing anisotropic scattering in all directions."""
np_single = tensor_isotropic | np_inplane_isotropic | np_xyz | np_full
"""A possible numpy array for a single energy value."""
tensor_type = tensor_isotropic | tensor_inplane_isotropic | tensor_xyz | tensor_full
"""The scattering tensor type for a single energy value.
Can be isotropic, in-plane isotropic, xyz, or full tensor."""


"""Any available tensor for a single energy value."""
np_array_isotropic = np.ndarray[tuple[int], np.dtype[np.floating | np.complexfloating]]
array_isotropic = (
    list[tensor_isotropic]
    | np_array_isotropic
    | kk.models.asp_abstract
    | kk.models.asf_abstract
)
"""An array of isotropic scattering tensors, representing isotropic scattering in any direction for
multiple energy values."""
np_array_inplane_isotropic = np.ndarray[
    tuple[int, 2], np.dtype[np.floating | np.complexfloating]
]
array_inplane_isotropic = (
    list[tensor_inplane_isotropic]
    | np_array_inplane_isotropic
    | tuple[array_isotropic, array_isotropic]
)
"""An array of in-plane isotropic scattering tensors, representing isotropic scattering in the plane"""
np_array_xyz = np.ndarray[tuple[int, 3], np.dtype[np.floating | np.complexfloating]]
array_xyz = (
    list[tensor_xyz]
    | np_array_xyz
    | tuple[array_isotropic, array_isotropic, array_isotropic]
)
"""An array of scattering tensors in the x, y, and z directions for multiple energy values."""
np_array_full = np.ndarray[tuple[int, 3, 3], np.dtype[np.floating | np.complexfloating]]
array_full = (
    list[tensor_full]
    | np_array_full
    | tuple[
        array_xyz,
        array_xyz,
        array_xyz,
    ]
)
"""An array of full 3x3 scattering tensors for multiple energy values."""

np_array_type = (
    np_array_isotropic | np_array_inplane_isotropic | np_array_xyz | np_array_full
)
array_type = (
    tensor_isotropic
    | tensor_inplane_isotropic
    | tensor_xyz
    | tensor_full
    | array_isotropic
    | array_inplane_isotropic
    | array_xyz
    | array_full
)
"""Any available tensor array type for multiple energy values."""

asp_iso = kk.models.asp_abstract
"""A single kkcalc2 asp model for isotropic scattering."""
asp_inplane_iso = tuple[kk.models.asp_abstract, kk.models.asp_abstract]
"""A tuple of two kkcalc2 asp models for in-plane isotropic scattering."""
asp_xyz = tuple[kk.models.asp_abstract, kk.models.asp_abstract, kk.models.asp_abstract]
"""A tuple of three kkcalc2 asp models for scattering in the x, y, and z directions."""
asp_full = tuple[asp_xyz, asp_xyz, asp_xyz]
"""A full 3x3 matrix of kkcalc2 asp models for anisotropic scattering in all directions."""
asp_array_type = asp_iso | asp_inplane_iso | asp_xyz | asp_full
"""Any possible format of scattering tensor using kkcalc2 asp models, excluding singleton types."""
