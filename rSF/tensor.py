"""
Class to store spectroscopic atomic scattering tensor information for a given atom in a material.

Atomic presence in the material could have alignment or be isotropic.

We define the substrate frame as follows:
- x is the in-plane direction along the substrate, perpendicular to the beam direction
- y is the in-plane direction along the substrate, parallel to the beam direction
- z is the out-of-plane direction, perpendicular to the substrate plane (i.e. the surface normal)
"""

# Stdlib
from typing import Sequence

# Third-party
import numpy as np

# Local
from rSF.tensor_types import (
    Alignment,
    array_tensor_full,
    array_tensor_inplane_isotropic,
    array_tensor_isotropic,
    array_tensor_xyz,
    tensor_full,
    tensor_inplane_isotropic,
    tensor_isotropic,
    tensor_xyz,
)


class AtomicScatteringTensor:
    """
    Class for atomic scattering tensor calculations.

    Parameters
    ----------
    atom : str
        The type of atom (e.g., "Fe", "O", etc.).
    energies : np.ndarray[tuple[int], np.dtype[np.floating]] | Sequence[int | float] | float | int
        The energies at which the scattering tensor is defined.
    tensor : tensor_isotropic | tensor_inplane_isotropic | tensor_xyz | tensor_full |
             array_tensor_isotropic | array_tensor_inplane_isotropic | array_tensor_xyz | array_tensor_full
        The scattering tensor values, which can be isotropic, in-plane isotropic, xyz, or full tensor.
    alignment : Alignment, optional
        The alignment of the tensor, default is Alignment.ISOTROPIC.

    Examples
    --------
    An example of creating an `Alignment.INPLANE-ISOTROPIC` AtomicScatteringTensor for an atom "S" (sulfur)
    at a set of energies:
    >>> energies = np.linspace(2450, 2550, 100) # Define energies from 2450 eV to 2550 eV
    >>> tensor_values_ip = np.random.rand(100) # In-plane random tensor values for demonstration
    >>> tensor_values_oop = np.random.rand(100) # Out-of-plane
    >>> tensor_values = np.column_stack((tensor_values_ip, tensor_values_oop)) # Combine into a 2D array
    >>> tensor_S = AtomicScatteringTensor(
        atom="S",
        energies=energies,
        tensor=tensor_values,
        alignment=Alignment.INPLANE_ISOTROPIC
    )
    """

    def __init__(
        self,
        atom: str,
        energies: np.ndarray[tuple[int], np.dtype[np.floating]]
        | Sequence[float]
        | float,
        tensor: tensor_isotropic
        | tensor_inplane_isotropic
        | tensor_xyz
        | tensor_full
        | array_tensor_isotropic
        | array_tensor_inplane_isotropic
        | array_tensor_xyz
        | array_tensor_full,
        alignment: Alignment | None = None,
    ):
        # Add the atom type
        self.atom_type = atom
        energies = np.array(energies, dtype=np.float64)
        # Remove single-dimensional entries from the shape of an array.
        if energies.ndim == 1 and energies.shape[0] == 1:
            energies = energies[0]
            assert isinstance(energies, (float, int)), (
                "Energies must be a float or int for singular energy."
            )

        # Add the energies to the instance
        self.energies: np.ndarray[tuple[int], np.dtype[np.floating]] | float | int = (
            energies
        )
        if not isinstance(tensor, (float, int)):
            tensor = np.array(tensor, dtype=np.float64)
        self._tensor = tensor

        # Singular energy values
        if self.L == 1:
            if isinstance(tensor, (float, int)):
                self.alignment = Alignment.ISOTROPIC
            elif tensor.ndim == 2 and tensor.shape[0] == 3 and tensor.shape[1] == 3:
                self.alignment = Alignment.FULL
            elif isinstance(self._tensor, (tuple, list)) and len(self._tensor) == 2:
                self.alignment = Alignment.INPLANE_ISOTROPIC
            elif isinstance(self._tensor, (tuple, list)) and len(self._tensor) == 3:
                self.alignment = Alignment.XYZ
            else:
                raise ValueError("Invalid tensor values provided for singular energy.")
        # Arrays
        elif isinstance(energies, np.ndarray):
            if not isinstance(self._tensor, (list, np.ndarray)):
                raise ValueError(
                    "Scattering tensors must be a list or numpy array when energies is an array."
                )
            shape = np.shape(self._tensor)
            assert energies.shape[0] == shape[0], (
                "Energies and values must have the same length."
            )
            if len(shape) == 1:
                self.alignment = Alignment.ISOTROPIC
            elif len(shape) == 2 and shape[1] == 2:
                self.alignment = Alignment.INPLANE_ISOTROPIC
            elif len(shape) == 2 and shape[1] == 3:
                self.alignment = Alignment.XYZ
            elif len(shape) == 3 and shape[1:] == (3, 3):
                self.alignment = Alignment.FULL
            else:
                raise ValueError(f"Invalid tensor values provided with shape {shape}.")
        else:
            raise ValueError("Invalid tensor values provided.")

        if alignment is not None and alignment != self.alignment:
            raise ValueError(
                f"Provided alignment `{alignment}` does not match inferred alignment `{self.alignment}`."
            )

    @property
    def L(self) -> int:
        """
        Get the number of energies for which the scattering tensor is defined.

        Returns
        -------
        int
            The number of energies.
        """
        if isinstance(self.energies, np.ndarray):
            return self.energies.shape[0]
        elif isinstance(self.energies, (list, tuple)):
            return len(self.energies)
        else:
            return 1

    @property
    def tensor(self):
        """
        Get the scattering tensor values.

        Returns
        -------
        tensor_isotropic | tensor_inplane_isotropic | tensor_xyz | tensor_full | array_tensor_isotropic | array_tensor_inplane_isotropic | array_tensor_xyz | array_tensor_full
            The scattering tensor values.
        """
        return self._tensor


class CrystalScatteringTensor:
    """
    A class to store and manage scattering tensors for different atomic types in a crystal structure.

    Requires the same energy values for all atom types.

    Parameters
    ----------
    scattering_factors : dict[str, AtomicScatteringTensor] | list[AtomicScatteringTensor] | None
        A dictionary mapping atomic types to their corresponding scattering tensors, or a list of scattering tensors.
        By default, this is None, indicating that no scattering tensors are defined.

    Examples
    --------
    An example of creating an isotropic CrystalScatteringTensor with two atomic types, "S" and "O":
    >>> tensor1 = AtomicScatteringTensor(atom="S", energies=[...], tensor=[...])
    >>> tensor2 = AtomicScatteringTensor(atom="O", energies=[...], tensor=[...])
    >>> crystal_tensor = CrystalScatteringTensor(scattering_factors={"S": tensor1, "O": tensor2})

    or

    >>> crystal_tensor = CrystalScatteringTensor(scattering_factors=[tensor1, tensor2])

    To access or modify the scattering tensor for a specific atomic type:
    >>> s_tensor = crystal_tensor["S"]  # Get the scattering tensor for sulfur
    >>> crystal_tensor["O"] = new_tensor  # Set a new scattering tensor for oxygen
    """

    def __init__(
        self,
        scattering_factors: dict[str, AtomicScatteringTensor]
        | list[AtomicScatteringTensor]
        | None = None,
    ):
        self.energies: (
            np.ndarray[tuple[int], np.dtype[np.floating]] | float | int | None
        ) = None
        self.scattering_factors: dict[str, AtomicScatteringTensor] = {}
        if scattering_factors is not None:
            # Collect energies from the first tensor to ensure consistency
            tensor = (
                next(iter(scattering_factors.values()))  # dict
                if isinstance(scattering_factors, dict)
                else scattering_factors[0]  # list
            )
            self.energies = energies = tensor.energies

            if isinstance(scattering_factors, dict):
                for atom, tensor in (scattering_factors or {}).items():
                    # Check if the energies match
                    if isinstance(energies, np.ndarray):
                        if not np.all(tensor.energies == energies):
                            raise ValueError(
                                "All scattering tensors must have the same energies. "
                                f"Atom '{atom}' has different energies."
                            )
                    else:
                        if energies != tensor.energies:
                            raise ValueError(
                                "All scattering tensors must have the same energies. "
                                f"Atom '{atom}' has different energies ({tensor.energies} vs {energies})."
                            )
                    # Add the item.
                    if atom in self.scattering_factors:
                        raise ValueError(
                            f"Duplicate scattering tensor for atom type '{atom}' found."
                        )
                    self.__setitem__(atom, tensor)

            elif isinstance(scattering_factors, list):
                for tensor in scattering_factors:
                    # Check if the energies match
                    if isinstance(energies, np.ndarray):
                        if not np.all(tensor.energies == energies):
                            raise ValueError(
                                "All scattering tensors must have the same energies. "
                                f"Atom '{tensor.atom_type}' has different energies."
                            )
                    else:
                        if energies != tensor.energies:
                            raise ValueError(
                                "All scattering tensors must have the same energies. "
                                f"Atom '{tensor.atom_type}' has different energies ({tensor.energies} vs {energies})."
                            )
                    # Add the item
                    if tensor.atom_type in self.scattering_factors:
                        raise ValueError(
                            f"Duplicate scattering tensor for atom type '{tensor.atom_type}' found."
                        )
                    self.__setitem__(tensor.atom_type, tensor)

    def __setitem__(self, atom: str, tensor: AtomicScatteringTensor):
        """
        Set the scattering tensor for a specific atomic type.

        Parameters
        ----------
        atom : str
            The atomic type for which to set the scattering tensor.
        tensor : AtomicScatteringTensor
            The scattering tensor to be set for the specified atomic type.
        """
        self.scattering_factors[atom] = tensor

    def __getitem__(self, atom: str) -> AtomicScatteringTensor:
        """
        Get the scattering tensor for a specific atomic type.

        Parameters
        ----------
        atom : str
            The atomic type for which to retrieve the scattering tensor.

        Returns
        -------
        AtomicScatteringTensor
            The scattering tensor for the specified atomic type.
        """
        return self.scattering_factors[atom]

    def add_atom(self, tensor: AtomicScatteringTensor):
        """
        Add a scattering tensor for a specific atomic type.

        Parameters
        ----------
        tensor : AtomicScatteringTensor
            The scattering tensor to be added.
        """
        self.__setitem__(tensor.atom_type, tensor)

    def get_atom(self, atom: str) -> AtomicScatteringTensor:
        """
        Retrieve the scattering tensor for a specific atomic type.

        Parameters
        ----------
        atom : str
            The atomic type for which to retrieve the scattering tensor.

        Returns
        -------
        AtomicScatteringTensor
            The scattering tensor for the specified atomic type.
        """
        return self.__getitem__(atom)
