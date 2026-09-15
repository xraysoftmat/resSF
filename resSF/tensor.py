"""
Class to store spectroscopic atomic scattering tensor information for a given atom in a material.

Atomic presence in the material could have alignment or be isotropic.

We define the substrate frame as follows:
- x is the in-plane direction along the substrate, perpendicular to the beam direction
- y is the in-plane direction along the substrate, parallel to the beam direction
- z is the out-of-plane direction, perpendicular to the substrate plane (i.e. the surface normal)
"""

# Stdlib
from collections.abc import Sequence
from typing import Literal, Self
from enum import Enum

# Third-party
import kkcalc2 as kk
import numpy as np

# Local
from resSF.tensor_types import (
    Alignment,
    array_type,
    asp_array_type,
    np_array_type,
    np_single,
    tensor_type,
)


class Complexity(Enum):
    """
    An enumeration to represent the complexity of the scattering tensor.

    Complex values are automatically inferred.
    """

    REAL = "real"
    """The tensor is real-valued."""
    IMAG = "imag"
    """The tensor is imaginary-valued."""


class AtomicScatteringTensor:
    """
    Class for atomic scattering tensor calculations.

    Stores representations as a group of kkcalc2 piecewise polynomials,
    which can be used to calculate the scattering tensor at any energy within the defined energy range.

    Parameters
    ----------
    atom : str
        The type of atom (e.g., "Fe", "O", etc.).
    energies : np.ndarray[tuple[int], np.dtype[np.floating]] | Sequence[int | float] | float | int
        The energies at which the scattering tensor is defined.
    tensor : tensor_type | array_type
        The scattering tensor values, which can be isotropic, in-plane isotropic, xyz, or full tensor.
    alignment : Alignment, optional
        The alignment of the tensor, default is Alignment.ISOTROPIC.
    complexity : Literal["real" | "imag"] | Complexity, optional
        The complexity of the tensor, default is None. Required if the tensor is a numpy array and not complex.

    Notes
    -----
    This class is based on kkcalc2 piecewise polynomials, which offer a convenience way to handle scattering tensors factors.

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

    To access the scattering tensor at a specific energy value, you can call the `tensor` property with the desired energy:
    >>> energy_value = 2500  # eV
    >>> scattering_tensor_at_energy = tensor_S(energy_value)
    """

    def __init__(
        self,
        atom: str | None,
        energies: np.ndarray[tuple[int], np.dtype[np.floating]]
        | Sequence[float]
        | float
        | None = None,
        tensor: tensor_type | array_type | None = None,
        alignment: Alignment | None = None,
        complexity: Literal["real", "imag"] | Complexity | None = None,
    ):
        # First check if the tensors are kkcalc2 objects, which already have energies and values defined
        self.energy_domain: tuple[float, float] = (0, np.inf)
        """The valid energy range for the scattering tensor, defined as a tuple (min_energy, max_energy)."""
        self._tensor: asp_array_type | np_single
        """The scattering tensor values, which can be isotropic, in-plane isotropic, xyz, or full_tensor,
        represented as a kkcalc2 object or a tuple of kkcalc2 objects depending on the alignment."""
        if complexity in Complexity:
            complexity = Complexity(complexity)
        elif complexity is None:
            pass
        else:
            raise ValueError(
                f"Invalid complexity value provided; {complexity}. Must be 'real' or 'imag'."
            )

        if isinstance(tensor, (kk.models.asp_abstract, kk.models.asf_abstract)):
            # A single kkcalc2 object
            self._tensor = (
                tensor
                if isinstance(tensor, kk.models.asp_abstract)
                else tensor.to_ASP()
            )
            print("HERE")
            self.energy_domain = (tensor.energies[0], tensor.energies[-1])
            stoich = tensor.stoichiometry
            if stoich is None:
                if atom is None:
                    raise ValueError(
                        "The provided kkcalc2 object does not have a defined stoichiometry. "
                        "Please provide an atomic type for the AtomicScatteringTensor."
                    )
                self.atom_type = atom
            else:
                composition = stoich.composition
                if len(composition) > 1:
                    raise ValueError(
                        "The provided kkcalc2 object contains multiple atomic types. "
                        "Please provide a single atomic type for the AtomicScatteringTensor."
                    )
                else:
                    # Convert int to string label for the atom type
                    atom_num = composition[0][0]
                    if isinstance(atom_num, int):
                        self.atom_type = kk.stoichiometry._atomic_number_to_element(
                            atom_num
                        )
                    else:
                        self.atom_type = atom_num
            self._alignment = Alignment.ISOTROPIC

        elif isinstance(tensor, tuple) and isinstance(
            tensor[0], (kk.models.asp_abstract, kk.models.asf)
        ):
            # A tuple of kkcalc2 objects
            non_full_tensors: list[kk.models.asp_abstract] = []
            energy_window = None
            stoich = tensor[0].stoichiometry
            if stoich is None:
                if atom is None:
                    raise ValueError(
                        "The provided kkcalc2 object does not have a defined stoichiometry. "
                        "Please provide an atomic type for the AtomicScatteringTensor."
                    )
                self.atom_type = atom
            else:
                composition = stoich.composition
                if len(composition) > 1:
                    raise ValueError(
                        "The provided kkcalc2 object contains multiple atomic types. "
                        "Please provide a single atomic type for the AtomicScatteringTensor."
                    )
                else:
                    # Convert int to string label for the atom type
                    atom_num = composition[0][0]
                    if isinstance(atom_num, int):
                        self.atom_type = kk.stoichiometry._atomic_number_to_element(
                            atom_num
                        )
                    else:
                        self.atom_type = atom_num

            for i, t in enumerate(tensor):
                assert isinstance(t, (kk.models.asp_abstract, kk.models.asf)), (
                    "All tensors in the tuple must be kkcalc2 objects (asp or asf)."
                )
                if isinstance(t, kk.models.asf):
                    t_asp = t.to_ASP()
                    non_full_tensors.append(t_asp)
                elif isinstance(t, kk.models.asp_abstract):
                    non_full_tensors.append(t)
                else:
                    raise TypeError(
                        "All tensors in the tuple must be kkcalc2 objects (asp or asf)."
                    )
                if energy_window is None:
                    energy_window = (t.energies[0], t.energies[-1])
                else:
                    energy_window = (
                        max(energy_window[0], t.energies[0]),
                        min(energy_window[1], t.energies[-1]),
                    )
                stoich = tensor[i].stoichiometry
                if stoich is None:
                    if atom is None:
                        raise ValueError(
                            f"The provided kkcalc2 object at index {i} does not have a defined stoichiometry. "
                            "Please provide an atomic type for the AtomicScatteringTensor."
                        )
                    atom_type = atom
                else:
                    composition = stoich.composition
                    if len(composition) > 1:
                        raise ValueError(
                            f"The provided kkcalc2 object at index {i} contains multiple atomic types. "
                            "Please provide a single atomic type for the AtomicScatteringTensor."
                        )
                    else:
                        # Convert int to string label for the atom type
                        atom_num = composition[0][0]
                        if isinstance(atom_num, int):
                            atom_type = kk.stoichiometry._atomic_number_to_element(
                                atom_num
                            )
                        else:
                            atom_type = atom_num
                if self.atom_type != atom_type:
                    raise ValueError(
                        f"The kkcalc2 object at index {i} corresponds to a different atomic type ({atom_type}) than the first one ({self.atom_type})."
                    )

            tensors = tuple(non_full_tensors)
            if energy_window is not None:
                self.energy_domain = energy_window
            if len(tensors) == 3:
                self._alignment = Alignment.XYZ
                self._tensor = tensors
            elif len(tensors) == 2:
                self._alignment = Alignment.INPLANE_ISOTROPIC
                self._tensor = tensors
            else:
                raise ValueError(
                    "The tuple of kkcalc2 objects must have either 2 (in-plane isotropic) or 3 (xyz) elements."
                )
        elif (
            isinstance(tensor, tuple)
            and isinstance(tensor[0], tuple)
            and isinstance(tensor[0][0], (kk.models.asp_abstract, kk.models.asf))
        ):
            energy_window = None
            tensors: list[list[kk.models.asp_abstract]] = [[], [], []]
            stoich = tensor[0][0].stoichiometry
            if stoich is None:
                if atom is None:
                    raise ValueError(
                        "The provided kkcalc2 object does not have a defined stoichiometry. "
                        "Please provide an atomic type for the AtomicScatteringTensor."
                    )
                self.atom_type = atom
            else:
                composition = stoich.composition
                if len(composition) > 1:
                    raise ValueError(
                        "The provided kkcalc2 object contains multiple atomic types. "
                        "Please provide a single atomic type for the AtomicScatteringTensor."
                    )
                else:
                    # Convert int to string label for the atom type
                    atom_num = composition[0][0]
                    if isinstance(atom_num, int):
                        self.atom_type = kk.stoichiometry._atomic_number_to_element(
                            atom_num
                        )
                    else:
                        self.atom_type = atom_num
            for i, row in enumerate(tensor):
                assert isinstance(row, tuple) and len(row) == 3, (
                    "Each row in the tuple must be a tuple of 3 kkcalc2 objects (asp or asf)."
                )
                for j, t in enumerate(row):
                    assert isinstance(t, (kk.models.asp_abstract, kk.models.asf)), (
                        "All tensors in the tuple must be kkcalc2 objects (asp or asf)."
                    )
                    tensors[i].append(
                        t if isinstance(t, kk.models.asp_abstract) else t.to_ASP()
                    )
                    if energy_window is None:
                        energy_window = (t.energies[0], t.energies[-1])
                    else:
                        energy_window = (
                            max(energy_window[0], t.energies[0]),
                            min(energy_window[1], t.energies[-1]),
                        )
                    stoich = tensor[i][j].stoichiometry
                    if stoich is None:
                        if atom is None:
                            raise ValueError(
                                f"The provided kkcalc2 object at index {i},{j} does not have a defined stoichiometry. "
                                "Please provide an atomic type for the AtomicScatteringTensor."
                            )
                        atom_type = atom
                    else:
                        composition = stoich.composition
                        if len(composition) > 1:
                            raise ValueError(
                                f"The provided kkcalc2 object at index {i} contains multiple atomic types. "
                                "Please provide a single atomic type for the AtomicScatteringTensor."
                            )
                        else:
                            # Convert int to string label for the atom type
                            atom_num = composition[0][0]
                            if isinstance(atom_num, int):
                                atom_type = kk.stoichiometry._atomic_number_to_element(
                                    atom_num
                                )
                            else:
                                atom_type = atom_num
                    if self.atom_type != atom_type:
                        raise ValueError(
                            f"The kkcalc2 object at index {i},{j} corresponds to a different atomic type ({atom_type}) than the first one ({self.atom_type})."
                        )

            tensors_tuple = tuple(tuple(row) for row in tensors)
            self._alignment = Alignment.FULL
            assert len(tensors_tuple) == 3, (
                "The tuple of kkcalc2 objects must be a 3x3 structure for full tensor."
            )
            assert (
                len(tensors_tuple[0]) == 3
                and len(tensors_tuple[1]) == 3
                and len(tensors_tuple[2]) == 3
            ), "The tuple of kkcalc2 objects must be a 3x3 structure for full tensor."
            assert all(
                isinstance(t, (kk.models.asp_abstract, kk.models.asf))
                for row in tensors_tuple
                for t in row
            ), "All elements in the 3x3 tuple must be kkcalc2 objects (asp or asf)."
            self._tensor = tensors_tuple
            if energy_window is not None:
                self.energy_domain = energy_window

        # Check if KKCalc2 objects have already defined energies and values
        if self.energy_domain is not None and self.energy_domain != (0, np.inf):
            # KKcalc2 objects have already defined
            if energies is not None:
                raise ValueError(
                    "Energies should not be provided when using kkcalc2 objects, as they already define energies."
                )
            if atom is not None and atom != self.atom_type:
                raise ValueError(
                    f"Provided atom type `{atom}` does not match the atomic type inferred from the kkcalc2 object `{self.atom_type}`."
                )
            return
        else:
            # Otherwise, define the atom type by the input string.
            self.atom_type = atom

        # Use the provided energies and atomic scattering factors to create asp objects.
        energies = np.array(energies, dtype=np.float64)
        # Remove single-dimensional entries from the shape of an array.
        if (energies.ndim == 1 and energies.shape[0] == 1) or (energies.ndim == 0):
            energies = energies.item() if energies.ndim == 0 else energies[0]
            L = 1
            assert isinstance(energies, (float, int)), (
                "Energies must be a float or int for singular energy."
            )
        else:
            L = energies.shape[0]

        self.energy_domain = (np.min(energies), np.max(energies))

        if not isinstance(tensor, (float, int, kk.models.asp_abstract)):
            tensor = np.array(tensor)

        # Singular energy values
        if L == 1:
            shape = (1,)
            if isinstance(tensor, (float, int)):
                self._alignment = Alignment.ISOTROPIC
                self._tensor = tensor
            elif tensor.ndim == 2 and tensor.shape[0] == 3 and tensor.shape[1] == 3:
                self._alignment = Alignment.FULL
                self._tensor = tensor
            elif tensor.ndim == 1 and tensor.shape == (3,):
                self._alignment = Alignment.XYZ
                self._tensor = tensor
            elif tensor.ndim == 1 and tensor.shape == (2,):
                self._alignment = Alignment.INPLANE_ISOTROPIC
                self._tensor = tensor
            else:
                raise ValueError(
                    f"Invalid tensor values provided for singular energy; {tensor}."
                )

        # Arrays
        elif isinstance(energies, np.ndarray):
            if not isinstance(tensor, (list, np.ndarray)):
                raise ValueError(
                    "Scattering tensors must be a list or numpy array when energies is an array."
                )
            shape = np.shape(tensor)
            assert energies.shape[0] == shape[0], (
                "Energies and values must have the same length."
            )
            if len(shape) == 1:
                if isinstance(tensor[0], (complex)):
                    self._alignment = Alignment.ISOTROPIC
                    tensor_re = kk.models.asf_re(energies, tensor.real)
                    tensor_im = kk.models.asf_im(energies, tensor.imag)
                    self._tensor = kk.models.asf_complex(tensor_re, tensor_im).to_ASP()
                elif isinstance(tensor[0], (float, int)):
                    self._alignment = Alignment.ISOTROPIC
                    if complexity is None:
                        raise ValueError(
                            "Complexity must be specified as 'real' or 'imag' when the tensor is a numpy array and not complex."
                        )
                    elif complexity == Complexity.REAL:
                        self._tensor = kk.models.asf_re(energies, tensor).to_ASP()
                    else:
                        self._tensor = kk.models.asf_im(energies, tensor).to_ASP()
                elif isinstance(tensor[0], (kk.models.asp_abstract)):
                    if L == 2 and len(tensor) == 2:
                        self._alignment = Alignment.INPLANE_ISOTROPIC
                    elif L == 3 and len(tensor) == 3:
                        self._alignment = Alignment.XYZ
                    else:
                        raise TypeError(
                            f"Invalid number of kkcalc2 asp tensor values ({L} different asp objects) provided."
                        )
                    self._tensor = tensor
                else:
                    raise TypeError(
                        f"Invalid tensor values provided; {type(tensor[0])}."
                    )
            elif len(shape) == 2 and shape[1] == 2:
                self._alignment = Alignment.INPLANE_ISOTROPIC
                if complexity is None:
                    raise ValueError(
                        "Complexity must be specified as 'real' or 'imag' when the tensor is a numpy array and not complex."
                    )
                elif complexity == Complexity.REAL:
                    cls = kk.models.asf_re
                else:
                    cls = kk.models.asf_im
                self._tensor = (
                    cls(energies, tensor[:, 0]).to_ASP(),
                    cls(energies, tensor[:, 1]).to_ASP(),
                )
            elif len(shape) == 2 and shape[1] == 3:
                self._alignment = Alignment.XYZ
                if np.iscomplexobj(tensor):
                    # Convert to complex asp
                    asps: list[kk.models.asp_abstract] = []
                    for i in range(3):
                        tensor_re = kk.models.asf_re(energies, tensor.real)
                        tensor_im = kk.models.asf_im(energies, tensor.imag)
                        tensor_complex = kk.models.asf_complex(
                            tensor_re, tensor_im
                        ).to_ASP()
                        asps.append(tensor_complex)
                    self._tensor = tuple(asps)
                elif complexity is None:
                    raise ValueError(
                        "Complexity must be specified as 'real' or 'imag' when the tensor is a numpy array and not complex."
                    )
                elif complexity == Complexity.REAL:
                    cls = kk.models.asf_re
                else:
                    cls = kk.models.asf_im
                self._tensor = (
                    cls(energies, tensor[:, 0]).to_ASP(),
                    cls(energies, tensor[:, 1]).to_ASP(),
                    cls(energies, tensor[:, 2]).to_ASP(),
                )
            elif len(shape) == 3 and shape[1:] == (3, 3):
                self._alignment = Alignment.FULL
                if np.iscomplexobj(tensor):
                    # Convert to complex asp
                    asps: list[list[kk.models.asp_abstract]] = [[], [], []]
                    for i in range(3):
                        for j in range(3):
                            tensor_re = kk.models.asf_re(energies, tensor.real)
                            tensor_im = kk.models.asf_im(energies, tensor.imag)
                            tensor_complex = kk.models.asf_complex(
                                tensor_re, tensor_im
                            ).to_ASP()
                            asps[i].append(tensor_complex)
                    self._tensor = tuple(tuple(row) for row in asps)
                if complexity is None:
                    raise ValueError(
                        "Complexity must be specified as 'real' or 'imag' when the tensor is a numpy array and not complex."
                    )
                elif complexity == Complexity.REAL:
                    cls = kk.models.asf_re
                else:
                    cls = kk.models.asf_im
                self._tensor = (
                    (
                        cls(energies, tensor[:, 0, 0]).to_ASP(),
                        cls(energies, tensor[:, 0, 1]).to_ASP(),
                        cls(energies, tensor[:, 0, 2]).to_ASP(),
                    ),
                    (
                        cls(energies, tensor[:, 1, 0]).to_ASP(),
                        cls(energies, tensor[:, 1, 1]).to_ASP(),
                        cls(energies, tensor[:, 1, 2]).to_ASP(),
                    ),
                    (
                        cls(energies, tensor[:, 2, 0]).to_ASP(),
                        cls(energies, tensor[:, 2, 1]).to_ASP(),
                        cls(energies, tensor[:, 2, 2]).to_ASP(),
                    ),
                )
            else:
                raise ValueError(f"Invalid tensor values provided with shape {shape}.")
        else:
            raise ValueError("Invalid tensor values provided.")

        if alignment is not None and alignment != self._alignment:
            raise ValueError(
                f"Provided alignment `{alignment}` does not match inferred alignment `{self._alignment}`"
                + f" from tensors values with shape {shape} and energies {energies}."
            )

    @property
    def tensor(self):
        """
        Get the scattering tensor values.

        Returns
        -------
        asp_tensor_type
            The scattering tensor values, which can be isotropic, in-plane isotropic, xyz, or full_tensor,
            represented as a kkcalc2 object or a tuple of kkcalc2 objects depending on the alignment.
        """
        return self._tensor

    @property
    def alignment(self) -> Alignment:
        """
        Get the alignment of the scattering tensor.

        Returns
        -------
        Alignment
            The alignment of the scattering tensor, which can be isotropic, in-plane isotropic, xyz, or full.
        """
        return self._alignment

    def __call__(self, energy: float) -> np_array_type | np_single:
        """
        Evaluate the scattering tensor at a specific energy.

        Parameters
        ----------
        energy : float
            The energy at which to evaluate the scattering tensor.

        Returns
        -------
        np_array_tensor_type
            The scattering tensor evaluated at the specified energy, which can be isotropic, in-plane isotropic, xyz, or full_tensor.
        """
        if self.energy_domain[0] == self.energy_domain[1]:
            # Single value
            if self.energy_domain[0] != energy:
                raise ValueError(
                    f"Energy {energy} is outside the defined energy range of the tensor ({self.energy_domain[0]})."
                )
            # Otherwise
            return self._tensor

        match self._alignment:
            case Alignment.ISOTROPIC:
                # Single value
                asp = self._tensor
                assert isinstance(asp, kk.models.asp_abstract), (
                    f"Isotropic tensor described by a single kkcalc2 polynomial. Was {type(asp)}"
                )
                return asp(energy)

            case Alignment.INPLANE_ISOTROPIC:
                # Two values
                tensor = self._tensor
                assert isinstance(tensor, tuple) and len(tensor) == 2, (
                    "In-plane isotropic tensor described by two kkcalc2 polynomials. "
                    f"Was {type(tensor)} with length {len(tensor) if isinstance(tensor, tuple) else 'N/A'}"
                )
                asp_inplane, asp_oop = tensor
                return np.array([asp_inplane(energy), asp_oop(energy)])

            case Alignment.XYZ:
                # Three values
                tensor = self._tensor
                assert isinstance(tensor, tuple) and len(tensor) == 3, (
                    "XYZ tensor described by three kkcalc2 polynomials. "
                    f"Was {type(tensor)} with length {len(tensor) if isinstance(tensor, tuple) else 'N/A'}"
                )
                asp_x, asp_y, asp_z = tensor
                assert (
                    isinstance(asp_x, kk.models.asp_abstract)
                    and isinstance(asp_y, kk.models.asp_abstract)
                    and isinstance(asp_z, kk.models.asp_abstract)
                ), (
                    "XYZ tensor described by three kkcalc2 polynomials. "
                    f"Was {type(asp_x)}, {type(asp_y)}, {type(asp_z)}"
                )
                return np.array([asp_x(energy), asp_y(energy), asp_z(energy)])

            case Alignment.FULL:
                # 3x3 matrix
                tensor = self._tensor
                assert isinstance(tensor, tuple) and len(tensor) == 3, (
                    "Full tensor described by a 3x3 tuple of kkcalc2 polynomials. "
                    f"Was {type(tensor)} with length {len(tensor) if isinstance(tensor, tuple) else 'N/A'}"
                )
                asp_r1, asp_r2, asp_r3 = tensor
                assert (
                    isinstance(asp_r1, tuple)
                    and isinstance(asp_r2, tuple)
                    and isinstance(asp_r3, tuple)
                ), (
                    "Full tensor described by a 3x3 tuple of kkcalc2 polynomials."
                    f"Was {type(asp_r1)}, {type(asp_r2)}, {type(asp_r3)}"
                )
                asp_xx, asp_xy, asp_xz = asp_r1
                asp_yx, asp_yy, asp_yz = asp_r2
                asp_zx, asp_zy, asp_zz = asp_r3
                return np.array(
                    [
                        [asp_xx(energy), asp_xy(energy), asp_xz(energy)],
                        [asp_yx(energy), asp_yy(energy), asp_yz(energy)],
                        [asp_zx(energy), asp_zy(energy), asp_zz(energy)],
                    ]
                )
            case _:
                raise ValueError(f"Invalid alignment: {self._alignment}")

    def copy(self) -> Self:
        """
        Create a copy of the tensor object.

        Returns
        -------
        Self
            A new instance of the tensor object with the same properties as the original.
        """
        tensor_copy: asp_array_type | np_single
        if isinstance(self._tensor, tuple):
            if isinstance(self._tensor[0], tuple):
                tensor_copy = tuple(
                    tuple(t.copy() for t in row) for row in self._tensor
                )
            else:
                tensor_copy = tuple(t.copy() for t in self._tensor)
        elif isinstance(self._tensor, kk.models.asp_abstract):
            tensor_copy = self._tensor.copy()
        else:
            if isinstance(self._tensor, np.ndarray):
                tensor_copy = self._tensor.copy()
                # Also
            else:
                # int/float
                tensor_copy = self._tensor
            # Also need to copy the energy window, but since it's a single value, we can just use the same value.
            return self.__class__(
                atom=self.atom_type,
                energies=self.energy_domain[0],
                tensor=tensor_copy,
                alignment=self._alignment,
            )
        # Otherwise
        obj = self.__class__(
            atom=self.atom_type,
            energies=None,
            tensor=tensor_copy,
            alignment=self._alignment,
        )
        return obj


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
        self.energy_domain: tuple[float, float] = (0, np.inf)
        """The valid energy range for the scattering tensors, defined as a tuple (min_energy, max_energy)."""
        self.scattering_factors: dict[str, AtomicScatteringTensor] = {}
        """A dictionary mapping atomic types to their corresponding scattering tensors."""
        if scattering_factors is not None:
            # Collect energies from the first tensor to ensure consistency
            tensor = (
                next(iter(scattering_factors.values()))  # dict
                if isinstance(scattering_factors, dict)
                else scattering_factors[0]  # list
            )
            self.energy_domain = energy_domain = tensor.energy_domain

            if isinstance(scattering_factors, dict):
                for atom, tensor in (scattering_factors or {}).items():
                    # Restrict the new energy domain, and complain if no overlap.
                    other_domain = tensor.energy_domain
                    if (
                        other_domain[0] > energy_domain[1]
                        or other_domain[1] < energy_domain[0]
                    ):
                        raise ValueError(
                            f"Energy domains of scattering tensors for atom '{atom}' do not overlap. "
                            f"Existing domain: {energy_domain}, new domain: {other_domain}."
                        )
                    # Check if the atom type already exists in the scattering factors
                    if atom in self.scattering_factors:
                        raise ValueError(
                            f"Duplicate scattering tensor for atom type '{atom}' found."
                        )
                    # Modify the current domain
                    self.energy_domain = (
                        max(energy_domain[0], other_domain[0]),
                        min(energy_domain[1], other_domain[1]),
                    )
                    # Add the item.
                    self.__setitem__(atom, tensor)

            elif isinstance(scattering_factors, list):
                for tensor in scattering_factors:
                    # Restrict the new energy domain, and complain if no overlap.
                    other_domain = tensor.energy_domain
                    if (
                        other_domain[0] > energy_domain[1]
                        or other_domain[1] < energy_domain[0]
                    ):
                        raise ValueError(
                            f"Energy domains of scattering tensors for atom '{tensor.atom_type}' do not overlap. "
                            f"Existing domain: {energy_domain}, new domain: {other_domain}."
                        )
                    # Modify the current domain
                    self.energy_domain = (
                        max(energy_domain[0], other_domain[0]),
                        min(energy_domain[1], other_domain[1]),
                    )
                    # Add the item
                    if tensor.atom_type in self.scattering_factors:
                        raise ValueError(
                            f"Duplicate scattering tensor for atom type '{tensor.atom_type}' found."
                        )
                    self.__setitem__(tensor.atom_type, tensor)
            else:
                raise TypeError(
                    "scattering_factors must be a dictionary or a list of AtomicScatteringTensor objects."
                )

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
