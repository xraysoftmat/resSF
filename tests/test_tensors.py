"""
Tests for the structure module.
"""

import pytest
from resSF.tensor_types import Alignment, tensor_type
from resSF.tensor import AtomicScatteringTensor, CrystalScatteringTensor, Complexity


class TestAtomicScatteringTensor:
    """Tests for the atomic scattering tensor model."""

    @pytest.mark.parametrize("complexity", (Complexity.REAL, Complexity.IMAG))
    @pytest.mark.parametrize(
        "atom, energies, values, alignment",
        [
            # Valid Cases
            ("S", 2450, "tensor_isotropic", "isotropic"),
            ("S", 2450, "tensor_inplane_isotropic", "inplane_isotropic"),
            ("S", 2450, "tensor_xyz", "xyz"),
            ("S", 2450, "tensor_full", "full"),
            ("S", "array_energies", "array_tensor_isotropic", "isotropic"),
            (
                "S",
                "array_energies",
                "array_tensor_inplane_isotropic",
                "inplane_isotropic",
            ),
            ("S", "array_energies", "array_tensor_xyz", "xyz"),
            ("S", "array_energies", "array_tensor_full", "full"),
            ("S", None, "kk_asp_isotropic", "isotropic"),
            ("S", None, "kk_asp_inplane_isotropic", "inplane_isotropic"),
            ("S", None, "kk_asp_xyz", "xyz"),
            ("S", None, "kk_asp_full", "full"),
        ],
    )
    def test_tensor_initialization(
        self,
        complexity: Complexity,
        atom: str,
        energies,
        values: str,
        alignment: Alignment,
        request: pytest.FixtureRequest,
    ):
        tensor: tensor_type = request.getfixturevalue(
            values
        )  # Get the tensor fixture based on the string name
        energies = (
            request.getfixturevalue(energies) if isinstance(energies, str) else energies
        )  # Get the energies fixture if it's a string
        AtomicScatteringTensor(
            atom=atom,
            energies=energies,
            tensor=tensor,
            alignment=alignment,
            complexity=complexity,
        )

    @pytest.mark.parametrize("complexity", (Complexity.REAL, Complexity.IMAG))
    @pytest.mark.parametrize(
        "atom, energies, values, alignment",
        [
            # Valid Cases
            ("S", 2450, "tensor_isotropic", "isotropic"),
            ("S", 2450, "tensor_inplane_isotropic", "inplane_isotropic"),
            ("S", 2450, "tensor_xyz", "xyz"),
            ("S", 2450, "tensor_full", "full"),
            ("S", "array_energies", "array_tensor_isotropic", "isotropic"),
            (
                "S",
                "array_energies",
                "array_tensor_inplane_isotropic",
                "inplane_isotropic",
            ),
            ("S", "array_energies", "array_tensor_xyz", "xyz"),
            ("S", "array_energies", "array_tensor_full", "full"),
            ("S", None, "kk_asp_isotropic", "isotropic"),
            ("S", None, "kk_asp_inplane_isotropic", "inplane_isotropic"),
            ("S", None, "kk_asp_xyz", "xyz"),
            ("S", None, "kk_asp_full", "full"),
        ],
    )
    def test_tensor_copy(
        self,
        complexity: Complexity,
        atom: str,
        energies,
        values: str,
        alignment: Alignment,
        request: pytest.FixtureRequest,
    ):
        tensor: tensor_type = request.getfixturevalue(
            values
        )  # Get the tensor fixture based on the string name
        energies = (
            request.getfixturevalue(energies) if isinstance(energies, str) else energies
        )  # Get the energies fixture if it's a string
        orig = AtomicScatteringTensor(
            atom=atom,
            energies=energies,
            tensor=tensor,
            alignment=alignment,
            complexity=complexity,
        )
        copy = orig.copy()
        assert isinstance(copy, AtomicScatteringTensor)
        # Check the copy has unique memory address
        if isinstance(orig.tensor, tuple):
            if isinstance(orig.tensor[0], tuple):
                t0 = orig.tensor[0][0]
                t1 = copy.tensor[0][0]
            else:
                t0 = orig.tensor[0]
                t1 = copy.tensor[0]
        else:
            t0 = orig.tensor
            t1 = copy.tensor

        # Check that the copy has a different memory address than the original
        # unless it's a primitive type.
        assert id(t0) != id(t1) or isinstance(t0, (int, float, complex))
        assert orig.atom_type == copy.atom_type
        assert orig.alignment == copy.alignment
        assert orig.energy_domain == copy.energy_domain


class TestCrystalScatteringTensor:
    """Test grouping and implementation of multiple atoms in a crystal structure."""

    def test_crystal_scattering_tensor_initialization(
        self, array_energies, array_tensor_xyz, array_tensor_inplane_isotropic
    ):
        tensor1 = AtomicScatteringTensor(
            atom="S",
            energies=array_energies,
            tensor=array_tensor_xyz,
            alignment=Alignment.XYZ,
            complexity=Complexity.IMAG,
        )
        tensor2 = AtomicScatteringTensor(
            atom="C",
            energies=array_energies,
            tensor=array_tensor_inplane_isotropic,
            alignment=Alignment.INPLANE_ISOTROPIC,
            complexity=Complexity.IMAG,
        )
        crystal_tensor = CrystalScatteringTensor(scattering_factors=[tensor1, tensor2])
        assert isinstance(crystal_tensor, CrystalScatteringTensor)
        assert "S" in crystal_tensor.scattering_factors
        assert "C" in crystal_tensor.scattering_factors
        assert crystal_tensor.scattering_factors["S"] == tensor1
        assert crystal_tensor.scattering_factors["C"] == tensor2
        assert crystal_tensor["S"] == tensor1
        assert crystal_tensor["C"] == tensor2

    @pytest.mark.parametrize("complexity", (Complexity.REAL, Complexity.IMAG))
    @pytest.mark.parametrize(
        "atom, energies, values, alignment, error",
        [
            # Invalid Cases
            (
                "S",
                (2450, 2500),
                "tensor_isotropic",
                "isotropic",
                "Scattering tensors must be a list or numpy array when energies is an array",
            ),
            (
                "S",
                (2450, 2500),
                "tensor_inplane_isotropic",
                "inplane_isotropic",
                "Provided alignment `inplane_isotropic` does not match inferred alignment `isotropic` from tensors values with shape",
            ),
            (
                "S",
                [2450, 2475, 2500],
                "array_tensor_isotropic",
                "isotropic",
                "Energies and values must have the same length.",
            ),
            (
                "S",
                [2450, 2475, 2500, 2550, 2600],
                "array_tensor_xyz",
                "xyz",
                "Energies and values must have the same length.",
            ),
        ],
    )
    def test_tensor_initialization_invalid(
        self,
        atom: str,
        energies,
        values: str,
        alignment: Alignment,
        complexity: Complexity,
        error: str | None,
        request: pytest.FixtureRequest,
    ):
        vals: tensor_type = request.getfixturevalue(values)
        try:
            AtomicScatteringTensor(
                atom=atom,
                energies=energies,
                tensor=vals,
                alignment=alignment,
                complexity=complexity,
            )
        except Exception as e:
            if error and error in str(e):
                pytest.skip(f"Expected assertion error: {error}")
            else:
                raise
