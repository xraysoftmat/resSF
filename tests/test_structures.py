"""
Generic unit tests for the public behaviour of ``structures.structure``.
"""

# Stdlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# External
import pytest

# Internal
from resSF.structures import structure

# ruff: disable[F401]
from tests.fixtures.structures import (
    ase_atoms,  # noqa: F401
    atom_records,  # noqa: F401
    example_cif,  # noqa: F401
    example_cif_cartesian,  # noqa: F401
)

# ruff: enable[F401]

pytest_plugins = ["tests.fixtures"]  # Prevent ruff auto-removing the fixtures import


class TestStructure:
    @staticmethod
    def test_init_from_internal_records_preserves_values_and_protocols(atom_records):  # noqa: F811
        obj = structure(atom_records)

        assert len(obj) == 4
        assert obj[0] == atom_records[0]
        assert obj[-1] == atom_records[-1]
        assert list(obj) == atom_records

    @staticmethod
    def test_init_from_ase_converts_symbols_and_positions(ase_atoms):  # noqa: F811
        obj = structure(ase_atoms)

        assert [symbol for symbol, _ in obj] == ["C", "O", "H"]
        assert [position for _, position in obj] == pytest.approx(
            [(0.0, 0.0, 0.0), (1.25, 2.5, 3.75), (-1.0, 0.5, 2.0)]
        )
        assert all(isinstance(position, tuple) for _, position in obj)

    @pytest.mark.parametrize(
        ("records", "expected"),
        [
            ([], ""),
            ([("C", (0, 0, 0))], "C"),
            (
                [("O", (0, 0, 0)), ("H", (0, 0, 1)), ("H", (0, 1, 0))],
                "H2O",
            ),
            (
                [("Na", (0, 0, 0)), ("Cl", (0, 0, 1)), ("Na", (0, 1, 0))],
                "ClNa2",
            ),
        ],
    )
    @staticmethod
    def test_formula_is_alphabetical_and_omits_count_one(records, expected):
        """
        Tests that the `formula` method returns a chemical formula string that is alphabetical
        """

        assert structure(records).formula() == expected

    @staticmethod
    def test_repr_uses_formula(atom_records):  # noqa: F811
        """
        Tests that the `__repr__` method returns a string representation of the structure
        that includes the chemical formula.
        """
        assert repr(structure(atom_records)) == "structure(atoms=CH2O)"

    @staticmethod
    def test_to_ase_preserves_symbols_and_cartesian_positions(atom_records):  # noqa: F811
        """
        Tests that the `to_ase` method preserves the symbols and Cartesian positions of the atoms.
        """

        converted = structure(atom_records).to_ase()

        assert converted.get_chemical_symbols() == ["O", "H", "C", "H"]
        assert np.isclose(
            converted.get_positions().tolist(),
            [list(position) for _, position in atom_records],
        ).all()

    @staticmethod
    def test_to_cif_delegates_to_ase_write(monkeypatch, tmp_path, atom_records):  # noqa: F811
        """
        Tests that the `to_cif` method delegates to the `write` method of the ASE object.
        """

        obj = structure(atom_records)
        target = tmp_path / "output.cif"
        sentinel_ase = SimpleNamespace()
        calls = []

        sentinel_ase.write = lambda filepath, format=None: calls.append(
            (filepath, format)
        )
        monkeypatch.setattr(obj, "to_ase", lambda: sentinel_ase)

        result = obj.to_cif(target)

        assert result is None
        assert calls == [(str(target), "cif")]

    @staticmethod
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (
                "type",
                [
                    ("C", (0.0, 0.0, 1.0)),
                    ("H", (0.0, 0.0, 0.0)),
                    ("H", (0.0, 1.0, 0.0)),
                    ("O", (1.0, 0.0, 0.0)),
                ],
            ),
            (
                "position",
                [
                    ("H", (0.0, 0.0, 0.0)),
                    ("C", (0.0, 0.0, 1.0)),
                    ("H", (0.0, 1.0, 0.0)),
                    ("O", (1.0, 0.0, 0.0)),
                ],
            ),
        ],
    )
    def test_sort_orders_in_place(atom_records, key, expected):  # noqa: F811
        """
        Tests that the `sort` method orders the atoms in place according to the specified key.
        """

        obj = structure(atom_records.copy())

        result = obj.sort(key=key)

        assert result is None
        assert obj.atoms == expected

    @staticmethod
    def test_sort_rejects_unknown_key(atom_records):  # noqa: F811
        """
        Tests that the `sort` method raises a `ValueError` when provided with an unknown sort key.
        """

        with pytest.raises(ValueError, match="Invalid sort key: mass"):
            structure(atom_records).sort(key="mass")

    @staticmethod
    def test_copy_creates_independent_atom_list(atom_records):  # noqa: F811
        """
        Tests that the `copy` method creates an independent copy of the atom list.
        """

        original = structure(atom_records.copy())
        copied = original.copy()

        assert copied is not original
        assert copied.atoms == original.atoms
        assert copied.atoms is not original.atoms

        copied.atoms.append(("N", (9.0, 9.0, 9.0)))
        assert len(copied) == len(original) + 1

    @staticmethod
    def test_copy_preserves_subclass():
        """
        Tests that the `copy` method preserves the subclass of the original object.
        """

        class DerivedStructure(structure):
            pass

        original = DerivedStructure([("C", (0.0, 0.0, 0.0))])
        assert isinstance(original.copy(), DerivedStructure)

    @staticmethod
    def test_from_cif_rejects_missing_path(tmp_path):
        """
        Tests that the `from_cif` method raises a `FileNotFoundError` when provided with a non-existent CIF file path.
        """

        missing = tmp_path / "missing.cif"

        with pytest.raises(FileNotFoundError, match="CIF file not found"):
            structure.from_cif(missing)

    @staticmethod
    def test_from_cif_returns_subclass_instance(example_cif):  # noqa: F811
        """
        Tests if the `from_cif` method returns an instance of the subclass when called on a subclass of `structure`.
        """

        class DerivedStructure(structure):
            pass

        result = DerivedStructure.from_cif(Path(example_cif))

        assert isinstance(result, DerivedStructure)

    @staticmethod
    def test_fractional_coordinates_return_correct_cartesian_positions(
        example_cif,  # noqa: F811
        example_cif_cartesian,  # noqa: F811
    ):
        """
        Tests that the `cartesian_positions` method returns the correct Cartesian positions
        for a structure defined with fractional coordinates.
        """
        fractional_structure = structure.from_cif(Path(example_cif))
        result = fractional_structure.cartesian_positions()
        print("Resulting Cartesian positions:", result)
        print("Expected Cartesian positions:", example_cif_cartesian)

        # Sort by element, then by the x, y, z coordinates to ensure consistent ordering for comparison
        result.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]))
        example_cif_cartesian.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]))

        for result_atom, expected_atom in zip(result, example_cif_cartesian):
            assert result_atom[0] == expected_atom[0]
            assert np.allclose(result_atom[1], expected_atom[1:], atol=1e-6), (
                f"Expected {expected_atom[1:]} but got {result_atom[1:]}"
            )
