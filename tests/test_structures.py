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


class TestStructure:
    @staticmethod
    def test_init_from_internal_records_preserves_values_and_protocols(atom_records):
        obj = structure(atom_records)

        assert len(obj) == 4
        assert obj[0] == atom_records[0]
        assert obj[-1] == atom_records[-1]
        assert list(obj) == atom_records

    @staticmethod
    def test_init_from_ase_converts_symbols_and_positions(ase_atoms):
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
    def test_repr_uses_formula(atom_records):
        """
        Tests that the `__repr__` method returns a string representation of the structure
        that includes the chemical formula.
        """
        assert repr(structure(atom_records)) == "structure(atoms=CH2O)"

    @staticmethod
    def test_to_ase_preserves_symbols_and_cartesian_positions(atom_records):
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
    def test_to_cif_delegates_to_ase_write(monkeypatch, tmp_path, atom_records):
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
    def test_sort_orders_in_place(atom_records, key, expected):
        """
        Tests that the `sort` method orders the atoms in place according to the specified key.
        """

        obj = structure(atom_records.copy())

        result = obj.sort(key=key)

        assert result is None
        assert obj.atoms == expected

    @staticmethod
    def test_sort_rejects_unknown_key(atom_records):
        """
        Tests that the `sort` method raises a `ValueError` when provided with an unknown sort key.
        """

        with pytest.raises(ValueError, match="Invalid sort key: mass"):
            structure(atom_records).sort(key="mass")

    @staticmethod
    def test_copy_creates_independent_atom_list(atom_records):
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
    def test_from_cif_returns_subclass_instance(example_cif):
        """
        Tests if the `from_cif` method returns an instance of the subclass when called on a subclass of `structure`.
        """

        class DerivedStructure(structure):
            pass

        result = DerivedStructure.from_cif(Path(example_cif))

        assert isinstance(result, DerivedStructure)
