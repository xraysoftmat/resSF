"""
Representations of crystal structures and molecules, including atomic types and positions.

Based on using pyXtal, which doesn't have energy dependent scattering factors.
"""

# StdLib
from pathlib import Path
from typing import Self

# Third-party
from ase import Atom, Atoms
from pymatgen.core.lattice import Lattice as pmgLattice
from pymatgen.core.structure import Structure as pmgStruct


class structure:
    """
    The atomic basis of a crystal / molecule.

    Contains atomic types and positions, and can be used to calculate structure factors.

    Parameters
    ----------
    atoms : list of tuples or Atoms object
        Each tuple contains the atomic type (str) and its position (tuple of 3 floats).
        If an Atoms object is provided, it will be converted to the internal representation.
    unitcell_lengths : tuple[float, float, float] | None, optional
        The lengths of the unit cell (in Angstroms) along the a, b, and c axes.
        By default, lengths are None.
        Not necessary for calculating structure factors, but may be useful for visualisation.
    angles : tuple[float, float, float] | None, optional
        The interaxial angles (in degrees) of rotation; alpha, beta and gamma.
        alpha is the angle between b and c, beta is the angle between a and c,
        and gamma is the angle between a and b. By default, angles are None.
        Not necessary for calculating structure factors, but may be useful for visualisation.
    """

    def __init__(
        self,
        atoms: list[tuple[str, tuple[float, float, float]]] | Atoms,
        unitcell_lengths: tuple[float, float, float] | None = None,
        angles: tuple[float, float, float] | None = None,
    ):
        if isinstance(atoms, Atoms):
            # Convert Atoms object to the internal representation
            atoms = [(atom.symbol, tuple(atom.position)) for atom in atoms]
        self.atoms = atoms
        """The atomic basis of the structure, as a list of tuples (type, position)."""
        self.unitcell_lengths = unitcell_lengths
        """The lengths of the unit cell (in Angstroms) along the a, b, and c axes."""
        self.angles = angles
        """The interaxial angles (in degrees) of rotation; alpha, beta and gamma."""

    @classmethod
    def from_cif(cls, filepath: str | Path) -> Self:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"CIF file not found: {filepath}")

        # Use pymatgen to read the CIF file and extract atomic types and positions
        pmg = pmgStruct.from_file(str(filepath))
        # Convert to the internal representation
        atoms = [(site.specie.symbol, tuple(site.frac_coords)) for site in pmg]
        lengths = tuple(pmg.lattice.abc)
        angles = tuple(pmg.lattice.angles)
        assert len(angles) == 3, (
            f"Expected 3 angles in the CIF file, but found {len(angles)}"
        )
        assert len(lengths) == 3, (
            f"Expected 3 unit cell lengths in the CIF file, but found {len(lengths)}"
        )

        return cls(atoms, unitcell_lengths=lengths, angles=angles)

    def __len__(self) -> int:
        return len(self.atoms)

    def __getitem__(self, index: int) -> tuple[str, tuple[float, float, float]]:
        return self.atoms[index]

    def formula(self) -> str:
        """
        Generate the chemical formula of the structure.

        Returns
        -------
        str
            The chemical formula as a string.
        """
        from collections import Counter

        counts = Counter(atom[0] for atom in self.atoms)
        formula = "".join(
            f"{elem}{count if count > 1 else ''}"
            for elem, count in sorted(counts.items())
        )
        return formula

    def __repr__(self) -> str:
        return f"structure(atoms={self.formula()})"

    def __iter__(self):
        return iter(self.atoms)

    def to_ase(self) -> Atoms:
        """
        Convert the internal representation to an ASE Atoms object.

        Returns
        -------
        Atoms
            An ASE Atoms object representing the structure.
        """
        ase_atoms = [Atom(symbol=atom[0], position=atom[1]) for atom in self.atoms]
        return Atoms(ase_atoms)

    def to_cif(self, filepath: str | Path) -> None:
        """
        Save the structure to a CIF file.

        Parameters
        ----------
        filepath : str or Path
            The path to the output CIF file.
        """
        ase_atoms = self.to_ase()
        ase_atoms.write(str(filepath), format="cif")

    # Sort method
    def sort(self, key: str = "type") -> None:
        """
        Sort the atoms in the structure.

        Parameters
        ----------
        key : str
            The sorting key. Can be "type" for atomic type or "position" for atomic position.
        """
        if key == "type":
            self.atoms = sorted(
                self.atoms, key=lambda x: (x[0], x[1][0], x[1][1], x[1][2])
            )
        elif key == "position":
            self.atoms = sorted(
                self.atoms, key=lambda x: (x[1][0], x[1][1], x[1][2], x[0])
            )
        else:
            raise ValueError(f"Invalid sort key: {key}. Use 'type' or 'position'.")

    def copy(self) -> Self:
        """
        Create a copy of the structure.

        Returns
        -------
        structure
            A new instance of the structure with the same atoms.
        """
        return self.__class__(self.atoms.copy())

    def cartesian_positions(self) -> list[tuple[str, tuple[float, float, float]]]:
        """
        Get the Cartesian positions of the atoms.

        Returns
        -------
        list of tuples
            A list of tuples representing the Cartesian positions of the atoms.
        """
        lengths = self.unitcell_lengths
        if lengths is None:
            raise ValueError(
                "Unit cell lengths must be set to convert fractional coordinates to Cartesian positions."
            )
        angles = self.angles
        if angles is None:
            raise ValueError(
                "Angles must be set to convert fractional coordinates to Cartesian positions."
            )
        # Convert fractional coordinates to Cartesian using pymatgen
        lattice = pmgLattice.from_parameters(*lengths, *angles)
        cart_coords = [
            (atom_type, tuple(lattice.get_cartesian_coords(frac_coord)))
            for atom_type, frac_coord in self.atoms
        ]
        return cart_coords


if __name__ == "__main__":
    import os

    # Example usage
    cif_path = os.path.join(os.getcwd(), "tests/data/P3HT/Kayunkid_P3HT.cif")
    crystal_structure = structure.from_cif(cif_path)
    print(crystal_structure)
    # Check number of atoms
    print(f"Number of atoms: {len(crystal_structure)}")
    # Check atom positions match the atoms in "Kayunkid P3HT Model Expanded.txt" [Type, X, Y, Z]
    alternative_path = os.path.join(
        os.getcwd(), "tests/data/P3HT/Kayunkid P3HT Model Expanded.txt"
    )
    # Load
    import pandas as pd

    df = pd.read_csv(
        alternative_path, header=None, delimiter=r"\s+", names=["Type", "X", "Y", "Z"]
    )
    # Sort the dataframe by Type and then by X, Y, Z
    df = df.sort_values(by=["Type", "X", "Y", "Z"])
    print(df.head())
    # Compare
    # Sort the crystal structure
    crystal_structure.sort(key="type")
    # Convert crystal_structure to a DataFrame for comparison
    df_cry_data = [
        (atom[0], atom[1][0], atom[1][1], atom[1][2]) for atom in crystal_structure
    ]
    df_crystal = pd.DataFrame(df_cry_data, columns=["Type", "X", "Y", "Z"])
    print(df_crystal.head())
    # Save to file
    pathout = os.path.join(os.path.dirname(__file__), "crystal_structure.txt")
    df_crystal.to_csv(pathout, sep="\t", index=False)

    for i, atom in enumerate(crystal_structure):
        print(f"Atom {i}: {atom[0]} at position {atom[1]}")

        expected_type = df.iloc[i]["Type"]
        expected_position = df.iloc[i][["X", "Y", "Z"]].values
        assert atom[0] == expected_type, (
            f"Atom type mismatch at index {i}: expected {expected_type}, got {atom[0]}"
        )
        assert all(abs(atom[1][j] - expected_position[j]) < 1e-5 for j in range(3)), (
            f"Atom position mismatch at index {i}: expected {expected_position}, got {atom}"
        )
