"""
Representations of crystal structures and molecules, including atomic types and positions.

Based on using pyXtal, which doesn't have energy dependent scattering factors.
"""

from pathlib import Path
from typing import Self
from ase import Atoms, Atom
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
    """

    def __init__(self, atoms: list[tuple[str, tuple[float, float, float]]] | Atoms):
        if isinstance(atoms, Atoms):
            # Convert Atoms object to the internal representation
            atoms = [(atom.symbol, tuple(atom.position)) for atom in atoms]
        self.atoms = atoms

    @staticmethod
    def from_cif(filepath: str | Path) -> "structure":
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"CIF file not found: {filepath}")

        # Use pymatgen to read the CIF file and extract atomic types and positions
        pmg = pmgStruct.from_file(str(filepath))
        # Convert to the internal representation
        atoms = [(site.specie.symbol, tuple(site.frac_coords)) for site in pmg]
        return structure(atoms)

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
        #
        expected_type = df.iloc[i]["Type"]
        expected_position = df.iloc[i][["X", "Y", "Z"]].values
        assert atom[0] == expected_type, (
            f"Atom type mismatch at index {i}: expected {expected_type}, got {atom[0]}"
        )
        assert all(abs(atom[1][j] - expected_position[j]) < 1e-5 for j in range(3)), (
            f"Atom position mismatch at index {i}: expected {expected_position}, got {atom}"
        )
