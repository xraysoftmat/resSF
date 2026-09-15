"""
Shared pytest fixtures for testing ``structures.py``.
"""

from pathlib import Path

import pytest

ase = pytest.importorskip("ase")
Atoms = ase.Atoms


@pytest.fixture
def atom_records():
    """A deliberately unsorted, mixed-element internal representation."""
    return [
        ("O", (1.0, 0.0, 0.0)),
        ("H", (0.0, 1.0, 0.0)),
        ("C", (0.0, 0.0, 1.0)),
        ("H", (0.0, 0.0, 0.0)),
    ]


@pytest.fixture
def ase_atoms():
    """An ASE object whose Cartesian positions are easy to verify."""
    return Atoms(
        symbols=["C", "O", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.25, 2.5, 3.75),
            (-1.0, 0.5, 2.0),
        ],
    )


@pytest.fixture
def empty_cif(tmp_path: Path) -> Path:
    """An existing placeholder path for mocking pymatgen's CIF parser."""
    path = tmp_path / "input.cif"
    path.write_text("# parser is mocked by the test\n", encoding="utf-8")
    return path


@pytest.fixture
def example_cif(tmp_path: Path) -> Path:
    """A CIF file with a P3HT structure for testing."""
    cif_content = """
    data_P3HT

    _audit_creation_method           'Hand-converted from SHELX .ins'
    _chemical_name_systematic        'Poly(3-hexylthiophene)'
    _symmetry_Int_Tables_number      14
    _symmetry_space_group_name_H-M   'P 21/c'
    _chemical_formula_structural     '(C20 H28 2S)n'
    _chemical_formula_sum            'C10 H14 S'

    # Unit cell parameters (example values in Angstroms and degrees)
    _cell_length_a                    16.000
    _cell_length_b                    07.800
    _cell_length_c                    07.800
    _cell_angle_alpha                 90.000
    _cell_angle_beta                  90.000
    _cell_angle_gamma                 86.500

    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    _atom_site_adp_type
    _atom_site_U_iso_or_equiv

    S1 S -0.44468 0.81314 0.34655 1.00000 Uiso 0.05000
    C2 C -0.50672 0.73694 0.18526 1.00000 Uiso 0.05000
    C3 C -0.57534 0.66367 0.25306 1.00000 Uiso 0.05000
    C4 C -0.57250 0.66829 0.43431 1.00000 Uiso 0.05000
    C5 C -0.51221 0.74937 0.50483 1.00000 Uiso 0.05000
    C6 C -0.63745 0.60984 0.53010 1.00000 Uiso 0.00000
    C7 C -0.70958 0.66251 0.44725 1.00000 Uiso 0.00000
    C8 C -0.76646 0.73348 0.56408 1.00000 Uiso 0.00000
    C9 C -0.83177 0.79791 0.45973 1.00000 Uiso 0.00000
    C10 C -0.89628 0.87833 0.55295 1.00000 Uiso 0.00000
    C11 C -0.95710 0.93574 0.43509 1.00000 Uiso 0.00000
    H12 H -0.61595 0.61934 0.17913 1.00000 Uiso 0.00000
    H13 H -0.63163 0.48809 0.53725 1.00000 Uiso 0.00000
    H14 H -0.63707 0.65831 0.64181 1.00000 Uiso 0.00000
    H15 H -0.69786 0.74597 0.36335 1.00000 Uiso 0.00000
    H16 H -0.73235 0.56684 0.39303 1.00000 Uiso 0.00000
    H17 H -0.78295 0.64677 0.64025 1.00000 Uiso 0.00000
    H18 H -0.74299 0.82379 0.62584 1.00000 Uiso 0.00000
    H19 H -0.81143 0.87930 0.38200 1.00000 Uiso 0.00000
    H20 H -0.85288 0.70598 0.39642 1.00000 Uiso 0.00000
    H21 H -0.91795 0.79874 0.63091 1.00000 Uiso 0.00000
    H22 H -0.87648 0.97310 0.61396 1.00000 Uiso 0.00000
    H23 H -1.01087 0.93481 0.48661 1.00000 Uiso 0.00000
    H24 H -0.94711 1.04938 0.40034 1.00000 Uiso 0.00000
    H25 H -0.95483 0.86220 0.33789 1.00000 Uiso 0.00000
    """
    cif_path = tmp_path / "example.cif"
    cif_path.write_text(cif_content.strip(), encoding="utf-8")
    return cif_path
