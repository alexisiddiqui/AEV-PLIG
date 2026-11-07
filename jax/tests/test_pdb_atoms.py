"""PDB pocket reader (``LoadPDBasDF`` in generate_pdbbind_graphs.py, plus the pocket annotations).

tests/data/pocket.pdb is hand-written. Chain A: ALA 1, SER 2 (OG in altlocs A and B), GLY 52,
ASN 52A (insertion code), MSE 53 (one ATOM, ``MSE-SE``, not in the key table); chain B: VAL 1,
LYS 2; then a HETATM water 1.9 A from the ligand. Hydrogens: ``H``, ``HB2``, ``HG`` (3 characters),
``HD21``, ``HD22`` (4 characters, H first) and ``1HG1`` (4 characters, H second). Each B-factor is
10 + serial, except GLY 52 O (serial 19), whose field is blank. Against ``LIG`` the LYS NZ
(serial 48) is 11.0 A from the nearest ligand atom, the LYS CE (47) 9.5 A, every other atom < 9 A.
"""

from pathlib import Path

import numpy as np
import pytest

from aevplig_jax.constants import species_table
from aevplig_jax.pdb_atoms import (
    STANDARD_RESIDUES,
    PocketArrays,
    atom_name_table,
    _within,
    read_pocket,
    typed_pairs,
    typed_within,
)

DATA = Path(__file__).resolve().parent / "data"
POCKET = DATA / "pocket.pdb"
KEYS_CSV = Path(__file__).resolve().parents[2] / "data" / "PDB_Atom_Keys.csv"

LIG = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]])

# Atoms kept with the default 10 A radius, in serial order.
KEPT_SERIALS = [1, 2, 3, 4, 5,
                8, 9, 10, 11, 12, 13, 14,
                16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27,
                30,
                32, 33, 34, 35, 36, 37, 38,
                40, 41, 42, 43, 44, 45, 46, 47]
KEPT_KEYS = ["ALA-N", "ALA-CA", "ALA-C", "ALA-O", "ALA-CB",
             "SER-N", "SER-CA", "SER-C", "SER-O", "SER-CB", "SER-OG", "SER-OG",
             "GLY-N", "GLY-CA", "GLY-C", "GLY-O",
             "ASN-N", "ASN-CA", "ASN-C", "ASN-O", "ASN-CB", "ASN-CG", "ASN-OD1", "ASN-ND2",
             "MSE-SE",
             "VAL-N", "VAL-CA", "VAL-C", "VAL-O", "VAL-CB", "VAL-CG1", "VAL-CG2",
             "LYS-N", "LYS-CA", "LYS-C", "LYS-O", "LYS-CB", "LYS-CG", "LYS-CD", "LYS-CE"]
# Typed atoms (species > 0) within 5.1 A of a ligand atom; the MSE SE (4.1 A) is untyped and the
# ALA C (5.24 A), SER N (5.23 A), ASN N (5.32 A) and VAL CB (5.34 A) are just outside.
TYPED_WITHIN_SERIALS = [1, 2, 5, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35]


@pytest.fixture(scope="module")
def species():
    if not KEYS_CSV.exists():
        pytest.skip(f"{KEYS_CSV} not present (code-only mirror)")
    return species_table(KEYS_CSV)


def _bfactor_of(serial):
    return 0.0 if serial == 19 else 10.0 + serial


def _assert_layout(pocket: PocketArrays, n: int):
    assert pocket.xyz.shape == (n, 3) and pocket.xyz.dtype == np.float32
    expected = {"species": np.int8, "res": np.int32, "resname": np.int8, "atom": np.int16,
                "chain": np.int8, "bfactor": np.float16}
    for field, dtype in expected.items():
        array = getattr(pocket, field)
        assert array.shape == (n,) and array.dtype == dtype, field


def _write_pdb(tmp_path, lines):
    path = tmp_path / "fragment.pdb"
    path.write_text("".join(line + "\n" for line in lines))
    return path


def _load_pdb_as_df(PDB, atom_keys):
    """``LoadPDBasDF`` copied from generate_pdbbind_graphs.py:43-63 (warning print left out)."""
    import pandas as pd

    prot_atoms = []

    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])

    f.close()

    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    return df


def test_filters_like_loadpdbasdf(species):
    pocket = read_pocket(POCKET, LIG, species)
    _assert_layout(pocket, len(KEPT_SERIALS))
    # B-factors encode serials: HETATM water (49) and hydrogens (6, 7, 15, 28, 29, 39) are absent.
    np.testing.assert_array_equal(pocket.bfactor, [_bfactor_of(s) for s in KEPT_SERIALS])
    np.testing.assert_array_equal(pocket.species, [species.get(key, 0) for key in KEPT_KEYS])
    assert "MSE-SE" not in species
    assert pocket.species[KEPT_KEYS.index("MSE-SE")] == 0
    altlocs = [i for i, key in enumerate(KEPT_KEYS) if key == "SER-OG"]
    np.testing.assert_array_equal(pocket.xyz[altlocs],
                                  np.float32([[6.4, 3.2, -3.3], [8.6, 3.4, -2.7]]))
    np.testing.assert_array_equal(pocket.species[altlocs], [22, 22])


def test_typed_atoms_match_loadpdbasdf_copy(species):
    pd = pytest.importorskip("pandas")
    atom_keys = pd.read_csv(KEYS_CSV)
    target = _load_pdb_as_df(POCKET, atom_keys)
    type_index = {name: i + 1 for i, name in enumerate(dict.fromkeys(atom_keys["ATOM_TYPE"]))}

    pocket = read_pocket(POCKET, LIG, species, radius=np.inf)
    typed = pocket.species > 0
    assert typed.sum() == len(target) == len(KEPT_SERIALS)  # 41 kept, MSE SE untyped
    np.testing.assert_array_equal(pocket.xyz[typed], target[["X", "Y", "Z"]].to_numpy(np.float32))
    np.testing.assert_array_equal(pocket.species[typed], target["ATOM_TYPE"].map(type_index))


def test_residue_serials_unique_with_insertion_code(species):
    pocket = read_pocket(POCKET, LIG, species)
    # ALA A 1, SER A 2, GLY A 52, ASN A 52A, MSE A 53, VAL B 1, LYS B 2.
    np.testing.assert_array_equal(pocket.res,
                                  [0] * 5 + [1] * 7 + [2] * 4 + [3] * 8 + [4] + [5] * 7 + [6] * 8)
    np.testing.assert_array_equal(pocket.chain, [0] * 25 + [1] * 15)


def test_radius_crop(species):
    far, near = np.float32([-11.0, 0.0, 0.0]), np.float32([-9.5, 0.0, 0.0])
    assert np.linalg.norm(LIG - far, axis=1).min() == pytest.approx(11.0)
    assert np.linalg.norm(LIG - near, axis=1).min() == pytest.approx(9.5)

    cropped = read_pocket(POCKET, LIG, species)
    full = read_pocket(POCKET, LIG, species, radius=12.0)
    assert len(cropped.xyz) == 40 and len(full.xyz) == 41
    assert not (cropped.xyz == far).all(axis=1).any()
    assert (cropped.xyz == near).all(axis=1).any()
    assert (full.xyz == far).all(axis=1).any()
    assert full.bfactor[-1] == 58.0  # LYS NZ, serial 48, last in serial order


def test_species_lookup_matches_key_table(species):
    assert species["ALA-CA"] == 2
    pocket = read_pocket(POCKET, LIG, species)
    assert pocket.species[KEPT_KEYS.index("ALA-CA")] == 2


def test_residue_and_atom_codes(species):
    assert len(STANDARD_RESIDUES) == 20 and list(STANDARD_RESIDUES) == sorted(STANDARD_RESIDUES)
    names = atom_name_table(species)
    assert len(names) == 37 and list(names) == sorted(names) and names[:3] == ("C", "CA", "CB")

    pocket = read_pocket(POCKET, LIG, species)
    # ALA 0, SER 15, GLY 7, ASN 2, MSE other (20), VAL 19, LYS 11.
    np.testing.assert_array_equal(pocket.resname,
                                  [0] * 5 + [15] * 7 + [7] * 4 + [2] * 8 + [20] + [19] * 7 + [11] * 8)
    expected_atom = [names.index(key.split("-")[1]) if key != "MSE-SE" else -1 for key in KEPT_KEYS]
    np.testing.assert_array_equal(pocket.atom, expected_atom)
    assert pocket.atom[0] == names.index("N") == 17


def test_typed_within(species):
    pocket = read_pocket(POCKET, LIG, species)
    mask = typed_within(pocket, LIG)
    assert mask.shape == (len(KEPT_SERIALS),) and mask.dtype == bool
    np.testing.assert_array_equal(np.flatnonzero(mask),
                                  [KEPT_SERIALS.index(s) for s in TYPED_WITHIN_SERIALS])
    np.testing.assert_array_equal(typed_within(pocket, LIG, cutoff=10.0), pocket.species > 0)


@pytest.mark.parametrize(
    ("name", "kept"),
    [("H", False), ("HB2", False), ("HD21", False), ("1HG1", False),
     ("CA", True), ("OXT", True), ("C1X2", True),
     ("1HA", True)],  # 3 characters, H not first: LoadPDBasDF keeps it (untyped here)
)
def test_hydrogen_rule_on_name_field(tmp_path, species, name, kept):
    field = name if len(name) == 4 else f" {name:<3}"
    line = f"ATOM      1 {field} GLY A   1       0.000   0.000   0.000  1.00 10.00           X"
    assert line[12:16].replace(" ", "") == name and line[30:38] == "   0.000"
    pocket = read_pocket(_write_pdb(tmp_path, [line]), np.zeros((1, 3)), species)
    assert len(pocket.xyz) == int(kept)
    if kept:
        assert pocket.species[0] == species.get(f"GLY-{name}", 0)


def test_rows_sorted_by_serial_and_bfactor_fallback(tmp_path, species):
    path = _write_pdb(tmp_path, [
        "ATOM      5  CA  ALA A   2       1.000   0.000   0.000  1.00******           C",
        "ATOM      3  CA  ALA A   1       2.000   0.000   0.000  1.00 12.50           C",
        "ATOM      4  CB  ALA A   1       3.000   0.000   0.000  1.00",
    ])
    pocket = read_pocket(path, np.zeros((1, 3)), species)
    np.testing.assert_array_equal(pocket.xyz[:, 0], [2.0, 3.0, 1.0])
    np.testing.assert_array_equal(pocket.bfactor, [12.5, 0.0, 0.0])
    np.testing.assert_array_equal(pocket.res, [0, 0, 1])


def test_empty_pocket(tmp_path, species):
    far_ligand = np.full((2, 3), 100.0)
    pocket = read_pocket(POCKET, far_ligand, species)
    _assert_layout(pocket, 0)
    assert typed_within(pocket, far_ligand).shape == (0,)

    path = _write_pdb(tmp_path, [
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 10.00           O",
        "ATOM      2  HA  ALA A   2       1.000   0.000   0.000  1.00 10.00           H",
        "ATOM      3 1HB2 ALA A   2       1.500   0.000   0.000  1.00 10.00           H",
        "TER       4",
        "END",
    ])
    _assert_layout(read_pocket(path, np.zeros((1, 3)), species), 0)


def _within_loop(points, centres, cutoff):
    """The original per-centre loop (Task 3), kept as the reference for the vectorised version."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nearest = np.full(len(points), np.inf)
    for centre in np.asarray(centres, dtype=np.float64).reshape(-1, 3):
        np.minimum(nearest, ((points - centre) ** 2).sum(axis=1), out=nearest)
    return nearest <= cutoff**2


def _ulp_ring(cutoff):
    """Points on the axes at exactly ``cutoff`` and one ulp either side, around the origin."""
    ring = []
    for axis in range(3):
        for value in (cutoff, np.nextafter(cutoff, 0.0), np.nextafter(cutoff, 2 * cutoff),
                      -cutoff, np.nextafter(-cutoff, 0.0)):
            point = np.zeros(3)
            point[axis] = value
            ring.append(point)
    return np.array(ring), np.zeros((1, 3))


def _boundary_shell(rng, cutoff):
    """600 points at distance ``cutoff`` (up to rounding) from one of four ligand atoms near
    (40, 40, 40): the loop's float64 arithmetic puts some just inside and some just outside, and the
    Gram form alone would decide about 1 in 40 of them differently."""
    centres = rng.uniform(-3, 3, (4, 3)) + 40.0
    directions = rng.normal(size=(600, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return centres[rng.integers(0, 4, 600)] + cutoff * directions, centres


def _within_cases():
    rng = np.random.default_rng(3)
    yield "boundary shell 5.1", *_boundary_shell(rng, 5.1), 5.1
    yield "boundary shell 10", *_boundary_shell(rng, 10.0), 10.0
    yield "pocket", rng.uniform(-10, 18, (1500, 3)).astype(np.float32), \
        rng.uniform(0, 8, (30, 3)).astype(np.float32), 5.1
    yield "radius 10", rng.uniform(-30, 40, (5000, 3)), rng.uniform(0, 8, (12, 3)), 10.0
    yield "far coordinates", (rng.uniform(-10, 18, (800, 3)) + 950).astype(np.float32), \
        (rng.uniform(0, 8, (20, 3)) + 950).astype(np.float32), 5.1
    yield "large ligand", rng.uniform(-15, 35, (3000, 3)), rng.uniform(0, 20, (224, 3)), 5.1
    yield "ulp ring 5.1", *_ulp_ring(5.1), 5.1
    yield "ulp ring 10", *_ulp_ring(10.0), 10.0
    yield "no centres", rng.uniform(0, 5, (7, 3)), np.zeros((0, 3)), 5.1
    yield "no points", np.zeros((0, 3)), rng.uniform(0, 5, (4, 3)), 5.1


@pytest.mark.parametrize(("name", "points", "centres", "cutoff"), list(_within_cases()),
                         ids=[case[0] for case in _within_cases()])
def test_within_matches_loop_reference(name, points, centres, cutoff):
    got = _within(points, centres, cutoff)
    assert got.dtype == bool and got.shape == (len(points),)
    np.testing.assert_array_equal(got, _within_loop(points, centres, cutoff))


def test_typed_pairs_are_the_typed_close_pairs_in_order(rng):
    lig = rng.uniform(0, 8, (25, 3)).astype(np.float32)
    xyz = rng.uniform(-10, 18, (900, 3)).astype(np.float32)
    species_codes = rng.integers(0, 23, 900).astype(np.int8)
    zeros = np.zeros(900)
    pocket = PocketArrays(xyz=xyz, species=species_codes, res=zeros.astype(np.int32),
                          resname=zeros.astype(np.int8), atom=zeros.astype(np.int16),
                          chain=zeros.astype(np.int8), bfactor=zeros.astype(np.float16))
    lig_index, pocket_index = typed_pairs(pocket, lig)
    d2 = ((lig.astype(np.float64)[:, None] - xyz.astype(np.float64)[None]) ** 2).sum(axis=2)
    want_lig, want_pocket = np.nonzero((d2 <= 5.1**2) & (species_codes[None] > 0))
    np.testing.assert_array_equal(lig_index, want_lig)  # sorted by ligand atom, then pocket atom
    np.testing.assert_array_equal(pocket_index, want_pocket)
    mask = np.zeros(900, dtype=bool)
    mask[pocket_index] = True
    np.testing.assert_array_equal(typed_within(pocket, lig), mask)
    assert [len(a) for a in typed_pairs(pocket, np.zeros((0, 3)))] == [0, 0]
