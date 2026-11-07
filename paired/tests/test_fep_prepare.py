"""Tests for src/fep_prepare.py on a tiny synthetic copy of the benchmark layout."""
import io
import os

import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402

import fep_prepare  # noqa: E402

PROTEIN = (
    "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1      13.164   6.188  -5.146  1.00  0.00           C\n"
    "ATOM      4  C   ACE A   0      10.500   6.000  -7.000  1.00  0.00           C\n"
    "HETATM    5  O   HOH A 101      14.000   7.000  -4.000  1.00  0.00           O\n"
    "END\n"
)


def _mol(smiles, title, dg, tag, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol.SetProp("_Name", title)
    mol.SetProp("r_exp_dg", str(dg))
    mol.SetProp("s_tag", tag)
    return mol


def _append(sdf_path, mol):
    """Append one SD record (with properties) to an existing SDF."""
    buf = io.StringIO()
    writer = Chem.SDWriter(buf)
    writer.write(mol)
    writer.close()
    with open(sdf_path, "a") as fh:
        fh.write(buf.getvalue())


def _make_benchmark(root, series="my_series_x", entries=None):
    """Write <root>/fep_benchmark_inputs/{benchmark_metadata.csv, structure_inputs/subA/...}."""
    if entries is None:
        entries = [("a", -7.04, "first", "CCO"), ("b", -8.0, "only", "CCN"), ("a", -9.0, "second", "CCC")]
    sub_dir = os.path.join(root, "fep_benchmark_inputs", "structure_inputs", "subA")
    os.makedirs(sub_dir)
    writer = Chem.SDWriter(os.path.join(sub_dir, f"{series}_ligands.sdf"))
    for k, (title, dg, tag, smi) in enumerate(entries):
        writer.write(_mol(smi, title, dg, tag, seed=k + 1))
    writer.close()
    with open(os.path.join(sub_dir, f"{series}_protein.pdb"), "w") as fh:
        fh.write(PROTEIN)
    with open(os.path.join(sub_dir, f"{series}_edges.csv"), "w") as fh:
        fh.write("Ligand1,Ligand2\na,b\n")
    header = ("Group name,Group abbreviation,Protein,Input file naming scheme,Reference PDB,"
              "Number of nodes,Dynamic range (kcal/mol),DOI of source\n")
    line = f"Sub A,subA,Prot,{series},1ABC,2,1.0,doi:x\n"
    with open(os.path.join(sub_dir, "subset_metadata.csv"), "w") as fh:
        fh.write(header + line)
    with open(os.path.join(root, "fep_benchmark_inputs", "benchmark_metadata.csv"), "w") as fh:
        fh.write(header + line)
    return root


def test_pk_conversion():
    assert fep_prepare.dg_to_pk(-7.04) == pytest.approx(5.160, abs=1e-3)


def test_fep_prepare_dedupes_titles(tmp_path):
    bench = _make_benchmark(str(tmp_path / "bench"))
    out = str(tmp_path / "out")
    df = fep_prepare.prepare_benchmark(bench, out)
    assert len(df) == 2
    assert df["n_entries"].tolist() == [2, 1]
    row_a = df.iloc[0]
    assert row_a["unique_id"].endswith("/a")
    assert row_a["pK"] == pytest.approx(fep_prepare.dg_to_pk(-7.04))
    # the per-ligand file holds exactly the first "a" entry and is readable as process_data reads it
    suppl = Chem.SDMolSupplier(row_a["sdf_file"], removeHs=False)
    assert len(suppl) == 1
    mol = suppl[0]
    assert mol is not None
    assert mol.GetProp("s_tag") == "first"
    assert mol.GetProp("_Name") == "a"
    # the csv on disk matches the returned frame
    on_disk = fep_prepare.pd.read_csv(os.path.join(out, "fep_benchmark.csv"))
    assert on_disk["unique_id"].tolist() == df["unique_id"].tolist()
    assert list(on_disk.columns) == fep_prepare.COLUMNS


def test_unique_id_format(tmp_path):
    bench = _make_benchmark(str(tmp_path / "bench"), series="my_series_x")
    out = str(tmp_path / "out")
    df = fep_prepare.prepare_benchmark(bench, out)
    assert df["unique_id"].tolist() == ["subA/my_series_x/a", "subA/my_series_x/b"]
    assert set(df["series"]) == {"my_series_x"}
    assert set(df["subset"]) == {"subA"}
    assert set(df["ref_pdb"]) == {"1ABC"}
    for path in list(df["sdf_file"]) + list(df["pdb_file"]):
        assert os.path.isabs(path) and os.path.isfile(path)
        assert os.path.dirname(path) == os.path.join(out, "structures")
    # the protein is copied unchanged
    with open(df["pdb_file"].iloc[0]) as fh:
        assert fh.read() == PROTEIN


def test_titles_with_unsafe_characters_get_safe_unique_files(tmp_path):
    entries = [("set 1: 1", -7.0, "x", "CCO"), ("set 1/ 1", -8.0, "y", "CCN")]
    bench = _make_benchmark(str(tmp_path / "bench"), series="s", entries=entries)
    df = fep_prepare.prepare_benchmark(bench, str(tmp_path / "out"))
    assert df["unique_id"].tolist() == ["subA/s/set 1: 1", "subA/s/set 1/ 1"]
    names = [os.path.basename(p) for p in df["sdf_file"]]
    assert len(set(names)) == 2
    assert all(" " not in n and "/" not in n and ":" not in n for n in names)


def test_missing_exp_dg_is_skipped(tmp_path):
    bench = _make_benchmark(str(tmp_path / "bench"), series="s")
    sdf = os.path.join(bench, "fep_benchmark_inputs", "structure_inputs", "subA", "s_ligands.sdf")
    mol = _mol("CCCl", "c", 0.0, "z", seed=9)
    mol.ClearProp("r_exp_dg")
    _append(sdf, mol)
    df = fep_prepare.prepare_benchmark(bench, str(tmp_path / "out"))
    assert "subA/s/c" not in set(df["unique_id"])
    assert len(df) == 2


def test_residue_census_counts_dropped_atoms(tmp_path):
    pdb = tmp_path / "p.pdb"
    pdb.write_text(PROTEIN)
    allowed = {"ALA-N", "ALA-CA", "ALA-C"}
    census = fep_prepare.residue_census(str(pdb), allowed)
    by_res = {(r["record"], r["residue_name"]): r for r in census}
    assert set(by_res) == {("ATOM", "ACE"), ("HETATM", "HOH")}
    assert by_res[("ATOM", "ACE")]["n_dropped_atoms"] == 1
    assert by_res[("ATOM", "ACE")]["dropped_atom_names"] == "C"
    assert by_res[("HETATM", "HOH")]["n_residues"] == 1


def test_fepplus_exp_dg_fallback(tmp_path):
    bench = _make_benchmark(str(tmp_path / "bench"), series="s")
    sdf = os.path.join(bench, "fep_benchmark_inputs", "structure_inputs", "subA", "s_ligands.sdf")
    mol = _mol("CCCl", "c", 0.0, "z", seed=9)
    mol.ClearProp("r_exp_dg")
    mol.SetProp("r_fepplus_exp_dg", "-9.88")
    _append(sdf, mol)
    df = fep_prepare.prepare_benchmark(bench, str(tmp_path / "out"))
    row = df[df["unique_id"] == "subA/s/c"]
    assert len(row) == 1
    assert row["pK"].iloc[0] == pytest.approx(fep_prepare.dg_to_pk(-9.88))


def test_scored_set_filter(tmp_path):
    bench = _make_benchmark(str(tmp_path / "bench"), series="s")
    res = os.path.join(bench, "21_4_results")
    os.makedirs(os.path.join(res, "ligand_predictions", "subA"))
    with open(os.path.join(res, "benchmark_output_metadata.csv"), "w") as fh:
        fh.write("Group name,Group abbreviation,Protein,Reference PDB,Input file naming scheme,"
                 "Output file naming scheme\nSub A,subA,Prot,1ABC,s,s_bmcorr_out\n")
    with open(os.path.join(res, "ligand_predictions", "subA", "s_bmcorr_out.csv"), "w") as fh:
        fh.write("Ligand,Exp. dG (kcal/mol),Pred. dG (kcal/mol),Pred. dG std. error (kcal/mol)\n"
                 "a,-7.04,-7.0,0.1\nzz,-6.0,-6.1,0.1\n")
    df = fep_prepare.prepare_benchmark(bench, str(tmp_path / "out"))
    assert df["unique_id"].tolist() == ["subA/s/a"]
    assert df["n_entries"].tolist() == [2]
    df_all = fep_prepare.prepare_benchmark(bench, str(tmp_path / "out_all"), scored_only=False)
    assert df_all["unique_id"].tolist() == ["subA/s/a", "subA/s/b"]
