# AEV-PLIG raw training data (downloaded 2026-11-04, Isambard login node, wget -c)
Archives are kept as single files and NOT extracted (inode budget).

| file | URL | bytes (verified) | md5 | licence | tar top level |
|---|---|---|---|---|---|
| BindingNetv1.tar.gz | https://zenodo.org/api/records/11069450/files/BindingNetv1.tar.gz/content | 312464986 | 0d72f2bd5fe7fec359d75041a85f1520 | CC-BY-4.0 (Zenodo 11069450) | ./BindingNetv1/ |
| Crystal_Templates_for_BindingNet1.tar.gz | https://zenodo.org/api/records/11069450/files/Crystal_Templates_for_BindingNet1.tar.gz/content | 2182806108 | 30f51c93523b55abf879812e2ad0ca60 | CC-BY-4.0 (Zenodo 11069450) | ./Crystal_Templates_for_BindingNet1/ |
| surflex.tar | https://www.bindingdb.org/rwd/data/surflex/surflex.tar | 9246187520 | b6ce45dce5612aebaaa36aba25504864 | BindingDB data terms (CC BY 4.0 per BindingDB site; not re-checked) | none: 1598 folders directly at tar root (e.g. 11GS_EAA/), no surflex/ prefix |

## Layout
- BindingNetv1.tar.gz: ./BindingNetv1/<pdb>/target_<CHEMBL>/<compnd>/<pdb>_<CHEMBL>_<compnd>.{sdf,pdb}; 159061 .sdf, 159084 .pdb, 0 rec_h_opt.pdb, no from_chembl_client dir. 491654 entries.
- Crystal_Templates_for_BindingNet1.tar.gz: ./Crystal_Templates_for_BindingNet1/<pdb>/{rec_h_opt.pdb,cry_lig_opt_converted.sdf}; 17179 rec_h_opt.pdb; 51538 entries.
- To get the loader layout: from_chembl_client/<pdb>/target_<T>/<compnd>/<pdb>_<T>_<compnd>.sdf = contents of BindingNetv1/ and from_chembl_client/<pdb>/rec_h_opt.pdb = Crystal_Templates_for_BindingNet1/<pdb>/rec_h_opt.pdb, i.e. merge the two trees per <pdb> dir under one from_chembl_client/ root (or symlink/point the loader at both roots).
- surflex.tar: 164172 entries, 90542 .mol2, 21865 .pdb, files at <folder>/<file> (loader expects surflex/<folder>/<file>, so extract into a dir called surflex/ or strip accordingly).

## Spot checks vs oxpig/AEV-PLIG data CSVs (main branch)
- bindingnet_processed.csv (69727 rows): all 69727 expected BindingNetv1/<pdb>/target_<target>/<compnd>/<pdb>_<target>_<compnd>.sdf paths exist in the tar; all 5903 distinct CSV pdb ids have a rec_h_opt.pdb in the crystal-template archive.
- bindingdb_processed.csv (8822 rows): all 8822 (folder/mol2_file) and (folder/pdb_file) paths exist in surflex.tar. Samples: 4DBU_BT9/4DBU-results_50337278.mol2 + 4DBU.pdb; 3Q32_J2I/3Q32-results_50337339.mol2 + 3Q32.pdb; 2JFH_LK1/2JFH-results_26445.mol2 + 2JFH.pdb; 2DC9_74M/2DC9-results_16500.mol2 + 2DC9.pdb.

## Added 2026-11-05 (paired-data work; public, no account needed)
- chembl_37_sqlite.tar.gz (5,764,252,857 B; sha256 33c203740555f96067710cdfc1c3c55d890660e5908ec5cbf5817492c290d281, matches EBI checksums.txt) from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ ; licence CC BY-SA 3.0. Purpose: recover assay id / activity type / relation / pChEMBL for every BindingNet (target, compound) row. Extract only on /local inside a job (the SQLite is ~25 GB, one file).
- schrodinger_public_binding_free_energy_benchmark-main.tar.gz (69,125,063 B; md5 5b67005428c9b6fce4c2f6ca4dcf252f) = github.com/schrodinger/public_binding_free_energy_benchmark @ main, fetched 2026-11-05. Ross et al. 2023 FEP benchmark: fep_benchmark_inputs/structure_inputs/<subset>/<series>_{protein.pdb,ligands.sdf,edges.csv} (92 series, ligands tagged with experimental dG in kcal/mol), benchmark_metadata.csv, 21_4_results/ (FEP+ predictions). This is the FEP set AEV-PLIG evaluates on (series/ligand ids match the paper CSVs).
- aev-plig-paper-main.tar.gz (11,637,648 B; md5 bd6152c498676f0ca7e578dbeefcf55b) = github.com/matthewtwarren/aev-plig-paper @ main: the paper's per-ligand predictions (10 seeds) for FEP benchmark at ligsim 60/70/80/90, Merck/Schrodinger subsets, CASF-2016, OOD Test, 0 Ligand Bias (data/), plus the figure notebook.
- OOD-Test-main.tar.gz (819,644 B; md5 afee8e703c50a78dbcdb82f7814336e2) = github.com/isakvals/OOD-Test @ main: index_oodtest.csv with the OOD Test train/test split.
