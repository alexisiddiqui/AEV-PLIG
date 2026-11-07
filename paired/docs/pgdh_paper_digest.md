# Rich, Dodda, Price 2026 (ChemMedChem, 10.1002/cmdc.70531): digest for the AEV-PLIG arms

## 1. The 15-PGDH data
- **Source and series** (Sec 2, Tables 1-2): 43 compounds from the Nimbus campaign in Dodda et al., J Med Chem 2025, 68:18436.
  - **S1** (n=11): deconstructs the 6,6 benzomorpholine core.
  - **S2** (n=7): 6,5 scaffold hops (e.g. 7-azaindole) and a benzodihydropyran.
  - **S3** (n=13): Suzuki replacements of the bromine on published Compound 4. This is the **novel-subpocket series**, built to displace 2-3 high-energy waters; 11/12 picks equipotent to 500x more potent.
  - **S4** (n=12): a fused cycloalkyl-triazole amide isostere.
- **Potency:** the text uses pIC50. S4 spans 8.0-10.0, above its ChEMBL exemplars at 5.4-7.7 (Sec 4.4). The repo's "pKi" is the experimental ΔG from the FEP+ map divided by 1.364. My ranges from it: S1 7.59-10.50, S2 6.13-10.08, S3 5.80-10.25, S4 7.51-10.09. **The paper never states the assay type.**
- **Structures:** the receptor is X-ray **9PFL** (compound 1, 1.65 Å, Maestro-prepared). No other test ligand has an experimental complex.
  - Vina and AEV-PLIG scored **Glide MCS-constrained docked FEP starting poses, Vina-minimised**.
  - Boltz-2 cofolded each compound with NAD+.
  - The S3 RMSD reference is FEP+ terminal-window snapshots.
- **Release:** github.com/inductive-bio-misc-collab/ml-affinity-15-PGDH-case-study (no licence file).
  - Contents: `data/poses.sdf` (docked poses, SMILES, series, pKi, FEP_pKi), `data/reference.pdb` (9PFL), MSA, `scoring/vina_predictions.sdf` (the poses AEV-PLIG scored), all method predictions, 43 Boltz `model_0` PDBs, scripts.
  - **Missing:** the finetuned weights (README: "TODO"), the 40 added training complexes, the S3 FEP snapshots (`boltz_rmsd.py` uses the docked poses for all series), and the second Boltz seed.
  - Their CSVs reproduce Table 2 to 2 dp.

## 2. How AEV-PLIG was run (Sec 3.1.2)
- **Ensemble:** 10-fold, averaged.
- **Default:** the released `model_GATv2Net_ligsim90_fep_benchmark` weights (PDBbind + BindingNet + BindingDB-DCS; no 15-PGDH).
- **RT (retrained):** the published training script, hyperparameters unchanged, plus **40 ligands from US Patent 11,345,702** (Rai et al.; text says "Ria") "docked into the same receptor".
  - The docking method for these 40 is not given.
  - Max Morgan-2 Tanimoto to the test series: 0.369 (S1/S2), 0.625 (S3), 0.323 (S4).
  - RT fits these 40 at MAE 0.06, ρ 0.99.
- **Positive control:** both ensembles reproduce the 8-target JACS FEP+ mean per-target Spearman, within 0.01 (default) and 0.03 (RT).
- **Metrics:** **Spearman only**, with a paired bootstrap and BH-FDR against FEP+. No Pearson, Kendall or RMSE is reported.

Table 2 (Spearman ρ):

| Method | All (43) | S1&2 (18) | S3 (13) | S4 (12) |
|---|---|---|---|---|
| FEP+ | 0.70 | 0.64 | 0.78 | 0.66 |
| Boltz-2 | 0.31* | 0.68 | 0.17 [-0.52,0.74] | 0.55 |
| Vina | 0.45 | 0.55 | 0.57 | 0.67 |
| AEV-PLIG | 0.09 [-0.22,0.38] | 0.14 | 0.25 | -0.14 |
| AEV-PLIG RT | 0.06 | 0.11 | 0.43 | -0.44 [-0.82,0.18] |
| HAC | 0.09 | 0.55 | -0.44 | -0.15 |

\*padj<0.05 vs FEP+; this is the only significant difference.

Heavy-atom count (HAC) is the only null; there is **no 2D-similarity or ligand-only baseline**.

## 3. Diagnosis
- **AEV-PLIG** (Sec 5):
  - Losing to Vina on the same poses is "consistent with overfitting … rather than learning the underlying physics". Caveat: the poses are Vina-minimised.
  - The authors say this "cannot distinguish overfitting from … domain shift or pose sensitivity" and "We lack a fully satisfying explanation".
  - The RT gain is local: S3 is the series nearest the added ligands, and RT still ranks the parent top.
  - Suggested causes: 40 complexes swamped, hyperparameters need retuning, other training data counterproductive, or hidden benchmark bias inflated the reported OOD performance.
- **Boltz-2** (Sec 4.1, 4.3, Table 1):
  - Median ligand RMSD is 0.55, 0.56 and 0.81 Å for S1, S2 and S4. S3 is worse: mean 1.61, median 1.19 Å.
  - Between seeds the median RMSD is 0.29 Å, but two S3 compounds land >4 Å off with a flipped solvent-exposed group.
  - **No S3 analogue was predicted more potent than Compound 4**, the only member likely in training, despite gains up to 500x. The top compound ranked mid-table despite a 0.5 Å RMSD, so ranking does not track pose quality.
  - Predicted ranges are compressed toward the training distribution: S1/S2 predictions sit at 7.3-8.0, close to the ChEMBL range of 6.8-7.9.
  - Pooled ρ 0.31: a Simpson-like series offset.

## 4. Bearing on our arms
- **Pose sensitivity:** Boltz affinity is pose-invariant. The two seeds differ by ≤0.12 log units, and by ≤0.05 where the pose changed. AEV-PLIG was never pose-perturbed.
- **Decoys:** none.
- **Augmentation with modelled complexes:** background only (~20k to ~100k complexes in AEV-PLIG). There is no ablation.
- **Pairwise/ΔΔG:** none.
- **Activity cliffs:** not discussed; only that "adding four heavy atoms" can leave the domain of applicability.
- **Extrapolation:** the central theme. Predicted ranges are compressed, and S4 potencies lie above the training range ("must capture additive SAR").
- **Benchmark bias** (Sec 2): HAC correlates with potency in 4 of the 8 Wang FEP+ targets (ρ 0.66-0.77), and ligand-only models match structure-based ones on some benchmarks. Independent support for our arm-1 ligand-only result.

## 5. Assessment
1. (a) Add this set as an external sanity test. It is feasible: `vina_predictions.sdf` plus 9PFL is exactly AEV-PLIG's input, and their default CSV gives an exact reproduction target. First check our training series for HPGD (P15428).
2. (a) With n=12-18 per series the CIs are 0.6-1.3 wide, so the set cannot rank arms or resolve seed effects. Report it beside HAC and Vina; never use it for model selection.
3. (b) The paper names pose sensitivity but does not test it. Its evidence (losing to Vina on identical poses, memorising the added complexes without transfer) points to poor generalisation, not poor pose discrimination.
4. (b) A direct probe with the released files: score the Boltz poses against the docked poses and check whether arm-4 prediction shifts track RMSD.
5. (b) If arm 4 really makes scores depend on the interactions, AEV-PLIG should move toward Vina on S3/S4. A null would match arm 1.
