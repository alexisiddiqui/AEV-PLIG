# Pocket 3DZD: target identification, held-out groups, negatives

**Setup.** 18,828 PDBbind units with a UniProt (345 of the 19,173 have none), 3,863 UniProts.
- **Queries:** 16,976 units whose UniProt has ≥ 2 units; the database is every other unit.
- **Chance:** R@1 0.0027, R@5 0.013.
- **Columns:** diff-lig drops the query's `lig_key` from the database; macro is the mean over UniProts.
- **Standardisation:** fit on all 96,947 rows.

## 1. Same-UniProt nearest-neighbour recall

Each row shows the better metric. The full grid is in `nn_recall*.csv`.

| Variant | Metric | R@1 | R@5 | R@1 diff-lig | R@1 macro |
|---|---|---|---|---|---|
| (a) F raw | cos | 0.727 | 0.819 | 0.710 | 0.560 |
| (a) F standardised | euc | 0.732 | 0.827 | 0.715 | 0.566 |
| (b) F channel-L2 | either | 0.677 | 0.773 | 0.660 | 0.503 |
| **(c) F+G std, PCA-50 (current)** | cos | **0.689** | 0.796 | 0.672 | 0.519 |
| (d) F slog, standardised | euc | 0.729 | 0.827 | 0.713 | 0.560 |
| (e) occupancy only | cos | 0.631 | 0.739 | 0.608 | 0.461 |
| (f) chemistry channels | euc | 0.721 | 0.820 | 0.704 | 0.554 |
| (g) F / channel weight | euc | 0.670 | 0.772 | 0.654 | 0.499 |
| **F+G std, no PCA** | cos | **0.746** | 0.839 | 0.732 | 0.588 |
| F std + signed-sqrt G std | cos | 0.756 | 0.845 | 0.743 | 0.599 |
| (h) ligand F std (control) | cos | 0.456 | 0.590 | 0.421 | 0.320 |

- **The descriptor identifies the target.** R@1 is about 270× chance; replicate ligands add only 0.014.
- **PCA-50 is the loss.** It keeps 44% of the variance and costs 0.057 R@1 (paired UniProt-bootstrap CI −0.064 to −0.051).
- **Compressing G is a small extra.** Signed-sqrt G adds +0.010 (CI +0.007 to +0.013).
- **The ligand alone gets about 0.46.** That is the shortcut size.
- **The pocket signal is not a ligand proxy.** Regressing ligand F out of the pocket vector leaves R@1 at 0.746; size features alone give 0.037.

## 2. Families (F+G std, cosine, k = 10)

Same-UniProt share of each unit's 10 neighbours, mean / median:

| Target (units) | Share | Target (units) | Share |
|---|---|---|---|
| CA2 (410) | 0.94 / 1.0 | HIV-1 PR P03366 (111) | 0.42 / 0.4 |
| BACE1 (342) | 0.96 / 1.0 | Glycogen phosphorylase (100) | 0.81 / 1.0 |
| CDK2 (244) | 0.84 / 1.0 | Proteasome α2 (98) | 0.89 / 1.0 |
| BRD4 (197) | 0.75 / 0.9 | Aurora A (95) | 0.57 / 0.6 |
| Trypsin (194) | 0.92 / 1.0 | sEH (94) | 0.82 / 1.0 |
| Thrombin (173) | 0.84 / 1.0 | FXa (93) | 0.97 / 1.0 |
| HSP90α (159) | 0.82 / 0.9 | PTP1B (89) | 0.74 / 1.0 |
| p38α (147) | 0.66 / 0.7 | uPA (88; 24% of neighbours are trypsin) | 0.56 / 0.6 |
| TNKS2 (115) | 0.86 / 1.0 | PIM1 (88) | 0.87 / 1.0 |
| HIV-1 PR P03367 (83) | 0.25 / 0.2 | CHK1 (82) | 0.61 / 0.7 |

- **Cross-UniProt mutual pairs are close relatives.** 33% of the 43,685 mutual 10-NN pairs cross UniProts, and the top ones are orthologs, viral isolates or paralogs:
  - PDE4B–4D (105 pairs)
  - HIV-1 PR isolates (80 and 53)
  - AChE Torpedo–mouse (77)
  - PKA bovine–human (53)
  - BRD4–BRD2 (48)
  - PDE10A, NS5B genotypes, p38α species, TNKS1–2, JAK1/2/3, GluA2–3, MMP3/12/13, PI3Kα–γ, caspase-3–7, MDM2–MDMX
- **12% of these pairs share a `lig_key`.**
- **19% of rank-1 misses are the same protein name** under another accession.
- **Kinases do not pool at k = 10.** Trypsin-like proteases partly do.

## 3. Held-out groups

| Descriptor clustering | Cut | Clusters | Largest | Co-clustering | Purity | Units in split UniProts | NN in another cluster |
|---|---|---|---|---|---|---|---|
| Current (PCA-50 cos, average) | 0.547 | 849 | 7.2% | 0.42 | 0.34 | 76% | 12% |
| F+G std cos, average | 0.789 | 1,319 | 8.6% | 0.44 | 0.42 | 72% | 11% |
| F+G std cos, complete | 1.139 | 96 | 4.1% | 0.38 | 0.18 | 75% | 14% |
| F+G std cos, average, fine | 0.398 | 9,414 | 1.1% | 0.09 | 0.87 | 84% | 37% |

- **Co-clustering ≥ 0.9 needs one cluster** under every linkage.
- **Purity reaches 0.9 only at near-singleton cuts.** Even at the fine cut, 84% of units sit in a split UniProt.
- **Why:** the tail of same-target nearest-neighbour distances (q90 0.65) reaches past the other-target median (0.47).

| Target-first groups | n | Largest | NN in another group |
|---|---|---|---|
| UniProt | 3,863 | 2.2% | 32.7% |
| UniProt ∪ same PDBbind name | 2,996 | 3.3% | 25.6% |
| … + group-average cosine merge at 0.817 | 780 | 9.9% | 15.9% |

The name merge over-merges generic names ("INTEGRASE" covers 62 UniProts). The largest merged group (1,860 units, 638 UniProts) is a catch-all of shallow sites, not a family.

## 4. Pocket-definition evidence

| Ligand heavy atoms | < 40 | 40–50 | 50–70 | ≥ 70 |
|---|---|---|---|---|
| R@1 | 0.75–0.79 | 0.69 | 0.65 | 0.54 |

| Pocket atoms beyond 12 Å | < 20% | 30–50% | > 50% |
|---|---|---|---|
| R@1 | 0.77 | 0.64 | 0.55 |

Within a UniProt, pocket distance tracks the ligand:

| Pairs | Spearman with ligand-F distance | Spearman with heavy-atom difference |
|---|---|---|
| Same UniProt | 0.61 | 0.54 |
| Different UniProt | 0.12 | 0.14 |

## 5. Recommendations

**(i) Do not use descriptor clusters for the held-out split.**
- Split by target: UniProt merged across orthologs and isolates. Use sequence clustering; the name merge is a stopgap.
- Hold out whole groups and report each held-out unit's cosine distance to its nearest training pocket; 25.6% currently have their nearest pocket in another group.
- Optionally, merge groups at cosine 0.82 (780 groups, largest 9.9%), or drop training units within 0.26 of any held-out pocket (0.26 is q10 of the nearest other-group distance).

**(ii) Use cosine on standardised F+G with no PCA** (signed-sqrt G optional).

| Nearest-neighbour distance | Median | q10 | q90 |
|---|---|---|---|
| Same group | 0.27 | 0.07 | 0.63 |
| Other group | 0.52 | 0.26 | 0.70 |

- For 19.8% of queries an other-group pocket is nearer than any same-group one.
- Define "other target" by group: 14% of nearest other-UniProt pockets have the same protein name.
- Exclude negatives that share the `lig_key`.
- These negatives come ligand-size-matched: heavy-atom difference median 4 vs 13 for random pairs.

**(iii) Change the mask; keep the ligand-centroid centre.**
- **Problem:** the 8 Å shell follows the ligand and is cropped by R (median 14% of pocket atoms beyond R, q95 34%).
- **Change:** use a fixed sphere around the ligand centroid with radius below R (e.g. 10 Å with R = 12 Å). It removes ligand shape and size from the mask and crops nothing.
- **Pocket-centroid centring** fixes neither effect.
- **Unmeasured:** fields were not recomputed here. Run an A/B on about 2k units first.

**(iv) Issues found.**
1. **Failed rows hold zeros, not NaN.** The three failures (2r1w and two BindingDB rows, 2IJ7_TPF_bak_… and 2BUC_008_…) store pocket F/G = 0 and `n_pocket_atoms` = 0, not NaN / −1 as documented, because `process_group` (`featurise_zernike.py`) calls `res.update` before raising on `n_pocket == 0`. `ok` protects the clustering, but a filter on finiteness alone would not.
2. **F drops the sign of the hydrophobicity total.** Hydrophobicity F_00 = \|Σ logP\| (94% of pockets have a negative total, 6% positive). Only G(occ, hyd) keeps the sign.
3. **G and F scales differ.** G column SDs run 0.005–16 against 0.05–2.4 for F, so never use raw F+G. Variant (g) is unstable: the hydrophobicity total can be near 0, and the pos/neg channels are empty in 3–5k pockets.
4. **Identical pocket F across rows.** 1,592 rows in 679 groups:
   - 1,576 are BindingNet within one template; 445 groups mix different `lig_key`s, with a median pK spread of 0.30.
   - 6 are PDBbind pairs, two across splits: 1o3f (test) / 1o3g (train), and 1npa (train) / 1npw (valid).

   Deduplicate before mining negatives.
5. **Ligands are cropped too.** 2,489 ligands extend beyond 12 Å.
6. **Otherwise clean.** No NaNs, no constant columns, and F_00 matches the atom counts to 1e-5.
