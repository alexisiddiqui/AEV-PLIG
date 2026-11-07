# 3D Zernike descriptors of pockets and ligands, and target groups

This is the descriptor layer for the soft-negative line (`docs/zernike_soft_negatives_design.md`,
Part 2 §1 descriptor, §3 pocket characterisation, §4 cost). It turns every complex of `paired_meta.csv`
into rotation-invariant 3D Zernike descriptors (3DZD) of a property-weighted pocket field and a
property-weighted ligand field. It then builds target groups for held-out splits (`pocket_groups.py`,
§8) and gives each row its cosine distance to the nearest training pocket of another group. Descriptor
clusters (`pocket_clusters.py`, §5) are kept for reference but not used for the split
(`docs/pocket_descriptor_analysis.md` §3, §5).

| File | Role |
|---|---|
| `src/zernike3d.py` | Moments and invariants of a weighted point cloud. Pure numpy/scipy. |
| `src/pocket_fields.py` | Channel assignment (ligand: RDKit; protein: residue tables), pocket mask, `pocket_descriptor`, `ligand_descriptor`. |
| `src/featurise_zernike.py` | Multiprocessing CLI over `paired_meta.csv`, or over the FEP benchmark with `--fep-dir`. Writes `zernike.npz` and `zernike_summary.json`. |
| `src/pocket_clusters.py` | Standardise, PCA to 50 dimensions, average-linkage cosine clustering with a same-UniProt cut. Writes `pocket_clusters.csv` and `pocket_clusters_summary.json`. Not used for the split. |
| `src/pocket_groups.py` | Target groups: UniProt merged across non-generic PDBbind names, optional group-average descriptor merge, nearest training pocket of another group. Writes `pocket_groups.csv` and `pocket_groups_summary.json`. |
| `jobs/featurise_zernike.sbatch` | Isambard job: extracts five archives to `/local`, featurises with 72 workers, then builds the target groups. |
| `tests/test_zernike3d.py`, `tests/test_pocket_fields.py`, `tests/test_pocket_clusters.py`, `tests/test_pocket_groups.py` | CPU tests. `test_pocket_fields.py` also runs `featurise_zernike` end to end on a synthetic parent layout (both pocket modes, failed rows, signed totals). |

## 1. Moments and invariants

The definitions follow Novotni & Klein 2003. A point i has a position r_i, a weight w_ic in channel c,
and is mapped into the unit ball as x_i = (r_i − centre) / R.

```
Z_nlm(r, θ, φ) = R_nl(r) Y_lm(θ, φ)          n ≤ N, (n − l) even, |m| ≤ l
R_nl(r)  = Σ_{v=0..k} q_kl^v r^(2v+l),     k = (n − l)/2
q_kl^v   = (−1)^k / 2^(2k) · sqrt((2l + 4k + 3)/3) · C(2k, k) · (−1)^v · C(k, v) · C(2(k+l+v)+1, 2k) / C(k+l+v, k)
Ω^c_nlm  = (3 / 4π) Σ_i w_ic conj(Z_nlm(x_i))     (points with |x_i| > 1 are dropped and counted)
F^c_nl   = sqrt(Σ_m |Ω^c_nlm|²)                     per-channel invariants
G^ab_nl  = Re Σ_m Ω^a_nlm conj(Ω^b_nlm),  a < b      cross-channel invariants (SOAP-style)
```

- Y_lm is `scipy.special.sph_harm`: orthonormal on the sphere, with the Condon-Shortley phase.
- Only m ≥ 0 is evaluated. Weights are real, so Ω_{n,l,−m} = (−1)^m conj(Ω_nlm) fills in the rest.
- The rotation-invariance argument: a rotation acts on each Ω_nl· through the unitary Wigner matrix D^l. That matrix is the same for every n and every channel, so both F and G are invariant.
- **Layout.** `moments()` returns `Omega[c, n, l, m + N]` with shape (C, N+1, N+1, 2N+1). Invalid entries are 0.
- **Invariant order.** `nl_index(N)` orders the invariants by n ascending, then l ascending. There are 49 at N = 12 (25 at N = 8, 72 at N = 15, 121 at N = 20).
- **Pair order.** `channel_pairs(7)` is `itertools.combinations` order, giving 21 pairs, so G has shape [21, 49].
- **Field.** The field is a sum of delta functions at the heavy atoms. There is no 1 Å Gaussian smoothing, unlike design §1. Convolving with a Gaussian would act as a radial low-pass filter on each (n, l), and at N = 12 and R = 12 Å the resolution is already about R/N ≈ 1 Å.
- **Scale convention.** With orthonormal Y_lm, ∫₀¹ R_nl² r² dr = 1/3. Novotni's C++ code and ZMPY3D normalise Y_lm to 4π on the sphere instead. Every Ω, and hence every F and G, here is therefore exactly 1/sqrt(4π) times theirs (`zernike3d.NK_CODE_SCALE`). The ZMPY3D test below confirms the factor. Standardisation downstream removes it.

**Fixed frame.** Both objects are centred on the ligand heavy-atom centroid, with a fixed radius per
object. `--pocket-mode` picks the pocket mask:

| Mode | Pocket atoms | Pocket R | Options |
|---|---|---|---|
| `shell` (default, the original run) | protein heavy atoms within `--cutoff` 8 Å of any ligand heavy atom | `--radius` 12 Å | `--pocket-radius` is an error |
| `sphere` | protein heavy atoms within `--pocket-radius` (default 10 Å) of the ligand centroid | = `--pocket-radius` | `--radius` and `--cutoff` are errors |

- **Ligand.** The ligand descriptor has its own radius, `--ligand-radius` (default 12 Å, the original value), in both modes. Ligand atoms beyond it are dropped and counted in `n_outside_ligand`.
- **Size is kept.** Because R is fixed, pockets are not rescaled, so size stays in the descriptor and must be matched downstream.
- **Atoms beyond R (shell mode).** Pocket atoms farther than 12 Å from the centroid are dropped and counted in `n_outside_pocket`. On the FEP set this is about 9% of pocket atoms at the median and 23% at q95; on the full table 14% at the median and 34% at q95.
- **Why the sphere.** The 8 Å shell follows the ligand, so a large or elongated ligand both enlarges the mask and pushes it past R, where it is cropped. Recall of the same UniProt falls with ligand size (R@1 0.75-0.79 below 40 heavy atoms, 0.54 at 70 or more) and with the cropped fraction (0.77 below 20%, 0.55 above 50%), and within a UniProt the pocket distance tracks the ligand-F distance (Spearman 0.61). A fixed sphere around the ligand centroid, with R equal to its radius, removes the ligand's shape and size from the mask and crops nothing (`n_outside_pocket` is 0 by construction: the mask uses the same inside-the-ball test as `zernike3d.moments`). The centroid centre is kept, because pocket-centroid centring fixes neither effect (`docs/pocket_descriptor_analysis.md` §4, §5 iii).
- **Status of the sphere.** It has not been featurised yet. Run it as an A/B variant (`OUT_NAME`, `VARIANT_NOTE`, §7) and compare same-UniProt recall on about 2k units before switching the default.

## 2. Channels (fixed order, C = 7)

`occupancy, hydrophobicity, pos, neg, aromatic, donor, acceptor`. All points are heavy atoms. Hydrogen
properties are folded onto the bonded heavy atom (united atom) on both sides, so the ligand and protein
scales match.

**Ligand** (RDKit molecule; missing Hs are added internally for the property calculations):

| Channel | Weight |
|---|---|
| occupancy | 1 |
| hydrophobicity | Wildman-Crippen atom logP. `_CalcCrippenContribs` on the explicit-H molecule, with each H's contribution added to its heavy atom, so the channel sums to `Crippen.MolLogP`. |
| pos / neg | max(q, 0) / max(−q, 0), where q is the united-atom Gasteiger charge: `ComputeGasteigerCharges` on the explicit-H molecule, H charges folded, NaN set to 0. The two channels sum to the formal charge. |
| aromatic | `atom.GetIsAromatic()` |
| donor / acceptor | Heavy atoms of the `Donor` / `Acceptor` features of RDKit `BaseFeatures.fdef`. A carboxylate O⁻ is not an acceptor there; only C=O is. |

Folding H onto heavy atoms is a decision. With raw heavy-atom Gasteiger charges, every CH carbon would
show up as a small negative charge (benzene would get 6 × −0.06 e).

**Protein** (standard-residue heavy atoms):

| Channel | Weight |
|---|---|
| occupancy | 1 |
| hydrophobicity | `CRIPPEN_TABLE` (below) |
| pos | LYS NZ 1; ARG NE, NH1, NH2 1/3 each; HIS ND1, NE2 0.1 each |
| neg | ASP OD1, OD2 0.5 each; GLU OE1, OE2 0.5 each; OXT (C-terminus) 1 |
| aromatic | Ring atoms of PHE and TYR (CG, CD1, CD2, CE1, CE2, CZ), TRP (CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2) and HIS (CG, ND1, CD2, CE1, NE2) |
| donor | Backbone N (not PRO); ARG NE, NH1, NH2; ASN ND2; GLN NE2; HIS ND1, NE2; LYS NZ; SER OG; THR OG1; TRP NE1; TYR OH; CYS SG |
| acceptor | Backbone O and OXT; ASP OD1, OD2; GLU OE1, OE2; ASN OD1; GLN OE1; HIS ND1, NE2; SER OG; THR OG1; TYR OH; MET SD |

`CRIPPEN_TABLE` is hard-coded from one RDKit 2024.03.3 run (`pocket_fields.crippen_table_from_rdkit`):

- **Template.** For each residue X, take the middle residue of `Chem.MolFromSequence("G"+X+"G")`. This gives neutral side chains and in-chain backbone typing.
- **H folding.** `AddHs`, then Crippen contributions with each H folded onto its heavy atom.
- **Symmetrisation.** In a charged residue some atom pairs are equivalent, but they differ in the neutral template. They are averaged: ASP OD1/OD2, GLU OE1/OE2, ARG NH1/NH2, and HIS ND1/NE2 (tautomer unknown). OXT is the mean of the C-terminal template's carboxyl O and OXT (−0.07195).
- **Test.** The test recomputes the table and requires agreement to 1e-9.

Backbone values are N −0.4954 (PRO −0.3187), CA −0.0821 (GLY 0.0425), C −0.2783 and O −0.1526. Side
chains range from SER OG −0.557 to MET SD 0.648.

**Protein loader** (`load_protein_atoms`). It copies the heavy-atom name rule of the clone's
`LoadPDBasDF`, and keeps an atom only when RES-NAME is one of the 187 keys of the clone's
`data/PDB_Atom_Keys.csv`. That key set is the clone's inner merge, and `STANDARD_ATOMS` equals it
(checked against the census copy).

`LoadPDBasDF` itself is not called, because its output drops residue names and numbers, which the
channel tables need. The deliberate differences from the clone's atom set are:

- only the first MODEL is read;
- for a repeated (chain, residue number, insertion code, atom name), the first record wins (altlocs);
- HID/HIE/HIP/HSD/HSE/HSP, CYX/CYM, ASH, GLH and LYN are read as their standard residue. Their charges stay those of the standard residue.

HETATM records, waters, caps (ACE/NMA/NME) and modified residues (TPO, PTR, ...) are not read, as in
the clone.

## 3. Validation (`tests/test_zernike3d.py`, local venv, Apple silicon)

| Check | Result |
|---|---|
| (a) Rotation invariance: 400 points, 3 channels, 20 random rotations about the centre | Max abs deviation of F and G 2.7e-12 (max \|F\| 6.3). The bound is 1e-8. |
| Direct definition: every (n, l, m), negative m included, against the term-by-term sum | < 1e-12 |
| (b) A single point at the origin | Non-zero only for l = 0. F_n0 = (3/4π)\|R_n0(0)\|/sqrt(4π). |
| (c) Translation | A shift of (2, −1, 3) Å changes F. Passing the shift as `center` restores F to 1e-10. |
| (d) ZMPY3D 0.0.2 (GPL, test only), order 8, sphere (radius 0.9) and cube (half-width 0.5) voxelised in [−1, 1]³ | **Exact quadrature** (16³ grid, 5-point Gauss-Legendre per axis inside each voxel; ZMPY3D integrates each voxel exactly): max relative difference of F × sqrt(4π) is 5.5e-12 (sphere) and 2.0e-13 (cube). **Voxel centres as points** (64³, midpoint rule): 1.9e-2 (sphere) and 5.5e-3 (cube); max abs / max F is 1.9e-3 and 1.5e-3. This is the O(h²) quadrature gap and shrinks to 8e-4 / 7e-4 at 96³. |
| (e) Timing: 1,500 points × 7 channels at order 12 (`moments` + F + G) | 16-26 ms best of 7 (bound 50 ms). About 12 ms of that is `sph_harm`. |

The ZMPY3D comparison uses ZMPY3D's own grid moments (`calculate_bbox_moment` and
`calculate_bbox_moment_2_zm` with its order-20 cache), with our own centre and scale. Its invariant is
sqrt(|Ω_l0|² + 2 Σ_{m>0} |Ω_lm|²), the same F. The tests skip when `ZMPY3D` is not importable, as on
Isambard.

## 4. Featurisation run on the FEP benchmark (local, 92 series)

Run: `featurise_zernike.py --fep-dir <benchmark> --out zernike_fep.npz`.

- **Rows.** There is one row per distinct SDF title (first entry kept): 1,343 rows over 92 series. The 1,184 scored titles of `fep_benchmark.csv` are a subset, selectable with `--fep-ids`.
- **Result.** All rows ok. Outputs are identical with 1 and 8 workers.

| Quantity (single worker) | Value |
|---|---|
| Per complex (ligand read and typing, pocket and ligand descriptors, plus its share of the protein parse) | median 11.8 ms, q95 17.5 ms. The first row is 0.22 s because it builds the RDKit feature factory. |
| Protein parse (FEP proteins, 4-9k heavy atoms) | median 9.4 ms per protein |
| One protein per complex (PDBbind-like), estimate | about 21 ms per complex |
| 1,343 rows, wall time | 16.9 s with 1 worker; 4.1 s with 8 workers |
| Pocket heavy atoms (8 Å) | median 245 (134-342); 23 beyond 12 Å at the median |
| Ligand heavy atoms | median 28 (6-53); none beyond 12 Å |
| Output size | 9.9 kB per row as float32 compressed npz, so about 0.95 GB for 96k rows |

The Isambard estimate for the full table (about 96k complexes at roughly 20 ms each) is about
30 CPU-minutes, i.e. under a minute on 72 workers. Archive extraction dominates the job.

## 5. Pocket clusters

The method (`pocket_clusters.py`) runs in five steps.

1. **Embedding.** X = [pocket_F | pocket_G] flattened gives 343 + 1,029 = 1,372 columns. StandardScaler, then PCA to 50 dimensions, both fit on all usable rows.
2. **Units.** Each PDBbind row is its own unit. BindingNet and BindingDB rows are grouped by `series_key` (template / folder), and the unit is represented by its medoid row. This keeps every complex built on one template in one cluster, which the hold-out in design §3.1 needs. It also keeps the linkage at about 23k units instead of 96k rows: the condensed distance matrix for 96k rows would be about 37 GB, and scipy copies it.
3. **Clustering.** `scipy.cluster.hierarchy.linkage(method="average", metric="cosine")`, which is the algorithm of sklearn's `AgglomerativeClustering(linkage="average", metric="cosine")`.
4. **Cut.** The cut is the smallest merge height at which at least 90% of the pairs of PDBbind units sharing a UniProt fall in one cluster. Pairs are pair-weighted. Units with an empty or `------` UniProt form no pairs. The rate is monotone along one dendrogram, so a bisection over merge heights finds the exact height. A 60-point grid of thresholds is reported as the curve.
5. **Per-row outputs.** `cluster_id` (0 = largest by rows; −1 = no usable descriptor), `cluster_size` (rows), `nearest_other_cluster_dist` (cosine distance from the row to the closest unit of another cluster) and `unit_key`.

**FEP check (92 series as units).** The series' protein name stands in for the UniProt, giving 48
proteins and 167 same-protein pairs: BACE1 13 series, CHK1 10, Factor Xa 6, HSP90 6, ...

The same-protein co-clustering curve (cosine distance cut):

| Threshold | Same-protein rate | Group-mean rate | Clusters | Largest (units) |
|---|---|---|---|---|
| 0.18 | 0.144 | 0.289 | 72 | 3 |
| 0.35 | 0.257 | 0.567 | 60 | 6 |
| 0.51 | 0.317 | 0.575 | 51 | 8 |
| 0.69 | 0.413 | 0.692 | 33 | 12 |
| 0.75 | 0.677 | 0.817 | 25 | 20 |
| 0.86 | 0.832 | 0.961 | 16 | 32 |
| **0.890 (chosen)** | **0.928** | **0.990** | **13** | **32** |
| 1.004 | 0.928 | 0.990 | 4 | 36 |
| 1.022 | 1.000 | 1.000 | 3 | 41 |

At the chosen cut there are 13 clusters, 4 of them singletons, and 8 contain two or more proteins. The
mixed clusters are mostly pocket families:

- **Cluster 0 (32 units): ATP sites.** All 10 CHK1 series, with CDK2, IRAK4, JNK1, TYK2, P38, JAK2, SYK, cMet, CK2, DLK, JAK1, ITK, EGFR, BTK and LigA. SHP2 is also here.
- **Cluster 2: aspartyl proteases.** 12 of 13 BACE1 series and renin.
- **Cluster 3: trypsin-like serine proteases.** Factor Xa ×6, thrombin ×3, HNE and β-tryptase.
- **Cluster 5:** HSP90 ×6, with one BACE1, scytalone dehydratase and urokinase.
- **Cluster 1:** the hydrophobic/PPI and GPCR pockets (BRD4, Mcl1, P2Y1, A2A, ER α, ...).
- **Misplaced.** HIV-1 protease pairs with Eg5 rather than with the aspartyl proteases.

A 90% same-target cut on these descriptors therefore gives family-level clusters. This is the
granularity design §3 asks for when holding out pockets. The pair-weighted rate is dominated by BACE1
(78 of the 167 pairs) and CHK1 (45). All 12 pairs still split at the cut involve the one BACE1 series
that sits with HSP90, and that series only joins at 1.02, where 3 clusters remain.

**Diagnostic, not adopted.** Alternative inputs were tried on the same 92 series:

| Input | Clusters at the 90% cut | Same-protein rate at about 20 clusters |
|---|---|---|
| Spec input | 13 | 0.68 |
| F only | 17 | 0.63 |
| Signed log1p of F and G | 16 | 0.70 |
| F and G normalised by occupancy F_00 | 11 | 0.71 |

None changes the picture, so the spec's standardisation is kept.

## 6. Outputs

**`zernike.npz`** (`np.load(..., allow_pickle=False)`). Rows are in `paired_meta.csv` order. Arm-4 pose
rows (`kind != 0`) are skipped and counted.

- **Identity:** `unique_id`, `source`, `ok`, `error` (str).
- **Descriptors:** `pocket_F` [N,7,49], `pocket_G` [N,21,49], `ligand_F` and `ligand_G` (float32).
- **Signed totals:** `pocket_total` and `ligand_total` [N,7] (float32), the signed sum of each channel's weights over the atoms the descriptor keeps (inside R). F_00 = |total| · (3/4π)/sqrt(4π), so F loses the sign of a channel total; the totals keep it. This matters for hydrophobicity: 94% of shell pockets have a negative Crippen total and 6% a positive one, which F_00 alone cannot tell apart (only G(occupancy, hydrophobicity) carried the sign before). For occupancy the total is the number of kept atoms.
- **Counts:** `n_pocket_atoms` (atoms in the mask), `n_ligand_atoms`, `n_outside_pocket`, `n_outside_ligand` (int).
- **Failed rows.** A row with `ok` False has NaN in all four descriptor arrays and both total arrays, and −1 in all four counts, whatever step failed. So `ok` equals the finiteness of any descriptor. Before this fix, a row whose pocket came out empty (2r1w and two BindingDB rows in the first full run) stored zeros and `n_pocket_atoms` 0, because the result was stored before the empty-pocket check.
- **Timing:** `seconds`.
- **Run parameters (0-d arrays):** `pocket_mode` (str), `pocket_radius` (NaN in shell mode), `order`, `radius` (the pocket descriptor R), `cutoff` (NaN in sphere mode), `ligand_radius`, `variant_note` (str). Older files lack these keys; they are shell-mode, order 12, R 12 Å, cutoff 8 Å.

**`zernike_summary.json`** contains:

- counts by source and failures (`error_counts`, with paths and numbers masked, and `failures_first`);
- per-source quantiles of seconds per row, protein parse time, pocket and ligand atom counts, and outside counts;
- `variant_note` (the `--variant-note` free text that labels an A/B run);
- the parameters: `order`, `radius`, `cutoff`, `pocket_mode`, `pocket_radius`, `ligand_radius` (null where the mode does not use one), `nl_index` and `channel_pairs`.

`--fep-dir` also writes `<out stem>_meta.csv` for `pocket_clusters.py`.

**Inputs are treated as untrusted:**

- paths are built only from validated CSV components (no absolute paths, no `..`, no NUL);
- file sizes are capped (protein 200 MB, ligand 50 MB) and so are atom counts (500k protein, 2k ligand);
- a bad row is recorded, never fatal;
- outputs are written atomically.

**`pocket_clusters.csv`** has the columns `unique_id, source, uniprot, cluster_id, cluster_size,
nearest_other_cluster_dist, unit_key`, in meta order, including rows with no descriptor.
**`pocket_clusters_summary.json`** contains:

- the threshold and the rule that chose it;
- the number of clusters and the size quantiles (rows and units);
- the co-clustering rate at the cut, pair-weighted and group-mean, and the number of pairs;
- the number of clusters mixing two or more UniProts (all rows, and PDBbind rows);
- quantiles of `nearest_other_cluster_dist`;
- the curve.

## 7. Isambard job

`sbatch jobs/featurise_zernike.sbatch`. It uses the Global Constraints flags, `--time=02:00:00`, and
`/local/user/$(id -u)/aevplig-paired/zernike-$SLURM_JOB_ID` with an EXIT trap.

A/B example: `POCKET_MODE=sphere POCKET_RADIUS=10 OUT_NAME=zernike_sphere10 VARIANT_NOTE="sphere 10 A"
sbatch jobs/featurise_zernike.sbatch`.

- **Archives.** It extracts from `/scratch/u6xq/hussain.u6xq/aevplig/raw/`:
  - `PDBbind_v2020_{refined,other_PL}.tar.gz`: ligand mol2 and protein pdb only, linked as `root/{refined-set,general-set}`;
  - `BindingNetv1.tar.gz`: `*.sdf` only;
  - `Crystal_Templates_for_BindingNet1.tar.gz`: `rec_h_opt.pdb`, symlinked into `BindingNetv1/<pdb>/`;
  - `surflex.tar`: extracted into `surflex/`, with a single wrapping `surflex/` also accepted.
  - `PDBbind_v2020_plain_text_index.tar.gz` is read in place by `pocket_groups.py`.
- **Steps.** It runs `featurise_zernike.py` with 72 workers (one BLAS thread each) into `$OUT_DIR/$OUT_NAME.npz` and `${OUT_NAME}_summary.json`. It then runs `pocket_groups.py` (no longer `pocket_clusters.py`) into `$OUT_DIR/pocket_groups.csv` and `pocket_groups_summary.json`, and prints both summaries. When `OUT_NAME` is not `zernike`, the groups files are named `${OUT_NAME}_pocket_groups.*`, so an A/B run never overwrites the main groups file.
- **Overrides:**
  - `OUT_DIR` (default `$WORK/data`) and `OUT_NAME` (default `zernike`; a plain file stem);
  - `POCKET_MODE` (`shell` or `sphere`; default `shell`, which passes `--radius 12 --cutoff 8` as before);
  - `POCKET_RADIUS` (sphere mode only; empty means the script default of 10 Å; set in shell mode, the job stops);
  - `VARIANT_NOTE` (stored in the summary and the npz);
  - `MERGE_COSINE` (passed to `pocket_groups.py --merge-cosine`; empty means no descriptor merge);
  - `LIMIT` (first N rows per source, for a smoke run) and `META`.
- **Status.** Not yet run on Isambard. The archive member names are those given in the task brief, and the job checks each archive and directory and fails with a message if one is missing. The argument building was checked locally with stubbed commands.

## 8. Target groups (`pocket_groups.py`)

```
pocket_groups.py --meta work/data/paired_meta.csv --pdbbind-index-tgz FILE (or --pdbbind-index-dir DIR)
                 --zernike work/data/zernike.npz --out work/data/pocket_groups.csv
                 [--merge-cosine 0.82] [--generic-min-uniprots 5] [--block 1024]
```

Descriptor clusters cannot reach 90% same-UniProt co-clustering without collapsing into one cluster, and
reach 0.9 purity only near singletons (`docs/pocket_descriptor_analysis.md` §3). The split is therefore
made by target, and the descriptor only measures how far each row sits from the training pockets of
other groups.

**Groups.** Rows are the `kind == 0` rows of the meta table (every row if it has no `kind` column).

1. **UniProt.** A row starts in the group of its `uniprot`. BindingNet rows carry their template's UniProt and BindingDB rows their folder PDB's (`build_paired_dataset.py`), so they join that group.
2. **No UniProt.** A row without a UniProt is in group `single:<anchor_pdb>`. For a PDBbind row this is `single:<unique_id>`. BindingNet and BindingDB rows on the same structure join it, so rows built on one structure (a template, a BindingDB folder) never split. In the full table these are 345 PDBbind, 103 BindingNet and 1,973 BindingDB rows (the BindingDB anchors are not in the PDBbind index).
3. **Name merge.** UniProts that share a normalised protein name of `INDEX_general_PL_name.2020` are merged: upper case, non-alphanumerics read as spaces. The merge takes the connected components of the UniProt-name graph over every index entry with a UniProt. The file is read with `build_paired_dataset`'s tar-member and comment helpers, the same way `read_pdbbind_index` reads it. That function returns only the UniProt column, so the names are parsed here.
4. **Generic names.** A name that maps to `--generic-min-uniprots` (5) or more UniProts in the index is generic and is not merged on. The skipped names are listed in the summary. This stops catch-alls such as INTEGRASE, BETA LACTAMASE and HIV 1 PROTEASE from chaining hundreds of UniProts together. The analysis's ungated name merge built a 623-unit group out of 62 UniProts.
5. **Descriptor merge (optional).** With `--merge-cosine T` the groups are merged by average linkage (UPGMA) while the smallest group-average cosine distance is below T; the default is off.
   - **Units.** Units weigh equally. A PDBbind row is a unit. BindingNet and BindingDB rows form one unit per `series_key` (`pocket_clusters.unit_keys`), represented by the normalised mean of their rows' unit vectors.
   - **No distance matrix.** For unit vectors the mean pairwise cosine distance between groups A and B is 1 − mean_A · mean_B, so the merge is exact UPGMA over units. The test checks it against scipy's average linkage.
6. **group_id.** It is `uniprot:<the member UniProt with most rows>`, or the `single:` key of a group without a UniProt.

**Descriptor.** The vector is [pocket_F | sign(pocket_G)·sqrt|pocket_G|] flattened, with each of the 1,372 columns standardised over the rows with a usable descriptor. Zero-variance columns keep scale 1, there is no PCA, and vectors are compared by cosine. This is the best variant of the analysis (R@1 0.756, against 0.689 for the PCA-50 input of `pocket_clusters.py`).

**nn_train_dist.** This is the cosine distance from a row's descriptor to the nearest descriptor among the `split == "train"` rows of a **different** group. The row itself never counts.
- **NaN** when the row has no usable descriptor (failed, or absent from the npz), or when every train row is in its own group.
- **Computation.** One float32 matmul of unit vectors per block of `--block` query rows, never a full distance matrix. The full table (96,947 × 91,761) takes about 1 minute and 7 GB on a laptop.

**Outputs.**
- **`pocket_groups.csv`** has the columns `unique_id, source, split, uniprot, group_id, group_size_rows, group_size_units, nn_train_dist`, with every meta row in meta order.
- **`pocket_groups_summary.json`** contains:
  - group counts (UniProt, single, merged) and size quantiles in rows and units;
  - the 20 largest groups, each with its UniProts, top names (counted by distinct structures) and rows per split;
  - the generic names skipped, each with its index and meta UniProt counts;
  - the fraction of rows whose nearest training pocket (any group) lies in another group, overall and per split;
  - `nn_train_dist` quantiles, overall and per split;
  - the fraction of each split's rows whose group holds other train rows;
  - the number of groups spanning two or more splits;
  - the descriptor-merge record and the npz run parameters.

**Local run on the first full table** (96,950 rows, 96,947 with descriptors; shell mode):

| | Default (no descriptor merge) | `--merge-cosine 0.82` |
|---|---|---|
| Groups (UniProt / single) | 4,014 (3,314 / 700) | 893 (873 / 20) |
| Groups with ≥ 2 UniProts | 382 (931 UniProts) | 472 (3,462 UniProts) |
| Largest group | BACE1, 4,263 rows (4.4%), 624 units | CHK1/JAK2/SYK kinases, 5,998 rows (6.2%), 37 UniProts |
| Median group size, rows / units | 2 / 1 | 8 / 6 |
| Nearest training pocket in another group (all rows / valid / test / series_valid) | 6.7% / 23.8% / 6.9% / 13.7% | 3.3% / 13.3% / 4.3% / 3.2% |
| nn_train_dist median (valid / test) | 0.50 / 0.47 | 0.59 / 0.57 |
| nn_train_dist q10 (valid / test) | 0.22 / 0.23 | 0.35 / 0.34 |
| Groups spanning ≥ 2 splits | 562 | 325 |

- **Generic names.** The default run skips 49 of them. The largest are HIV 1 PROTEASE (47 UniProts), BETA LACTAMASE (31) and NEURAMINIDASE (27).
- **Under-merging.** The generic rule also keeps true single targets apart. HIV-1 protease isolates (413 rows) fall into 43 groups, and acetylcholinesterase orthologs and dihydrofolate reductases into 5 and 8. The 0.82 merge brings them to 3, 1 and 1. Sequence clustering remains the proper fix; the name merge is a stopgap (analysis §5 i).
- **The existing split is not group-disjoint.** 92-100% of valid, test and series_valid rows sit in a group that also holds train rows. Holding out whole groups needs a new split; this script does not assign one.
