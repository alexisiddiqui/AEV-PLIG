# jax-Ka surface/augmentation line, and a 3D Zernike soft-negative design for AEV-PLIG

Repository: `/Users/alexi/Documents/jax-KA`, read through git only (working tree untouched). All file:line references are to `origin/main` = 95091c0 ("Merge Coulson augmentation and comparison work"). Every augmentation file is byte-identical on `origin/coulson-augmentation` (998d777). Within `benchmarks/`, `docs/`, `_HPC/submission/` and the store module, the two refs differ only in `docs/TORCH_STRUCTURES.md` and one other non-augmentation file. Elsewhere, main also carries the other branches' work. Paths below are relative to the repo root.

---

## PART 1: what the surface/augmentation line is

### (a) End to end

**Purpose.** The line asks one question: do ANM-perturbed, OpenMM-repaired protein conformers keep their local surfaces (geometry and chemistry)? It is a diagnostic and candidate filter for the free-partner conformer augmentation used in pKa/interface training. Surfaces are never a model input. The final recipe dropped the surface gate: "No local surface-preservation or APBS gate is applied" (`docs/augmentation_recipe_readiness.rst:8`).

**Which structures.** Protein-only heavy-atom units from ATLAS, PINDER (free partners) and pKPDB.
- Conformers come from ProDy ANM on CA atoms (15 A cutoff, 5 non-zero modes; `src/pkatrain/augmentation_generate.py:134`). There are 8 directions x 0.5/1/2 A target CA RMSD = 24 candidates.
- Each candidate gets a 100-step capped OpenMM amber14 minimisation with a 1000 kJ/mol/nm^2 CA restraint (`augmentation_generate.py:143,174-178`).
- The MaSIF audit uses 30 parents: one typical and one upper-tail per dataset x 5 length bins, deliberately enriched for local change (`benchmarks/augmentation_masif_audit.py:53-66`).
- Views per parent: prepared native, repaired zero, 4 selected conformers, historical zero, and one fixed direction at 0.5/1/2 A (`augmentation_masif_audit.py:135-157`).
- Smoke controls add a rigidly moved zero and a reseeded repeat (`augmentation_masif_audit.py:232-235`).

**Two surface constructions.**
1. **Cheap SAS (pure numpy/scipy), `benchmarks/augmentation_surface_audit.py`.**
   - Shrake-Rupley points on inflated vdW spheres (radius + 1.4 A, 256 Fibonacci points). The sphere directions follow each atom's residue backbone frame, which makes a rigid motion exactly invariant at finite sampling (`:54-86`).
   - Chemistry is a 4-class residue label (cationic/anionic/hydrophobic/polar, `:26-28`) plus area-weighted exposed amino-acid composition (`:114-116`).
   - Patches are 10 A Euclidean crops (`:109-111`) around up to 32 farthest-point anchor residues with at least 20 A^2 exposed area (`:123-144`).
2. **MaSIF route, `benchmarks/augmentation_masif_bridge.py`.** A Python 3.6 adapter runs inside the author container `pablogainza/masif` under udocker P1 mode. The host's GCC-8 libstdc++/libgcc are mounted at `/compat` because the bundled PyMesh needs CXXABI_1.3.11 (`_HPC/submission/augmentation-masif-setup.sbatch`; `augmentation_masif_audit.py:160-173`).
   - **Protonation:** Reduce `-HIS -NOFLIP`. His HD1/HE2 are fixed to the prepared source's pH-7 inventory, heavy atoms are verified unmoved (<= 0.0005 A), and CONECT records are remapped (`bridge:80-123`).
   - **Mesh:** MSMS (1.5 A probe), then PyMesh `fix_mesh` at 1 A resolution, then normals (`bridge:277-282`).
   - **Per-vertex channels:**
     - shape index from PyMesh mean/Gaussian curvature (`bridge:195-202`);
     - MaSIF geometric H-bond donor/acceptor potential (`computeCharges`), projected onto the regularised mesh (`bridge:278,283`);
     - hydropathy: upstream `computeHydrophobicity`, a per-residue Kyte-Doolittle value, divided by 4.5 (`bridge:278,284,204`);
     - electrostatics: PDB2PQR `--ff=parse --noopt --nodebump`, then APBS, then `multivalue` at the vertices; potential divided by 10, then upstream clip/normalise (`bridge:126-146,204`). Charge-assignment identity is hashed and must match across views (`audit:277-279`);
     - distance-dependent curvature, computed per patch (`bridge:211-213`);
     - residue owner per vertex (`bridge:285-286`) and per-vertex area (`bridge:149-153`).
   - **Anchors:** residue-frame-transported centres are mapped to the nearest vertex on the same residue within 2.5 A, with at least 3 required (`bridge:293-301`).
   - **Geodesic patches:** 12 A, at most 200 vertices. Upstream fast-MDS polar coordinates are computed only at the selected centres, which avoids the upstream quadratic all-centre allocation (`bridge:156-191`).
   - **Descriptor:** frozen MaSIF-search `sc05/all_feat` produces 80-d fingerprints on CPU with flip=False (`bridge:46-67,310-312`).
   - **Preflight:** the adapter is checked against the unmodified upstream pipeline on a closed octahedron (`bridge:219-263`).
   - **Electrostatics-free mode:** `geometry_only`/`raw_only` modes skip APBS and the network (`bridge:288-305`). Deferred APBS on the retained mesh is `bridge:348-358`.

**Patch supports.** Patch-scale sweep at 6/9/12/15 A geodesic radius, with no vertex cap: deduplicated edge graph, then `scipy.sparse.csgraph.dijkstra` with a limit, giving nested supports (`benchmarks/augmentation_patch_scale.py:22-38`).

**Reference-vs-moving metrics.**
- Global and patch-local Kabsch CA/heavy RMSD (`surface_audit.py:31-41`).
- Area-weighted symmetric nearest-surface distance, normal angle, chemistry-mismatch fraction, and matched-sample RMSD (`surface_audit.py:89-106`).
- Per-channel area-weighted nearest-vertex RMSE (shape index, H-bond, electrostatics, hydropathy) after local alignment, plus residue-exposure total variation (`patch_scale.py:41-52`).
- Fingerprint L2/cosine; surface coverage within 2.5 A; zero-gallery retrieval rank against three distractor pools: same protein with <10% support overlap, other protein from a different source group, and the 32 other-protein patches nearest in mean chemistry (`benchmarks/augmentation_masif_report.py:43-130`).
- Parent/view/patch-balanced quantiles and a 2000-draw protein bootstrap (`masif_report.py:19-40`).

**Filter.** Uncapped 9 A patches. Per-metric thresholds are the 95th percentile of rigid-rebuild controls on 18 calibration parents; a candidate fails above a calibrated bad-patch fraction; 12 parents are held out; 3 views are chosen by greedy farthest CA RMSD (`benchmarks/augmentation_surface_filter.py:24-28,30-46,150-172`). The staged variant meshes without APBS first and queries APBS only in diversity order (`augmentation_surface_staged.py:43-73`; `augmentation_surface_filter_summary.py:13-40`). One-core rebuild parity accepts a bijective vertex permutation (`augmentation_surface_filter_cpu.py:18-41`).

**Store and caching.**
- `src/pkatrain/augmentation_store.py` defines "KAAR1" records. One zstd frame per parent holds MAGIC + canonical JSON header (metadata + dtype/shape/offset per array) + raw buffers (`:19-72`). Shards are at most 2 GiB. The writer has an flock, checkpoint/resume, a per-record sha256, a 5,000-file budget, and the manifest as publication marker (`:75-180`). The reader is `AugmentationStore.raw()` (`:220-272`); merging per-worker releases is `finalize_workers` (`:274+`).
- The MaSIF audit stores one record per parent, with arrays named `vNN_<field>` and roles mapped to frame ids (`audit:262-265,356-367`).
- Frame cache: keyed by sha256(parent id). The signature fingerprints the coordinate hash, environment/binary/checkpoint hashes, config, His states, host and bridge code sha, anchor centres, role and seed. A frame is reused only if the receipt signature and npz sha match (`audit:219-261`). Intermediates go to node-local scratch and are pruned (`audit:176-191,269-274`).
- The `augmentation_cache_*.py` scripts are not about surfaces. They pack, transfer and report the free-partner graph-feature cache for ANM views (manifest-only SSH transfer: `augmentation_cache_transfer.py:89-187`).

**Documentation.** `docs/augmentation_masif_audit.rst` (all of the above, including the patch sweep, filter, staged APBS, ten-mode and experimental apo/holo comparisons), `docs/augmentation_recipe_readiness.rst`, `docs/augmentation_infrastructure.rst`. No `experiments/*/` note documents the MaSIF/surface work: notes 28-30 from the same commit are pKAI/oGQT, and 16/17 cover side-chain context augmentation. Measured per-frame timings are in Coulson receipts (`timings`: hydrogenation_s, surface_s, electrostatics_s), not in git.

### (b) What "Coulson augmentation" is

Coulson is a CPU cluster, not a method. Evidence: paths `/home/coulson/oc/lina4225/...`, partition `amd96`, and a guard that excludes `comp1400`. Branch head 998d777, "Add augmentation pipeline and pKAI/oGQT comparison work from Coulson", is "Uncommitted work from the Coulson benchmark checkout, 10-29 Oct". It covers the surface filter, MaSIF audit, patch scale, ten-mode ANM, pool generation, feature-cache build/transfer and online features, plus pKAI/oGQT comparisons. It was merged to main as 95091c0 (2026-11-03). `git log -S Coulson --all` returns only 998d777, 5bb7aa9 (relation-map plotting) and 3f7939c (2026-10-13: packed ANM augmentation infrastructure and GH200 repair pilot).

### (c) Reusable for protein-ligand pockets

| Need | Function (file:line) | Reuse notes |
|---|---|---|
| Per-PDB molecular surface + channels | `frame()` `augmentation_masif_bridge.py:266-345`, driven by `execute_frame`/`bridge` `augmentation_masif_audit.py:160-191` | Protein only (see d). Needs >= 3 anchors; for a pocket, pass pocket-lining residues as anchors. `raw_only` + `geometry_only` gives mesh, normals, owner, area, shape index, H-bond, hydropathy without APBS |
| Electrostatics on a fixed mesh | `electrostatics()` `bridge:126-146`, `deferred_electrostatics` `:348-358` | PARSE has no ligand parameters; apo pocket only |
| Shape index | `features_at()` `bridge:194-216` (lines 195-202) | Needs PyMesh curvature |
| Cheap, container-free surface | `surface()` `augmentation_surface_audit.py:62-86`, `sphere()` `:54-59` | Uses backbone frames for directions; for ligand atoms use the plain `sphere()` directions. RADII cover only C/N/O/S (`:24`) |
| Patch around a given centre | `crop()` `surface_audit.py:109-111` (Euclidean); `supports()` + `graph()` `augmentation_patch_scale.py:22-38` (geodesic, any radius, uncapped) | Centre can be the ligand centroid, or the nearest vertex to it |
| Rotation-invariant patch summaries already present | `composition()` `surface_audit.py:114-116` (area-weighted 20-aa exposed histogram); 4-class `CHEM` (`:26-28`); `atom_exposure()` `:119-120`; mean channel vector per patch (`masif_report.py:50`); frozen 80-d MaSIF fingerprint (`bridge:62-67`; rotation-invariant by angular max-pooling over 16 rotations) | No moments, spherical harmonics or shape histograms (see e) |
| Alignment-based comparison | `fit()`/`rms()` `surface_audit.py:31-41`; `shape()` `:89-106`; `surface_metrics()` `patch_scale.py:41-52` | Need a superposition, so not invariant |
| Property-matched / distractor negatives | `collect()` `masif_report.py:104-128`: same-protein distractors with <10% support overlap, other-protein distractors excluding the same source group, 32 chemistry-matched other-protein patches | Directly reusable logic for construction-matched negatives |
| Balanced statistics | `weights()`, `quantile()`, `bootstrap_margin()` `masif_report.py:19-40` | |
| Anchor selection | `anchors()` `surface_audit.py:123-144` | Farthest-point exposed residues |
| Storage, caching, transfer | `ReleaseWriter`/`AugmentationStore`/`fingerprint` `augmentation_store.py:29-30,75-272`; signature cache `audit:244-261`; `augmentation_cache_transfer.py` | One record per complex, inode-safe |

### (d) Runtime and tooling

**MaSIF route.**
- udocker 1.3.17 in a venv with the `pablogainza/masif` image: Python 3.6, TF1, PyMesh, MSMS, APBS, PDB2PQR, multivalue, Reduce. The host compat libraries are mounted.
- CPU only; the GPU is explicitly disabled (`bridge:52`; `audit:164`).
- Budget: 2 GiB per CPU, 4 cores per frame in the pilots, Slurm plus the `require_compute` guard.
- APBS is the stage the staged design defers, which tells you it is the bottleneck.

**Container-free pieces.**
- Cheap SAS and patch scale need numpy/scipy; the store needs `zstandard`; the conformation reference reads PDBs with biotite.
- Nothing needs a GPU.

**Ligand caveats.**
- `write_pdb` emits ATOM records for protein residues only (`audit:110-132`).
- The bridge parses residue owners from MSMS names and assumes sequential numbering (`bridge:285`).
- PARSE cannot charge ligands.
- `surface()` radii lack halogens, P and metals.
- A pocket surface is therefore apo: ligand stripped.

### (e) "zernike" / "moment" across all branches

- **zernike:** none, in any of the 14 remote branches or the 2 local branches (`*.py`, `*.md`).
- **moment:** 10 hits in 7 files, present on 13 branches (absent on `origin/solver-newton`). All of them are Adam optimiser moments, none geometric:
  - `experiments/3_paired_complex/22_gqt_backbone_5k_decay.md:3`
  - `experiments/3_paired_complex/31_gqt_converged_size_sweep.md:7`
  - `src/pkatrain/gqt_query_norm_pilot.py:88,109`
  - `src/pkatrain/gqt_skip_pilot.py:48,73,142`
  - `src/pkatrain/graph_decay.py:70`
  - `src/pkatrain/graph_pkmod_long.py:77`
  - `src/pkatrain/graph_shift_weight.py:150`

---

## Literature pins

- **3DZD:** Novotni & Klein 2003, ACM Solid Modeling 216-225, doi:10.1145/781606.781639. Invariants are F_nl = ||Omega_nl.||. Count: N=8 gives 25, N=12 gives 49, N=15 gives 72, N=20 gives 121 (checked numerically).
- **Ligand shape:** Venkatraman, Chakravarthy & Kihara 2009. This is J Cheminform 1:19, not JCIM. Connolly surface, shape only, order 12 best, about 1 s per molecule (https://pmc.ncbi.nlm.nih.gov/articles/PMC2820497/).
- **Kihara line:**
  - Patch-Surfer, Sael & Kihara 2012: Proteins and BMC Bioinf 13(S2):S7.
  - PL-PatchSurfer, Hu et al. 2014, IJMS 15:15122. Shape plus electrostatics split into positive and negative (144 numbers), 5 A patches, seeds 3 A apart (https://pmc.ncbi.nlm.nih.gov/articles/PMC4200761/).
  - PL-PatchSurfer2, Shin et al. 2016, JCIM 56:1676. Adds XlogP3 atom hydrophobicity (Fermi-weighted field) and +1/-1 donor/acceptor voxels; N=15; feature weights shape 1.0, electrostatics 0.4, hydrophobicity 0.1, H-bond 0.2; DUD decoys; server kiharalab.org/plps2 (https://pmc.ncbi.nlm.nih.gov/articles/PMC5037053).
  - 3D-SURFER, La et al. 2009, Bioinformatics 25:2843; EM-SURFER 2015, BMC Bioinf 16:181.
- **Other pocket work:** Di Rienzo et al. 2020, JCIM, doi:10.1021/acs.jcim.9b01066. Pocket 3DZD as shape plus positive/negative potential (3 x 121), with complementarity measured by pairing a pocket's positive channel against the partner's negative.
- **Code:**
  - pyzernike / `zernike-descriptors`: Python plus OpenMP wrapper of Novotni's GPL C code (https://github.com/maurerv/pyzernike).
  - Novotni C fork (https://github.com/DAlvGar/ZernikeMoments).
  - BioZernike: Java, MIT, behind the RCSB shape search (https://github.com/biocryst/biozernike; Guzenko et al. 2020, PLoS CB).
  - ZMPY3D: NumPy, CuPy, TF and PyTorch backends, GPL-3; order 20 at a 100^3 grid takes about 0.1-0.2 s per calculation from its benchmark table (https://github.com/tawssie/ZMPY3D; Lai et al. 2024, Bioinf Adv).
  - Mesh-exact moments: zernike3d (https://github.com/jerhoud/zernike3d; Houdayer & Koehl 2022) and pkzernike (https://codeberg.org/mbanach/pkzernike). Shape only, no property weights.
  - No public Kihara 3DZD repository was found on github.com/kiharalab.
- **Bias and decoys:**
  - Chen et al. 2019, PLoS ONE 14:e0220113: DUD-E analogue and decoy bias.
  - Sieg, Flachsenberg & Rarey 2019, JCIM 59:947.
  - DeepCoy, Imrie, Bradley & Deane 2021, Bioinformatics 37:2134.
  - Volkov et al. 2022, J Med Chem 65:7946: interaction terms add nothing over ligand- or protein-only descriptors, which is our arm-1 symptom.
- **Placement and data:**
  - RDKit `rdShapeAlign` (PubChem align3d, 2024.09): about 0.4 ms per alignment.
  - BindingNet v1 (Li et al. 2024, JCIM; https://github.com/hnlab/BindingNet): 69,816 complexes built by MCS alignment of ChEMBL ligands onto PDBbind-template ligands. Overlay placement is therefore already one of AEV-PLIG's positive constructions.
  - AEV-PLIG: Valsson et al. 2025, Commun Chem 8:41.
  - Ross et al. 2023, Commun Chem 6:222.

---

## PART 2: design proposal (sections 1-5, 895 words)

### 1. Descriptor

**Fields, not surfaces.** The pocket is the protein heavy atoms within 8 A of the cognate ligand, with the ligand removed. The ligand is its heavy atoms. Each channel c is a density f_c = sum_i w_ic G(x - r_i; 1 A), centred on the cognate-ligand (or ligand) centroid and scaled into the unit ball with a fixed R = 12 A. Fixed R keeps size, which must then be matched.

**Channels.**
- Occupancy: protein atoms weighted by apo SASA fraction.
- Hydrophobicity: Wildman-Crippen atom logP on both sides, so both sit on one scale. Kyte-Doolittle/4.5 per residue, as in MaSIF, is a protein-only option.
- Positive and negative charge as separate channels, because signed fields cancel. Ligand: Gasteiger. Protein: Amber/PARSE partials or formal charges.
- Aromatic ring atoms; donors; acceptors (RDKit features; residue templates).

**Invariants.** Use N = 12, giving 49 invariants per channel. Per-channel norms are unchanged when each channel rotates independently, so they lose where a charge sits relative to a hydrophobe. Add cross-channel invariants Re sum_m Omega^a_nlm conj(Omega^b_nlm) (SOAP-style). About 700 dimensions, then PCA. For point clouds, Omega_nlm = sum_i w_i conj Z_nlm(r_i), one matrix product with no grid, mesh or APBS. Validate against ZMPY3D grids.

### 2. Soft negatives

**Rule.** Each negative ligand must also be a positive in its own pocket, and each negative pocket must host its own positives. Then neither a ligand-only nor a pocket-only model can satisfy the hinge. "Near ligand, far pocket" breaks this rule. Far pockets are other families, so target identity solves it, as GQT's cross-complex decoys learned.

**A. Shape twin, pocket fixed.**
- Selection: for (P_i, L_i, y_i), pick L_k from another sequence cluster. It must be in the nearest 1% of L_i on occupancy channels and have the worst property complementarity to P_i, measured as the cross-object overlap of moments in the pocket frame.
- Placement: ETKDG conformers, shape-only `rdShapeAlign` onto L_i, then restrained clash relief.
- Construction match: L_i is regenerated through the identical path and kept as a positive at y_i.
- Loss: max(0, f(P_i,L_k) - f(P_i,L_i^ov) + 1 pK), plus regression on L_i^ov.
- Shortcut: property marginals. Match MW, charge, logP and aromatic-count distributions to the positives.
- False negatives: drop same-Pfam/UniProt and ChEMBL-measured pairs.

**B. Pocket twins, ligands swapped both ways.** Pair pockets from different families that are near in occupancy and far in property channels. Overlay L_i onto L_j in P_j and vice versa, with both self-overlays included. Each ligand and pocket is then positive once and negative once in the same batch. Resample twins per epoch from k-NN lists so they cannot become a lookup.

**C. In-place edits of the cognate ligand.**
- Edits: re-type atoms at the crystal pose using common matched-molecular-pair edits (COO- or NH3+ to CH3, phenyl to cyclohexyl, OH to F, donor to acceptor), then restrained minimisation in place.
- Each edit is applied twice:
  - where the ligand's field overlaps the pocket's complementary field, as a hinge with a contact-scaled margin;
  - at a solvent-exposed site, as a two-sided |f - y_i| <= delta.
- Construction match: the positive gets a null edit through the same pipeline.
- Effect: the transformation itself carries no label; its location in the pocket does. Zernike only picks positions here; contact rules would also do.

**Gates.** All must be at chance AUROC: a ligand-only classifier (negatives vs positives), a pose-quality classifier (clashes, buried fraction, contact distances) on overlay positives vs overlay negatives, and a family-ID classifier for B. Arm 4's hinge stays the baseline.

### 3. Pocket characterisation

Cluster PCA'd pocket descriptors (threshold: at least 90% of same-UniProt pockets co-cluster).

1. **Hold out pocket clusters.** Also drop BindingNet and BindingDB-DCS complexes built on templates in those clusters. This does not make the model use the protein. It makes the full-vs-ligand-only gap measurable, because target-specific chemotype memory can no longer transfer.
2. **Stratify batches by cluster.** Draw arm-1 pairs within a cluster, across targets.
3. **Report Ross targets by distance.** Plot each target's Kendall tau for both models against its nearest-training-pocket distance and sequence identity.

### 4. Cost (about 96k complexes)

**Atom-field route (estimates; benchmark 100 complexes first):**
- RDKit typing: 10-50 ms.
- Point-cloud moments: 10-50 ms per object.
- Exposure: jax-Ka `surface()` takes 0.5-1 s per 1,500-atom shell; freesasa takes milliseconds.
- Total: 0.1-1 s per complex, 3-27 CPU-h, under 30 min in one 72-CPU Isambard job.

**Other routes:**
- Voxels (ZMPY3D, 16 grids x 0.15 s): about 1 node-hour.
- MaSIF/APBS: container, ligand stripping, no ligand charges. Use it only on about 500 pockets, to check that atom fields track the Poisson-Boltzmann potential.
- Overlays: ETKDG once per ligand (30-80 CPU-h), plus about 0.1 s of alignment per complex.

**jax-Ka reuse:** `surface()`, `crop()`, `composition()`, `fit()`, the KAAR1 store, signature caching, and the distractor/chemistry-matched negative logic in `masif_report.collect`.

### 5. Open questions

1. Does "soft negatives constructed from hydrophobicity, charge, aromaticity features" mean perturbed property fields of the same ligand or pocket (C), or other ligands and pockets chosen by descriptor distance (A/B)? Does "soft" mean a one-sided bound or a down-weighted target?
2. May positives be regenerated through the overlay path, keeping their measured labels? Should negatives be confined to BindingNet-like ligands, since overlay placement is already BindingNet's construction?
3. Should the pocket-cluster split replace the held-out-series test or run alongside it? Should Ross targets define the held-out clusters?
