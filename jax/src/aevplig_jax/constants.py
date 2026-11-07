"""Fixed constants of the AEV-PLIG reproduction and the protein atom-type table.

Values are copied from the design spec (docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md),
which takes them from the oxpig release (generate_bindingnet_graphs.py, model_defs.py, training scripts).
Imports only the standard library.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Radial AEV (ANI-2x radial terms, protein neighbours only).
RC = 5.1  # radial cutoff, A
ETA = 19.7  # radial decay
SHIFTS = (0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
          2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83)  # radial shifts Rs, A
N_SPECIES = 22
N_SHIFTS = 16
AEV_DIM = 352  # N_SPECIES * N_SHIFTS, species-major columns

# Ligand graph features.
ATOM_SYMBOLS = ("F", "N", "Cl", "O", "Br", "C", "B", "P", "I", "S")
BOND_TYPES = (1, 12, 2, 3)  # RDKit bond types as numbers: single, aromatic, double, triple
RDKIT_DIM = 15
NODE_DIM = 367  # RDKIT_DIM + AEV_DIM
EDGE_DIM = 4

# Model.
HIDDEN = 256
HEADS = 3
LAYERS = 5
DENSE = (1024, 512, 256, 1)
ATTN_SLOPE = 0.2  # leaky-relu slope inside GATv2 attention
ACT_SLOPE = 0.01  # leaky-relu slope of the activations
BN_MOMENTUM = 0.1
BN_EPS = 1e-5
MASK_FILL = -1e9  # attention logit for masked neighbour slots

# Training protocol.
LR = 1.2291937615434127e-4
BATCH = 128
EPOCHS = 200
SELECTION_WINDOW = 8
SEEDS = (100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621)


def species_table(keys_csv: Path) -> dict[str, int]:
    """Map ``PDB_ATOM`` keys (e.g. ``ALA-CA``) to type indices 1..22.

    The index of an ``ATOM_TYPE`` is its order of first appearance in ``keys_csv``
    (``data/PDB_Atom_Keys.csv``), starting at 1, as in generate_bindingnet_graphs.py.
    """
    type_index: dict[str, int] = {}
    table: dict[str, int] = {}
    with open(keys_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            atom_type = row["ATOM_TYPE"]
            if atom_type not in type_index:
                type_index[atom_type] = len(type_index) + 1
            table[row["PDB_ATOM"]] = type_index[atom_type]
    return table
