"""Census of the species-0 pocket atoms of a store: the atoms kept for the pocket that the 22-type
table does not type (old-style hydrogen names such as ``1HA``, heavy atoms of unmapped residues such
as ``MSE``, rare unmapped heavy-atom names), per source, split by ``prot_element`` and by whether
the atom name is in the key table's names (``prot_atom >= 0``) or not (``prot_atom == -1``).

    python -m aevplig_jax.store_census --store DIR [--out FILE.json]

Prints, per source, the records, pocket atoms and species-0 atoms, then a table ``| source | element
| atom -1 | atom known | species 0 |``; ``--out`` also writes the census as JSON. Imports numpy, the
standard library and the package's store module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aevplig_jax.pdb_atoms import ELEMENTS, OTHER_ATOM
from aevplig_jax.store import Store


def census(store_path: Path) -> dict:
    """``{source: {records, atoms, species0, by_element: {element: {atom_unknown, atom_known}}}}``
    in store order of first appearance; elements in ``ELEMENTS`` order, absent ones left out."""
    totals: dict[str, dict] = {}
    with Store(store_path) as store:
        names = store.metadata.get("elements", list(ELEMENTS))
        for rid in store.ids:
            record = store.get(rid)
            entry = totals.setdefault(record.source, {"records": 0, "atoms": 0, "species0": 0,
                                                      "counts": np.zeros((len(names), 2), np.int64)})
            pocket = record.pocket
            untyped = pocket.species == 0
            entry["records"] += 1
            entry["atoms"] += len(pocket.species)
            entry["species0"] += int(untyped.sum())
            unknown = (pocket.atom[untyped] == OTHER_ATOM).astype(np.int64)
            np.add.at(entry["counts"], (pocket.element[untyped].astype(np.int64), 1 - unknown), 1)
    result = {}
    for source, entry in totals.items():
        counts = entry.pop("counts")
        entry["by_element"] = {names[code]: {"atom_unknown": int(counts[code, 0]),
                                             "atom_known": int(counts[code, 1])}
                               for code in range(len(names)) if counts[code].any()}
        result[source] = entry
    return result


def render(result: dict) -> str:
    lines = []
    for source, entry in result.items():
        share = entry["species0"] / entry["atoms"] if entry["atoms"] else 0.0
        lines.append(f"{source}: {entry['records']} records, {entry['atoms']} pocket atoms, "
                     f"{entry['species0']} species 0 ({100 * share:.2f}%)")
    lines += ["", "| source | element | atom -1 | atom known | species 0 |",
              "| --- | --- | --- | --- | --- |"]
    for source, entry in result.items():
        for element, counts in entry["by_element"].items():
            lines.append(f"| {source} | {element} | {counts['atom_unknown']} | "
                         f"{counts['atom_known']} | "
                         f"{counts['atom_unknown'] + counts['atom_known']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True, help="store directory")
    parser.add_argument("--out", type=Path, help="also write the census as JSON")
    args = parser.parse_args(argv)
    result = census(args.store)
    print(f"[census] {args.store}: species-0 pocket atoms by element and atom code")
    print(render(result), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
