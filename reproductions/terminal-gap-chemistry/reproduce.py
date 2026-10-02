"""Show that supplied carbon-bound H cannot silently become carboxylate OXT.

Run with tmol and AtomWorks installed. No private datasets or network needed.
Optional --source-root selects an exported tmol tree for historical comparison.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path)
p.add_argument('--fixture', type=Path, default=Path(__file__).with_name('3t0x_B106_prepared.pdb'))
p.add_argument('--output', type=Path)
p.add_argument('--expect', choices=('rejected', 'silent-conversion'), default='rejected')
a = p.parse_args()
if a.source_root:
    root = str(a.source_root.resolve())
    for finder in sys.meta_path:
        if 'tmol' in getattr(finder, 'known_packages', ()):
            finder.search_paths = [root]
    sys.path.insert(0, root)

import numpy as np
import torch
import tmol
import atomworks
from tmol.io import pose_stack_from_biotite
from biotite.structure.io.pdb import PDBFile

structure = PDBFile.read(a.fixture).get_structure(model=1, include_bonds=True)
carbon = int(np.flatnonzero(structure.atom_name == 'C')[0])
hydrogen = int(np.flatnonzero(structure.atom_name == 'HXT')[0])
assert structure.bonds.get_bonds(hydrogen)[0].tolist() == [carbon]
assert 'OXT' not in structure.atom_name
original = structure.copy()
result = {
    'tmol_import': tmol.__file__,
    'atomworks_import': atomworks.__file__,
    'fixture_sha256': hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
    'source_atom_count': len(structure),
    'source_carbon_hydrogen_distance_angstrom': float(np.linalg.norm(structure.coord[carbon] - structure.coord[hydrogen])),
    'source_carbon_neighbors': [
        {'name': str(structure.atom_name[i]), 'element': str(structure.element[i]), 'bond_order': int(kind)}
        for i, kind in zip(*structure.bonds.get_bonds(carbon))
    ],
    'source_formal_charges': structure.charge.tolist() if 'charge' in structure.get_annotation_categories() else None,
    'source_pdb_charge_fields': 'blank / not explicitly declared',
}
try:
    pose = pose_stack_from_biotite(structure, torch.device('cpu'), no_optH=True)
except ValueError as error:
    result.update(outcome='rejected', error=str(error))
    assert 'Unsupported terminal hydrogen count at B:106:VAL/C' in str(error)
else:
    block = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
    names = [atom.name for atom in block.atoms]
    neighbors = [b if first == 'C' else first for first, b, *_ in block.bonds if 'C' in (first, b)]
    result.update(outcome='converted_to_different_chemistry', block_type=block.name, output_atom_names=names, output_carbon_neighbors=neighbors, supplied_HXT_retained='HXT' in names, new_OXT_added='OXT' in names)
    assert 'OXT' in names and 'HXT' not in names
    offset = int(pose.block_coord_offset[0, 0])
    result['generated_OXT_coordinates'] = pose.coords[0, offset + block.atom_to_idx['OXT']].tolist()
    heavy = original.element != 'H'
    indices = [offset + block.atom_to_idx[n] for n in original.atom_name[heavy]]
    result['max_retained_heavy_coordinate_error'] = float(np.max(np.abs(pose.coords[0, indices].detach().numpy() - original.coord[heavy])))
np.testing.assert_array_equal(structure.coord, original.coord)
np.testing.assert_array_equal(structure.bonds.as_array(), original.bonds.as_array())
result['caller_input_unmodified'] = True
expected = 'rejected' if a.expect == 'rejected' else 'converted_to_different_chemistry'
assert result['outcome'] == expected, (result['outcome'], expected)
if a.output:
    a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
