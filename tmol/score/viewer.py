import numpy

import tmol.io.generic
import tmol.io.pdb_parsing as pdb_parsing

from . import bonded_atom


@tmol.io.generic.to_pdb.register(bonded_atom.BondedAtomScoreGraph)
def score_graph_to_pdb(score_graph):
    atom_coords = score_graph.coords.detach().numpy()
    atom_types = score_graph.atom_types

    render_atoms = numpy.flatnonzero(
        numpy.all(~numpy.isnan(atom_coords), axis=-1)
    )

    atom_records = numpy.zeros_like(
        render_atoms, dtype=pdb_parsing.atom_record_dtype
    )

    atom_records['record_name'] = "ATOM"
    atom_records["chain"] = "X"
    atom_records["resn"] = "UNK"
    atom_records["atomi"] = render_atoms
    atom_records["atomn"] = (
        score_graph.chemical_db.atom_properties.table.reindex(
            atom_types[render_atoms]
        )["elem"].values
    )

    atom_records["x"] = atom_coords[render_atoms][:, 0]
    atom_records["y"] = atom_coords[render_atoms][:, 1]
    atom_records["z"] = atom_coords[render_atoms][:, 2]

    atom_records["b"] = 0

    return pdb_parsing.to_pdb(atom_records)


@tmol.io.generic.to_cdjson.register(bonded_atom.BondedAtomScoreGraph)
def score_graph_to_cdjson(score_graph):
    coords = score_graph.coords.detach().numpy()
    elems = map(lambda t: t[0] if t else "x", score_graph.atom_types)
    bonds = list(map(tuple, score_graph.bonds))

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)
