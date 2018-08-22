import numpy

import tmol.io.generic
import tmol.io.pdb_parsing as pdb_parsing

from . import bonded_atom


@tmol.io.generic.to_pdb.register(bonded_atom.BondedAtomScoreGraph)
def score_graph_to_pdb(score_graph):
    if score_graph.stack_depth != 1:
        raise NotImplementedError("Can not convert stack_depth != 1 to pdb.")

    atom_coords = score_graph.coords[0].detach().numpy()
    atom_types = score_graph.atom_types[0]

    render_atoms = numpy.flatnonzero(numpy.all(~numpy.isnan(atom_coords), axis=-1))

    atom_records = numpy.zeros_like(render_atoms, dtype=pdb_parsing.atom_record_dtype)

    atom_records["record_name"] = "ATOM"
    atom_records["chain"] = "X"
    atom_records["resn"] = "UNK"
    atom_records["atomi"] = render_atoms
    atom_records["atomn"] = [t[0] for t in atom_types[render_atoms]]

    atom_records["x"] = atom_coords[render_atoms][:, 0]
    atom_records["y"] = atom_coords[render_atoms][:, 1]
    atom_records["z"] = atom_coords[render_atoms][:, 2]

    atom_records["b"] = 0

    return pdb_parsing.to_pdb(atom_records)


@tmol.io.generic.to_cdjson.register(bonded_atom.BondedAtomScoreGraph)
def score_graph_to_cdjson(score_graph):
    if score_graph.stack_depth != 1:
        raise NotImplementedError("Can not convert stack_depth != 1 to cdjson.")

    coords = score_graph.coords[0].detach().numpy()
    elems = map(lambda t: t[0] if t else "x", score_graph.atom_types[0])
    bonds = list(map(tuple, score_graph.bonds[:, 1:]))

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)
