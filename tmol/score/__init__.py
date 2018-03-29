import properties
import functools

import numpy

import toolz

import scipy.sparse
import scipy.sparse.csgraph

import torch
import torch.autograd

from properties import List, Dictionary, StringChoice

import tmol.io.generic
import tmol.io.pdb_parsing as pdb_parsing

from .bonded_atom import BondedAtomScoreGraph
from .interatomic_distance import BlockedInteratomicDistanceGraph
from .ljlk import LJLKScoreGraph
from .types import RealTensor

@functools.singledispatch
def system_graph_params(system, drop_missing_atoms=False):
    bond_graph = system.bond_graph
    coords = torch.autograd.Variable(RealTensor(system.coords), requires_grad=True)
    atom_types = system.atom_types.copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan( system.coords ), axis=-1)] = None

    return dict(
        system_size=len(coords),
        bond_graph=bond_graph,
        coords=coords,
        atom_types=atom_types
    )


class ScoreGraph(
        LJLKScoreGraph,
        BlockedInteratomicDistanceGraph,
        BondedAtomScoreGraph,
    ):
    pass

@tmol.io.generic.to_pdb.register(BondedAtomScoreGraph)
def score_graph_to_pdb(score_graph):
    atom_coords = score_graph.coords.detach().numpy()
    atom_types = score_graph.atom_types

    render_atoms = numpy.flatnonzero(numpy.all(~numpy.isnan(atom_coords), axis=-1))

    atom_records = numpy.zeros_like(render_atoms, dtype=pdb_parsing.atom_record_dtype)

    atom_records['record_name'] = "ATOM"
    atom_records["chain"] = "X"
    atom_records["resn"] = "UNK"
    atom_records["atomi"] = render_atoms
    atom_records["atomn"] = score_graph.chemical_db.atom_properties.table.reindex(atom_types[render_atoms])["elem"].values

    atom_records["x"] = atom_coords[render_atoms][:,0]
    atom_records["y"] = atom_coords[render_atoms][:,1]
    atom_records["z"] = atom_coords[render_atoms][:,2]

    #atom_scores = score_graph.atom_scores.detach().numpy()
    #atom_records["b"] = atom_scores[render_atoms]

    return pdb_parsing.to_pdb(atom_records)

@tmol.io.generic.to_cdjson.register(BondedAtomScoreGraph)
def score_graph_to_cdjson(score_graph):
    coords = score_graph.coords.detach().numpy()
    elems = map(lambda t: t[0] if t else "x", score_graph.atom_types)
    bond_graph = score_graph.bond_graph.tocoo()
    bonds = zip(bond_graph.row, bond_graph.col)

    return tmol.io.generic.pack_cdjson(coords, elems, bonds)
