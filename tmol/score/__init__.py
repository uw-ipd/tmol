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

from tmol.properties.reactive import derived_from
from tmol.properties.array import Array, VariableT, TensorT
from tmol.properties import eq_by_is

from .ljlk import LJLKScoreGraph

@functools.singledispatch
def system_graph_params(system, drop_missing_atoms=False):
    bond_graph = system.bond_graph
    coords = torch.autograd.Variable(torch.Tensor(system.coords), requires_grad=True)
    atom_types = system.atom_types.copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan( system.coords ), axis=-1)] = None

    return dict(bond_graph = bond_graph, coords=coords, atom_types=atom_types)

class ScoreGraph(LJLKScoreGraph, properties.HasProperties):
    bond_graph = eq_by_is(
            properties.Instance(
                "inter-atomic bond graph",
                instance_class=scipy.sparse.spmatrix))
    coords = VariableT("source atomic coordinates")
    atom_types = Array("atomic types", dtype=object)[:]

    @staticmethod
    def nan_to_num(var):
        return torch.where(
            ~numpy.isnan(var.detach()),
            var,
            torch.Tensor([0.0]))

    @derived_from("atom_types", 
            Array("mask of 'real' atom indicies", dtype=bool)[:])
    def real_atoms(self):
        return (self.atom_types != None)

    @derived_from("bond_graph", Array("inter-atomic minimum bonded path length", dtype="f4")[:,:])
    def bonded_path_length(self):
        return scipy.sparse.csgraph.shortest_path(
            self.bond_graph,
            directed=False,
            unweighted=True
        ).astype("f4")

    @derived_from("coords", VariableT("inter-atomic pairwise distance"))
    def dist(self):
        coords = self.nan_to_num(self.coords)
        deltas = coords.view((-1, 1, 3)) - coords.view((1, -1, 3))
        dist = torch.norm(deltas, 2, -1)
        dist.register_hook(self.nan_to_num)
        return dist

    @derived_from(("lj", "lk"), TensorT("inter-atomic total score"))
    def atom_scores(self):
        return torch.sum((self.lj + self.lk).data, dim=-1)

    @derived_from(("lj", "lk"), VariableT("system total score"))
    def total_score(self):
        return torch.sum(self.lj + self.lk)

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(dict(name="coords", prev=getattr(self, "coords"), mode="observe_set"))

        self.total_score.backward()
        return self.total_score


@tmol.io.generic.to_pdb.register(ScoreGraph)
def score_graph_to_pdb(score_graph):
    atom_coords = score_graph.coords.detach().numpy()
    atom_types = score_graph.atom_types
    atom_scores = score_graph.atom_scores.detach().numpy()

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

    atom_records["b"] = atom_scores[render_atoms]

    return pdb_parsing.to_pdb(atom_records)
