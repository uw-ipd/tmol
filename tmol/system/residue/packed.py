import properties as prop

import attr
import numpy
import scipy

from tmol.properties.array import Array
from tmol.properties.reactive import derived_from
from tmol.properties import eq_by_is

@attr.s(slots=True, frozen=True)
class Residue:
    residue_type = attr.ib()
    coords = attr.ib()
    @coords.default
    def _coord_buffer(self):
        return numpy.full((len(self.residue_type.atoms), 3), numpy.nan, dtype=float)
    
    @property
    def atom_coords(self):
        return self.coords.reshape(-1).view(self.residue_type.coord_dtype)
    
    def attach_to(self, coord_buffer):
        assert coord_buffer.shape == self.coords.shape
        assert coord_buffer.dtype == self.coords.dtype
        
        coord_buffer[:] = self.coords
        
        return attr.evolve(self, coords = coord_buffer)
    
    def _repr_pretty_(self, p, cycle):
        p.text("Residue")
        with p.group(1,"(", ")"):
            p.text("type=")
            p.pretty(self.residue_type)
            p.text(", coords=")
            p.break_()
            p.pretty(self.coords)

class PackedResidueSystem(prop.HasProperties):
    
    block_size = prop.Integer(
        "coord buffer block size, residue start indicies are aligned to block boundries",
        default=8, min=1, cast=True
    )
    
    residues = prop.List(
        "residue objects packed into system",
        prop=prop.Instance("attached residue", Residue)
    )
    
    start_ind = Array(
        "residue start indicies within `coords`",
        dtype=int)[:]

    system_size = prop.Integer("total system size")

    coords = Array(
        "atomic coordinate buffer, nan-filled in 'unused' regions",
        dtype=float, cast="unsafe")[:,3]

    bond_graph = eq_by_is(prop.Instance(
        "Inter-atomic bonds",
        scipy.sparse.spmatrix)
    )

    @derived_from(
        ("residues", "start_ind", "system_size"),
        Array("atom type tags", dtype=object)[:]
    )
    def atom_types(self):
        types = numpy.full(self.system_size, None, dtype=object)

        for si, res in zip(self.start_ind, self.residues):
            types[si:si+len(res.residue_type.atoms)] = [
                a.atom_type for a in res.residue_type.atoms
            ]

        return types

    @staticmethod
    def _ceil_to_size(size, val):
        d, m = numpy.divmod(val, size)
        return (d + (m != 0).astype(int)) * size
    
    def from_residues(self, res):
        """Initialize a packed residue system from list of residue containers."""

        res_lengths = numpy.array([len(r.coords) for r in res])
        res_segment_lengths = self._ceil_to_size(self.block_size, res_lengths)

        segment_ends = res_segment_lengths.cumsum()
        buffer_size = segment_ends[-1]

        segment_starts = numpy.empty_like(segment_ends)
        segment_starts[0] = 0
        segment_starts[1:] = segment_ends[:-1]

        cbuff = numpy.full((buffer_size, 3), numpy.nan)

        attached_res = [
            r.attach_to(cbuff[start:start+len(r.coords)])
            for r, start in zip(res, segment_starts)
        ]

        res_by_start = list(zip(segment_starts, res))

        intra_res_bonds = numpy.concatenate([
            r.residue_type.bond_indicies + start
            for start, r in res_by_start
        ])

        upchain_inter_res_bonds = numpy.array([
            [i.residue_type.upper_connect_idx + si, j.residue_type.lower_connect_idx + sj]
            for (si, i), (sj, j) in zip(res_by_start[:-1], res_by_start[1:])
        ])

        downchain_inter_res_bonds = numpy.flip(upchain_inter_res_bonds, axis=-1)

        bonds = numpy.concatenate([intra_res_bonds, upchain_inter_res_bonds, downchain_inter_res_bonds])

        self.residues = attached_res
        self.start_ind = segment_starts
        self.system_size = buffer_size

        self.coords = cbuff
        self.bond_graph = scipy.sparse.coo_matrix((
                numpy.ones(len(bonds), dtype=bool),
                (bonds[:,0], bonds[:,1])
            ),
            shape=(len(cbuff), len(cbuff))
        )

        self.validate()

        return self
