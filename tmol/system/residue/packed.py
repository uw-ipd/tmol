from properties import HasProperties, List, Integer, Instance
from tmol.properties.array import Array

import numpy

from typing import Sequence

from .restypes import Residue


class PackedResidueSystem(HasProperties):

    block_size: int = Integer(
        "coord buffer block size, residue start indicies are aligned to block boundries",
        default=8,
        min=1,
        cast=True
    )

    residues: Sequence[Residue] = List(
        "residue objects packed into system",
        prop=Instance("attached residue", Residue)
    )

    res_start_ind: Sequence[int] = Array(
        "residue start indicies within `coords`", dtype=int
    )[:]

    system_size: int = Integer("total system size")

    coords: numpy.ndarray = Array(
        "atomic coordinate buffer, nan-filled in 'unused' regions",
        dtype=float,
        cast="unsafe"
    )[:, 3]

    bonds: numpy.ndarray = Array(
        "inter-atomic bond indices", dtype=int, cast="unsafe"
    )[:, 2]

    atom_metadata_dtype = numpy.dtype([
        ("residue_name", object),
        ("atom_name", object),
        ("atom_type", object),
        ("atom_index", object),
        ("residue_index", float),
    ])

    atom_metadata: numpy.ndarray = Array(
        "atom metada", dtype=atom_metadata_dtype
    )[:]

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
            r.attach_to(cbuff[start:start + len(r.coords)])
            for r, start in zip(res, segment_starts)
        ]

        res_by_start = list(zip(segment_starts, res))

        intra_res_bonds = numpy.concatenate([
            r.residue_type.bond_indicies + start for start, r in res_by_start
        ])

        upchain_inter_res_bonds = numpy.array([
            [
                i.residue_type.upper_connect_idx + si,
                j.residue_type.lower_connect_idx + sj
            ]
            for (si, i), (sj, j) in zip(res_by_start[:-1], res_by_start[1:])
        ])  # yapf: disable

        downchain_inter_res_bonds = numpy.flip(
            upchain_inter_res_bonds, axis=-1
        )

        bonds = numpy.concatenate([
            intra_res_bonds, upchain_inter_res_bonds, downchain_inter_res_bonds
        ])

        self.residues = attached_res
        self.res_start_ind = segment_starts
        self.system_size = buffer_size

        self.coords = cbuff
        self.bonds = bonds

        self.atom_metadata = numpy.empty(
            self.system_size, self.atom_metadata_dtype
        )
        self.atom_metadata["atom_index"] = numpy.arange(
            len(self.atom_metadata)
        )
        self.atom_metadata["residue_index"] = None

        for ri, (rs, r) in enumerate(zip(self.res_start_ind, self.residues)):
            rt = r.residue_type
            residue_block = self.atom_metadata[rs:rs + len(rt.atoms)]
            residue_block["residue_name"] = rt.name
            residue_block["atom_name"] = [a.name for a in rt.atoms]
            residue_block["atom_type"] = [a.atom_type for a in rt.atoms]
            residue_block["residue_index"] = ri

        self.atom_metadata.flags.writeable = False

        self.validate()

        return self
