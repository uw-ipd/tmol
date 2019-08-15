import torch
import attr

from tmol.types.functional import convert_args
from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

import numpy
import pandas

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring import HBondDatabase
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.common.stack_condense import condense_numpy_inds
from ..bonded_atom import IndexedBonds

from tmol.utility.cpp_extension import load, modulename, relpaths

acceptor_dtype = numpy.dtype(
    [("a", int), ("b", int), ("b0", int), ("acceptor_type", object)]
)

donor_dtype = numpy.dtype([("d", int), ("h", int), ("donor_type", object)])


@attr.s(frozen=True, slots=True, auto_attribs=True)
class HBondElementAnalysis(ValidateAttrs):
    donors: NDArray(donor_dtype)[:, :]
    acceptors: NDArray(acceptor_dtype)[:, :]

    @classmethod
    @convert_args
    def setup(
        cls,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:, :],
        atom_is_acceptor: NDArray(bool)[:, :],
        atom_acceptor_hybridization: NDArray(int)[:, :],
        atom_is_donor: NDArray(bool)[:, :],
        atom_is_hydrogen: NDArray(bool)[:, :],
        bonds: NDArray(int)[:, 3],
    ):
        compiled = load(
            modulename(__name__), relpaths(__file__, "identification.pybind.cc")
        )

        # Only access bonds through index.
        bonds = IndexedBonds.from_bonds(bonds, minlength=atom_types.shape[1])

        nstacks = atom_types.shape[0]
        real = atom_types.astype(bool)

        def map_names(mapper, col_name):
            hbtypes = numpy.full_like(atom_types, None, dtype=object)
            try:
                # if there are no atoms that register as acceptors/donors,
                # pandas will throw a KeyError (annoying!)
                hbtypes_df = mapper.loc[atom_types[real].ravel()][col_name]
                hbtypes_df = hbtypes_df.where((pandas.notnull(hbtypes_df)), None)
                hbtypes[real] = numpy.array(hbtypes_df)
            except KeyError:
                pass
            return hbtypes

        atom_acceptor_type = map_names(hbond_database.acceptor_type_mapper, "acc_type")
        atom_donor_type = map_names(hbond_database.donor_type_mapper, "don_type")

        # Get the acceptor indicies and allocate base idx buffers
        is_definitely_acceptor = atom_is_acceptor & atom_acceptor_type.astype(bool)
        A_idx = condense_numpy_inds(is_definitely_acceptor)
        B_idx = numpy.full_like(A_idx, -1)
        B0_idx = numpy.full_like(A_idx, -1)

        # Yeeeehaw
        compiled.id_acceptor_bases(
            torch.from_numpy(A_idx),
            torch.from_numpy(B_idx),
            torch.from_numpy(B0_idx),
            torch.from_numpy(atom_acceptor_hybridization),
            torch.from_numpy(atom_is_hydrogen.astype(numpy.ubyte)),
            bonds.bonds,
            bonds.bond_spans,
        )

        assert not numpy.any(
            numpy.logical_and(A_idx != -1, B_idx == -1)
        ), "Invalid acceptor atom type."

        acceptors = numpy.full(A_idx.shape, -9999, dtype=acceptor_dtype)
        acceptors["a"] = A_idx
        acceptors["b"] = B_idx
        acceptors["b0"] = B0_idx
        real_acceptors = A_idx >= 0
        nz = numpy.nonzero(is_definitely_acceptor)
        acceptors["acceptor_type"][real_acceptors] = atom_acceptor_type[nz[0], nz[1]]
        acceptors["acceptor_type"][numpy.invert(real_acceptors)] = None

        # Identify donor groups via donor-hydrogen bonds.
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        # create an array that can be indexed with, where if the first
        # atom is the same as the second atom, then it's not a real bond
        real_bonds = numpy.zeros_like(bonds.bonds)
        for i in range(nstacks):
            n_real = torch.sum(bonds.bonds[i, :, 0] >= 0)
            real_bonds[i, :n_real, :] = bonds.bonds[i, :n_real, :]

        rb_stack = (
            numpy.arange(real_bonds.shape[0] * real_bonds.shape[1])
            / real_bonds.shape[1]
        )
        rb_stack = rb_stack.astype(int)

        rb01 = real_bonds.shape[0:2]

        bond_bw_don_and_h = (
            numpy.not_equal(real_bonds[:, :, 0], real_bonds[:, :, 1])
            & atom_is_donor[rb_stack, real_bonds[:, :, 0].ravel()].reshape(rb01)
            & atom_donor_type[rb_stack, real_bonds[:, :, 0].ravel()]
            .reshape(rb01)
            .astype(bool)
            & atom_is_hydrogen[rb_stack, real_bonds[:, :, 1].ravel()].reshape(rb01)
        )

        nkeep = numpy.sum(bond_bw_don_and_h, axis=1).reshape((nstacks, 1))
        max_donors = numpy.max(nkeep)
        counts = numpy.arange(max_donors, dtype=int).reshape((1, max_donors))
        lowinds = counts < nkeep
        donor_pair_idx = numpy.full((nstacks, max_donors, 2), -9999, dtype=int)
        donor_pair_idx[:, :, 0][lowinds] = real_bonds[:, :, 0][bond_bw_don_and_h]
        donor_pair_idx[:, :, 1][lowinds] = real_bonds[:, :, 1][bond_bw_don_and_h]

        donors = numpy.empty(
            (donor_pair_idx.shape[0], donor_pair_idx.shape[1]), dtype=donor_dtype
        )
        donors["d"] = donor_pair_idx[:, :, 0]
        donors["h"] = donor_pair_idx[:, :, 1]
        nz = numpy.nonzero(bond_bw_don_and_h)
        donors["donor_type"][lowinds] = atom_donor_type[
            nz[0], real_bonds[nz[0], nz[1], 0]
        ]
        donors["donor_type"][numpy.invert(lowinds)] = None

        return cls(donors=donors, acceptors=acceptors)

    @classmethod
    def setup_from_database(
        cls,
        chemical_database: ChemicalDatabase,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:],
        bonds: NDArray(int)[:, 3],
    ):

        atom_resolver = AtomTypeParamResolver.from_database(
            chemical_database, torch.device("cpu")
        )
        atom_type_idx = atom_resolver.type_idx(atom_types)
        atom_type_params = atom_resolver.params[atom_type_idx]

        return cls.setup(
            hbond_database=hbond_database,
            atom_types=atom_types,
            atom_is_acceptor=atom_type_params.is_acceptor.numpy(),
            atom_acceptor_hybridization=atom_type_params.acceptor_hybridization.numpy(),
            atom_is_donor=atom_type_params.is_donor.numpy(),
            atom_is_hydrogen=atom_type_params.is_hydrogen.numpy(),
            bonds=bonds,
        )
