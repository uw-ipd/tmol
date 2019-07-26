import torch
import attr

from tmol.types.functional import convert_args
from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

import numpy

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring import HBondDatabase
from tmol.score.chemical_database import AtomTypeParamResolver
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

        # Filter donors/acceptors to those with type definitions in hbond
        # database, this logic should likely be moved to parameter resolution.
        atom_type_acceptor_type = {
            p.a: p.acceptor_type for p in hbond_database.acceptor_atom_types
        }
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        nstacks = atom_types.shape[0]

        atom_acceptor_type = numpy.array(
            [
                [atom_type_acceptor_type.get(at, None) for at in atom_types[stack]]
                for stack in range(nstacks)
            ]
        )
        atom_donor_type = numpy.array(
            [
                [atom_type_donor_type.get(at, None) for at in atom_types[stack]]
                for stack in range(nstacks)
            ]
        )
        # print("atom_types")
        # print(atom_types)
        # print("atom_donor_type")
        # print(atom_donor_type)

        # Get the acceptor indicies and allocate base idx buffers
        A_idx_list = [
            numpy.flatnonzero(atom_is_acceptor[i] & atom_acceptor_type[i].astype(bool))
            for i in range(nstacks)
        ]
        max_acceptors = max(len(acc_inds) for acc_inds in A_idx_list)

        # ?? not sure what dtype to use
        A_idx = numpy.full((nstacks, max_acceptors), -9999, dtype=numpy.int64)
        for i in range(nstacks):
            i_inds = A_idx_list[i]
            A_idx[i, : len(i_inds)] = i_inds

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
            numpy.logical_and(A_idx != -9999, B_idx == -1)
        ), "Invalid acceptor atom type."

        acceptors = numpy.full(A_idx.shape, -9999, dtype=acceptor_dtype)
        acceptors["a"] = A_idx
        acceptors["b"] = B_idx
        acceptors["b0"] = B0_idx
        for i in range(nstacks):
            n_real = numpy.sum(A_idx[i, :] >= 0)
            acceptors[i, :n_real]["acceptor_type"] = atom_acceptor_type[
                i, acceptors[i, :n_real]["a"]
            ]

        # Identify donor groups via donor-hydrogen bonds.
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        # create an array that can be indexed with, where if the first
        # atom is the same as the second atom, then it's not a real bond
        real_bonds = numpy.zeros_like(bonds.bonds)
        for i in range(nstacks):
            n_real = torch.sum(bonds.bonds[i,:,0] >= 0)
            real_bonds[i,:n_real,:] = bonds.bonds[i,:n_real,:]

        # torch.set_printoptions(threshold=5000)
        # numpy.set_printoptions(threshold=5000)
        # print("bonds.bonds")
        # print(bonds.bonds)
        #     
        # print("real_bonds")
        # print(real_bonds)
        #     
        # print("real_bonds?")
        # i = 0
        # print(real_bonds[i,:,0] != real_bonds[i,:,1])
        # 
        # print("atom_is_donor")
        # print(atom_is_donor[i, real_bonds[i, :, 0]])
        # 
        # print("atom_donor_type")
        # print(atom_donor_type[i].astype(bool)[real_bonds[i, :, 0]])
        # 
        # print("atom_is_hydrogen")
        # print(atom_is_hydrogen[i, real_bonds[i, :, 1]])
        # 
        # print("put it all together")
        # print(numpy.not_equal(real_bonds[i,:,0],real_bonds[i,:,1])
        #       & atom_is_donor[i, real_bonds[i, :, 0]]
        #       & atom_donor_type[i].astype(bool)[real_bonds[i, :, 0]]  # None -> False
        #       & atom_is_hydrogen[i, real_bonds[i, :, 1]])

        
        donor_pair_idx_list = [
            real_bonds[i,
                numpy.not_equal(real_bonds[i,:,0], real_bonds[i,:,1])
                & atom_is_donor[i, real_bonds[i, :, 0]]
                & atom_donor_type[i].astype(bool)[real_bonds[i, :, 0]]  # None -> False
                                & atom_is_hydrogen[i, real_bonds[i, :, 1]], :
            ]
            for i in range(nstacks)
        ]
        max_donors = max(len(don_inds) for don_inds in donor_pair_idx_list)

        # print("donor pair idx list")
        # print(donor_pair_idx_list[0])
        # 
        # print("atom is donor?")
        # print(atom_is_donor[0,donor_pair_idx_list[0][:,0]])
        
        donor_pair_idx = numpy.full((nstacks, max_donors, 2), -9999, dtype=int)
        for i in range(nstacks):
            i_inds = donor_pair_idx_list[i]
            donor_pair_idx[i, : len(i_inds)] = i_inds

        donors = numpy.empty(
            (donor_pair_idx.shape[0], donor_pair_idx.shape[1]), dtype=donor_dtype
        )
        donors["d"] = donor_pair_idx[:, :, 0]
        donors["h"] = donor_pair_idx[:, :, 1]

        # gah, this is a gather, right? but there's no good way to gather
        # with numpy? and also, I need to say "this is an invalid index"
        # for that gather.
        for i in range(nstacks):
            n_real = numpy.sum(donor_pair_idx[i, :, 0] >= 0)
            # print("donor_pair_idx")
            # print(donor_pair_idx[i])
            # print("donor_pair_idx real")
            # print(donor_pair_idx[i,:,0]  >= 0)
            # print("sum")
            # print(n_real)
            # print("donor atom types")
            # print(atom_types[i, donor_pair_idx[i, :n_real, 0]])
            # print("donor types shape", atom_donor_type[
            #     i, donor_pair_idx[i, :n_real, 0]
            # ])
            donors[i, :n_real]["donor_type"] = atom_donor_type[
                i, donor_pair_idx[i, :n_real, 0]
            ]

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
