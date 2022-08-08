import numpy
import torch
import pandas

from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.hbond.params import HBondParamResolver
from tmol.score.common.stack_condense import condense_numpy_inds
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.score.bonded_atom import IndexedBonds

from tmol.utility.cpp_extension import load, modulename, relpaths


def annotate_bt_w_intrares_indexed_bonds(bt):
    bonds = numpy.zeros((bt.bond_indices.shape[0], 3), dtype=numpy.int32)
    bonds[:, 1:] = bt.bond_indices.astype(numpy.int32)
    ib = IndexedBonds.from_bonds(bonds, minlength=bt.n_atoms)
    setattr(bt, "intrares_indexed_bonds", ib)


def test_create_intrares_indexed_bonds(
    default_database, fresh_default_restype_set, rts_ubq_res, torch_device
):
    p = PoseStackBuilder.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )
    pbt = p.packed_block_types

    for bt in pbt.active_block_types:
        annotate_bt_w_intrares_indexed_bonds(bt)


def test_annotate_block_type_hbond_params(
    default_database, fresh_default_restype_set, rts_ubq_res, torch_device
):
    compiled = load(
        modulename("tmol.score.hbond.identification"),
        relpaths("tmol/score/hbond/identification.py", "identification.pybind.cc"),
    )

    hbdb = default_database.scoring.hbond
    hb_params = HBondParamResolver.from_database(
        default_database.chemical, hbdb, torch_device
    )
    atom_resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch.device("cpu")
    )
    p = PoseStackBuilder.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )
    packed_block_types = p.packed_block_types

    is_acceptor = numpy.full(
        (packed_block_types.n_types, packed_block_types.max_n_atoms),
        0,
        dtype=numpy.int32,
    )
    acceptor_type = numpy.full(
        (packed_block_types.n_types, packed_block_types.max_n_atoms),
        -1,
        dtype=numpy.int32,
    )

    is_donor = numpy.full_like(is_acceptor, 0)
    donor_type = numpy.full_like(acceptor_type, -1)

    def map_names(atom_types, mapper, col_name, type_index):
        # print("mapper", mapper)
        is_hbtype = numpy.full(len(atom_types), 0, dtype=numpy.int32)
        hbtype = numpy.full(len(atom_types), 0, dtype=numpy.int32)
        hbtypes = numpy.full_like(atom_types, None, dtype=object)
        try:
            # if there are no atoms that register as acceptors/donors,
            # pandas will throw a KeyError (annoying!)
            hbtypes_df = mapper.loc[atom_types][col_name]
            hbtypes_df = hbtypes_df.where((pandas.notnull(hbtypes_df)), None)
            hbtypes[:] = numpy.array(hbtypes_df)
        except KeyError:
            pass
        hbtype = type_index.get_indexer(hbtypes)
        is_hbtype = hbtype != -1

        return is_hbtype, hbtype

    for i, block_type in enumerate(packed_block_types.active_block_types):
        annotate_bt_w_intrares_indexed_bonds(block_type)
        i_atom_types = [x.atom_type for x in block_type.atoms]
        atom_type_idx = atom_resolver.type_idx(i_atom_types)
        atom_type_params = atom_resolver.params[atom_type_idx]
        atom_acceptor_hybridization = (
            atom_type_params.acceptor_hybridization.numpy().astype(numpy.int64)[None, :]
        )

        i_is_acc, i_acc_type = map_names(
            i_atom_types,
            hbdb.acceptor_type_mapper,
            "acc_type",
            hb_params.acceptor_type_index,
        )
        i_slice = slice(packed_block_types.n_atoms[i])
        is_acceptor[i, i_slice] = i_is_acc
        acceptor_type[i, i_slice] = i_acc_type

        i_is_don, i_don_type = map_names(
            i_atom_types, hbdb.donor_type_mapper, "don_type", hb_params.donor_type_index
        )
        is_donor[i, i_slice] = i_is_don
        donor_type[i, i_slice] = i_don_type

        # print("i_is_acc")
        # print(i_is_acc)
        # print("i_acc_type")
        # print(i_acc_type)
        is_definitely_acceptor = i_is_acc & (i_acc_type >= 0)
        # print("is_definitely_acceptor")
        # print(is_definitely_acceptor)
        A_idx = condense_numpy_inds(is_definitely_acceptor[None, :])
        # print("A_idx")
        # print(A_idx)

        B_idx = numpy.full_like(A_idx, -1)
        B0_idx = numpy.full_like(A_idx, -1)
        atom_is_hydrogen = atom_type_params.is_hydrogen.numpy()[None, :]

        compiled.id_acceptor_bases(
            torch.from_numpy(A_idx),
            torch.from_numpy(B_idx),
            torch.from_numpy(B0_idx),
            torch.from_numpy(atom_acceptor_hybridization),
            torch.from_numpy(atom_is_hydrogen).to(torch.bool),
            block_type.intrares_indexed_bonds.bonds,
            block_type.intrares_indexed_bonds.bond_spans,
        )

        # print("A_idx")
        # print(A_idx)
        # print("B_idx")
        # print(B_idx)
        # print("B0_idx")
        # print(B0_idx)
        # print("atom_acceptor_hybridization")
        # print(atom_acceptor_hybridization)
        # print("atom_is_hydrogen")
        # print(atom_is_hydrogen)
