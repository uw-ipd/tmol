import torch

# import cattr
# import attr

from tmol.database import ParameterDatabase
from tmol.database.scoring.elec import CountPairReps, PartialCharges, ElecDatabase
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from tmol.score.elec.params import ElecParamResolver


def test_construct_elec_param_resolver_smoke(default_database, torch_device):
    params = ElecParamResolver.from_database(
        default_database.scoring.elec, torch_device
    )
    assert params is not None


def test_elec_param_resolver_w_missing_cp_rep_exception_handling(
    default_database, fresh_default_restype_set, torch_device
):
    orig_elec_db = default_database.scoring.elec
    # let's just leave out a residue type??
    left_out_rt_name = orig_elec_db.atom_cp_reps_parameters[0].res
    bad_cp_reps_parameters = tuple(
        x
        for x in orig_elec_db.atom_cp_reps_parameters
        if x.res.partition(":")[0] != left_out_rt_name
    )
    bad_elec_db = ElecDatabase(
        global_parameters=orig_elec_db.global_parameters,
        atom_cp_reps_parameters=bad_cp_reps_parameters,
        atom_charge_parameters=orig_elec_db.atom_charge_parameters,
    )
    bad_params = ElecParamResolver.from_database(bad_elec_db, torch_device)

    left_out_rt = next(
        x for x in fresh_default_restype_set.residue_types if x.name == left_out_rt_name
    )
    try:
        bad_params.get_bonded_path_length_mapping_for_block(left_out_rt)
        assert False
    except KeyError as err:
        assert (
            str(err)
            == "'No elec count-pair representative definition for base name "
            + left_out_rt_name
            + "'"
        )


def test_elec_param_resolver_w_bad_cp_rep_exception_handling(
    default_database, fresh_default_restype_set, torch_device
):
    orig_elec_db = default_database.scoring.elec

    first_rt_name = orig_elec_db.atom_cp_reps_parameters[0].res
    bad_cp_rep = CountPairReps(res=first_rt_name, atm_inner="XX", atm_outer="C")

    # add a cp rep with something nonsensical
    # assumption: that the left out residue type
    # does not contain an atom named "XX"
    bad_cp_reps_parameters = (bad_cp_rep,) + orig_elec_db.atom_cp_reps_parameters

    bad_elec_db = ElecDatabase(
        global_parameters=orig_elec_db.global_parameters,
        atom_cp_reps_parameters=bad_cp_reps_parameters,
        atom_charge_parameters=orig_elec_db.atom_charge_parameters,
    )
    bad_params = ElecParamResolver.from_database(bad_elec_db, torch_device)

    first_rt = next(
        x for x in fresh_default_restype_set.residue_types if x.name == first_rt_name
    )
    assert "XX" not in first_rt.atom_to_idx
    try:
        bad_params.get_bonded_path_length_mapping_for_block(first_rt)
        assert False
    except KeyError as err:
        assert str(err) == "'Invalid elec cp mapping: " + first_rt_name + " C->XX'"


def test_elec_param_resolver_w_missing_partial_charge_exception_handling(
    default_database, fresh_default_restype_set, torch_device
):
    orig_elec_db = default_database.scoring.elec
    # let's just leave out a single partial charge
    first_charge = orig_elec_db.atom_charge_parameters[0]
    first_rt_name = first_charge.res
    first_atom_name = first_charge.atom
    bad_charge_parameters = orig_elec_db.atom_charge_parameters[1:]

    bad_elec_db = ElecDatabase(
        global_parameters=orig_elec_db.global_parameters,
        atom_cp_reps_parameters=orig_elec_db.atom_cp_reps_parameters,
        atom_charge_parameters=bad_charge_parameters,
    )
    bad_params = ElecParamResolver.from_database(bad_elec_db, torch_device)

    first_rt = next(
        x for x in fresh_default_restype_set.residue_types if x.name == first_rt_name
    )
    try:
        bad_params.get_partial_charges_for_block(first_rt)
        assert False
    except KeyError as err:
        assert (
            str(err)
            == "'Elec charge for atom "
            + first_rt_name
            + ","
            + first_atom_name
            + " not found'"
        )
