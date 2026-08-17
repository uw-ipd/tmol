import pytest
import torch

from tmol.pose import PackedBlockTypes
from tmol.pose import PoseStackBuilder

from tmol.io import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)

from tmol.io import pose_stack_from_pdb
from tmol.io import pose_stack_from_canonical_form

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_constraint_set_empty_initialization": (
        "test_constraint_set",
        "test_constraint_set_empty_initialization",
    ),
    "test_constraint_set_add_constraints": (
        "test_constraint_set",
        "test_constraint_set_add_constraints",
    ),
    "test_constraint_set_concatenate_constraints": (
        "test_constraint_set",
        "test_constraint_set_concatenate_constraints",
    ),
    "test_constraint_set_concatenate_constraints_2": (
        "test_constraint_set",
        "test_constraint_set_concatenate_constraints_2",
    ),
    "test_split_constraint_set": ("test_constraint_set", "test_split_constraint_set"),
    "test_load_packed_residue_types": (
        "test_packed_block_types",
        "test_load_packed_residue_types",
    ),
    "test_determine_real_atoms": (
        "test_packed_block_types",
        "test_determine_real_atoms",
    ),
    "test_packed_residue_type_atoms_downstream_of_conn": (
        "test_packed_block_types",
        "test_packed_residue_type_atoms_downstream_of_conn",
    ),
    "test_packed_block_types_ordered_torsions": (
        "test_packed_block_types",
        "test_packed_block_types_ordered_torsions",
    ),
    "test_packed_block_types_device": (
        "test_packed_block_types",
        "test_packed_block_types_device",
    ),
    "test_pdb_info_split": ("test_pdb_info", "test_pdb_info_split"),
    "test_n_poses": ("test_pose_stack", "test_n_poses"),
    "test_max_n_blocks": ("test_pose_stack", "test_max_n_blocks"),
    "test_max_n_atoms": ("test_pose_stack", "test_max_n_atoms"),
    "test_max_n_block_atoms": ("test_pose_stack", "test_max_n_block_atoms"),
    "test_max_n_pose_atoms": ("test_pose_stack", "test_max_n_pose_atoms"),
    "test_n_ats_per_pose_block": ("test_pose_stack", "test_n_ats_per_pose_block"),
    "test_real_atoms": ("test_pose_stack", "test_real_atoms"),
    "test_expand_coords": ("test_pose_stack", "test_expand_coords"),
    "test_round_trip_irregular_pose_stack_and_split": (
        "test_pose_stack",
        "test_round_trip_irregular_pose_stack_and_split",
    ),
    "test_concatenate_pose_stacks_ctor": (
        "test_pose_stack_construction",
        "test_concatenate_pose_stacks_ctor",
    ),
    "test_create_pose_from_sequence": (
        "test_pose_stack_construction",
        "test_create_pose_from_sequence",
    ),
    "test_pose_stack_builder_find_inter_block_sep_for_polymeric_monomers_lcaa": (
        "test_pose_stack_construction",
        "test_pose_stack_builder_find_inter_block_sep_for_polymeric_monomers_lcaa",
    ),
    "test_pose_stack_builder_inter_block_sep_mix_alpha_and_beta": (
        "test_pose_stack_construction",
        "test_pose_stack_builder_inter_block_sep_mix_alpha_and_beta",
    ),
    "test_take_real_conn_conn_intrablock_pairs_heavy": (
        "test_pose_stack_construction",
        "test_take_real_conn_conn_intrablock_pairs_heavy",
    ),
    "test_find_connection_pairs_for_residue_subset": (
        "test_pose_stack_construction",
        "test_find_connection_pairs_for_residue_subset",
    ),
    "test_find_connection_pairs_for_residue_subset2": (
        "test_pose_stack_construction",
        "test_find_connection_pairs_for_residue_subset2",
    ),
    "test_find_connections_in_sequences": (
        "test_pose_stack_construction",
        "test_find_connections_in_sequences",
    ),
    "test_find_connection_pairs_for_residue_subset_w_errors1": (
        "test_pose_stack_construction",
        "test_find_connection_pairs_for_residue_subset_w_errors1",
    ),
    "test_find_connection_pairs_for_residue_subset_w_errors2": (
        "test_pose_stack_construction",
        "test_find_connection_pairs_for_residue_subset_w_errors2",
    ),
    "test_calculate_interblock_bondsep_from_connectivity_graph_heavy": (
        "test_pose_stack_construction",
        "test_calculate_interblock_bondsep_from_connectivity_graph_heavy",
    ),
    "test_incorporate_extra_connections_into_inter_res_conn_set": (
        "test_pose_stack_construction",
        "test_incorporate_extra_connections_into_inter_res_conn_set",
    ),
    "test_incorporate_extra_connections_into_inter_res_conn_set2": (
        "test_pose_stack_construction",
        "test_incorporate_extra_connections_into_inter_res_conn_set2",
    ),
    "test_incorporate_inter_residue_connections_into_connectivity_graph": (
        "test_pose_stack_construction",
        "test_incorporate_inter_residue_connections_into_connectivity_graph",
    ),
    "test_construct_pose_stack_containing_disulfides_smoke": (
        "test_pose_stack_construction",
        "test_construct_pose_stack_containing_disulfides_smoke",
    ),
    "interblock_dslf_self_correction": (
        "test_pose_stack_construction",
        "interblock_dslf_self_correction",
    ),
    "interblock_dslf_pair_correction": (
        "test_pose_stack_construction",
        "interblock_dslf_pair_correction",
    ),
    "test_from_block_type_names_smoke": (
        "test_pose_stack_construction",
        "test_from_block_type_names_smoke",
    ),
    "test_pose_construction_from_sequence": (
        "test_pose_stack_construction_benchmark",
        "test_pose_construction_from_sequence",
    ),
    "LIGAND_SMILES": ("test_util", "LIGAND_SMILES"),
    "block_names": ("test_util", "block_names"),
    "real_coords": ("test_util", "real_coords"),
    "min_interatomic_distance": ("test_util", "min_interatomic_distance"),
    "assert_backbone_is_ideal": ("test_util", "assert_backbone_is_ideal"),
    "test_extended_pose_stack_protein_only": (
        "test_util",
        "test_extended_pose_stack_protein_only",
    ),
    "test_extended_pose_stack_dna_only": (
        "test_util",
        "test_extended_pose_stack_dna_only",
    ),
    "test_extended_pose_stack_ligand_only": (
        "test_util",
        "test_extended_pose_stack_ligand_only",
    ),
    "test_extended_pose_stack_protein_dna_and_ligand": (
        "test_util",
        "test_extended_pose_stack_protein_dna_and_ligand",
    ),
    "test_extended_pose_stack_repeated_protein": (
        "test_util",
        "test_extended_pose_stack_repeated_protein",
    ),
    "test_set_named_torsions_roundtrip": (
        "test_util",
        "test_set_named_torsions_roundtrip",
    ),
    "test_set_named_torsions_batch_roundtrip": (
        "test_util",
        "test_set_named_torsions_batch_roundtrip",
    ),
    "test_set_named_torsions_radians": ("test_util", "test_set_named_torsions_radians"),
    "test_set_named_torsions_absent_torsion_raises": (
        "test_util",
        "test_set_named_torsions_absent_torsion_raises",
    ),
    "test_set_named_torsions_undefined_torsion_raises": (
        "test_util",
        "test_set_named_torsions_undefined_torsion_raises",
    ),
    "c_to_n_fold_forest": ("test_util", "c_to_n_fold_forest"),
    "test_named_torsions_agree_across_fold_forests": (
        "test_util",
        "test_named_torsions_agree_across_fold_forests",
    ),
    "PROTEIN_GOLD_TORSION_NAMES": ("test_util", "PROTEIN_GOLD_TORSION_NAMES"),
    "DNA_GOLD_TORSION_NAMES": ("test_util", "DNA_GOLD_TORSION_NAMES"),
    "RNA_GOLD_TORSION_NAMES": ("test_util", "RNA_GOLD_TORSION_NAMES"),
    "test_get_torsion_names": ("test_util", "test_get_torsion_names"),
    "test_get_torsion_names_non_polymer": (
        "test_util",
        "test_get_torsion_names_non_polymer",
    ),
    "test_get_torsion_names_rejects_absent_block": (
        "test_util",
        "test_get_torsion_names_rejects_absent_block",
    ),
    "test_extended_pose_stack_device": ("test_util", "test_extended_pose_stack_device"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@pytest.fixture
def ubq_40_60_pose_stack(ubq_pdb, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=40)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=60)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    return poses


@pytest.fixture
def fresh_default_packed_block_types(fresh_default_restype_set, torch_device):
    return PackedBlockTypes.from_restype_list(
        fresh_default_restype_set.chem_db,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )


@pytest.fixture
def stack_of_two_six_res_ubqs(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=6
    )

    pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    return PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)


@pytest.fixture
def stack_of_two_six_res_ubqs_no_term(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=1, residue_end=7
    )

    canonical_form.res_not_connected = torch.zeros(
        (1, 6, 2), dtype=torch.bool, device=torch_device
    )
    canonical_form.res_not_connected[0, 0, 0] = True  # simplest test case: not N-term
    canonical_form.res_not_connected[0, 5, 1] = True  # simplest test case: not C-term
    pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    return PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)


@pytest.fixture
def distinct_pose_stacks(systems_bysize, torch_device):
    """Single-pose PoseStacks for three unrelated proteins (40, 75, 150 res)."""
    return [
        pose_stack_from_pdb(systems_bysize[nres], torch_device)
        for nres in (40, 75, 150)
    ]


@pytest.fixture
def stack_of_distinct_poses(distinct_pose_stacks, torch_device):
    """Jagged PoseStack of the three unrelated proteins in distinct_pose_stacks."""
    return PoseStackBuilder.from_poses(distinct_pose_stacks, torch_device)


@pytest.fixture
def jagged_stack_of_465_res_ubqs(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)

    def pose_stack_of_nres(nres):
        canonical_form = canonical_form_from_pdb(
            co, ubq_pdb, torch_device, residue_start=0, residue_end=nres
        )
        return pose_stack_from_canonical_form(co, pbt, *canonical_form)

    return PoseStackBuilder.from_poses(
        [pose_stack_of_nres(x) for x in [4, 6, 5]], torch_device
    )
