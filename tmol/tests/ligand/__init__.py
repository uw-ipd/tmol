_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "PLI_DIR": ("build_pli_ligand_cifs", "PLI_DIR"),
    "CIF_OUT_DIR": ("build_pli_ligand_cifs", "CIF_OUT_DIR"),
    "ensure_pli_ligand_cifs": ("build_pli_ligand_cifs", "ensure_pli_ligand_cifs"),
    "main": ("build_pli_ligand_cifs", "main"),
    "EquivalenceResult": ("equivalence", "EquivalenceResult"),
    "compare_ligand_preparations": ("equivalence", "compare_ligand_preparations"),
    "FixtureMismatch": ("fixture_integrity", "FixtureMismatch"),
    "Mol2Summary": ("fixture_integrity", "Mol2Summary"),
    "read_mol2_summary": ("fixture_integrity", "read_mol2_summary"),
    "require_paired_fixture": ("fixture_integrity", "require_paired_fixture"),
    "ReferenceParams": ("params_reference", "ReferenceParams"),
    "parse_reference_params": ("params_reference", "parse_reference_params"),
    "as_legacy_dict": ("params_reference", "as_legacy_dict"),
    "reference_charges": ("params_reference", "reference_charges"),
    "ChargeComparison": ("params_reference", "ChargeComparison"),
    "compare_charges": ("params_reference", "compare_charges"),
    "reference_bond_keys": ("params_reference", "reference_bond_keys"),
    "GeneratedFields": ("params_reference", "GeneratedFields"),
    "generated_fields_from_preparation": (
        "params_reference",
        "generated_fields_from_preparation",
    ),
    "StrictComparison": ("params_reference", "StrictComparison"),
    "compare_params_strict": ("params_reference", "compare_params_strict"),
    "compare_semantic": ("params_reference", "compare_semantic"),
    "LigandParityEntry": ("parity_manifest", "LigandParityEntry"),
    "load_parity_manifest": ("parity_manifest", "load_parity_manifest"),
    "default_dataset_manifest": ("parity_manifest", "default_dataset_manifest"),
    "test_assign_tmol_atom_types_is_robust": (
        "test_atom_typing_coverage",
        "test_assign_tmol_atom_types_is_robust",
    ),
    "test_assign_tmol_atom_types_spot_checks": (
        "test_atom_typing_coverage",
        "test_assign_tmol_atom_types_spot_checks",
    ),
    "test_assign_missing_hybridization_geometry_and_aromatic": (
        "test_atom_typing_coverage",
        "test_assign_missing_hybridization_geometry_and_aromatic",
    ),
    "test_bond_is_planar": ("test_atom_typing_coverage", "test_bond_is_planar"),
    "TestClassifierHelpers": ("test_atom_typing_coverage", "TestClassifierHelpers"),
    "test_assign_tmol_atom_types_returns_state": (
        "test_atom_typing_coverage",
        "test_assign_tmol_atom_types_returns_state",
    ),
    "test_get_hyb_distinguishes_ar_vs_aro_subtypes": (
        "test_atom_typing_regressions",
        "test_get_hyb_distinguishes_ar_vs_aro_subtypes",
    ),
    "test_ring_nitrogen_correction_forces_nim": (
        "test_atom_typing_regressions",
        "test_ring_nitrogen_correction_forces_nim",
    ),
    "test_classify_n_sp2_no_nameerror_on_aromatic_context": (
        "test_atom_typing_regressions",
        "test_classify_n_sp2_no_nameerror_on_aromatic_context",
    ),
    "test_state_marks_strained_and_aromatic_ring_atoms": (
        "test_atom_typing_regressions",
        "test_state_marks_strained_and_aromatic_ring_atoms",
    ),
    "test_amide_bond_correction_promotes_nad_cdp_single_bond": (
        "test_atom_typing_regressions",
        "test_amide_bond_correction_promotes_nad_cdp_single_bond",
    ),
    "test_classify_n_hyb8_amide_primary_and_tertiary": (
        "test_atom_typing_regressions",
        "test_classify_n_hyb8_amide_primary_and_tertiary",
    ),
    "test_classify_o2_oxime_guard_requires_sp2_n": (
        "test_atom_typing_regressions",
        "test_classify_o2_oxime_guard_requires_sp2_n",
    ),
    "test_classify_o3_aromatic_ring_oxygen_maps_to_ofu": (
        "test_atom_typing_regressions",
        "test_classify_o3_aromatic_ring_oxygen_maps_to_ofu",
    ),
    "test_classify_o3_nonaromatic_ring_oxygen_maps_to_oet": (
        "test_atom_typing_regressions",
        "test_classify_o3_nonaromatic_ring_oxygen_maps_to_oet",
    ),
    "test_six_member_mixed_sp2_sp3_ring_is_not_aromatic": (
        "test_atom_typing_regressions",
        "test_six_member_mixed_sp2_sp3_ring_is_not_aromatic",
    ),
    "test_missing_hybridization_assignment_for_nh_subtype": (
        "test_atom_typing_regressions",
        "test_missing_hybridization_assignment_for_nh_subtype",
    ),
    "test_p_and_s_follow_hyb5_classification": (
        "test_atom_typing_regressions",
        "test_p_and_s_follow_hyb5_classification",
    ),
    "test_modify_polar_c_promotes_cdp": (
        "test_atom_typing_regressions",
        "test_modify_polar_c_promotes_cdp",
    ),
    "test_long_ring_aromatic_planarity_gate": (
        "test_atom_typing_regressions",
        "test_long_ring_aromatic_planarity_gate",
    ),
    "test_classify_n_pl3_ring_hetero_tertiary_maps_to_nim": (
        "test_atom_typing_regressions",
        "test_classify_n_pl3_ring_hetero_tertiary_maps_to_nim",
    ),
    "test_conjugated_single_bond_promotion_for_conjugating_classes": (
        "test_atom_typing_regressions",
        "test_conjugated_single_bond_promotion_for_conjugating_classes",
    ),
    "test_classify_o2_uses_rosetta_first_bond_behavior": (
        "test_atom_typing_regressions",
        "test_classify_o2_uses_rosetta_first_bond_behavior",
    ),
    "test_five_member_ring_sp3_oxygen_aromatic_exception": (
        "test_atom_typing_regressions",
        "test_five_member_ring_sp3_oxygen_aromatic_exception",
    ),
    "test_classify_n_hetero_accepts_sp2_oxygen_neighbor_without_double_bond": (
        "test_atom_typing_regressions",
        "test_classify_n_hetero_accepts_sp2_oxygen_neighbor_without_double_bond",
    ),
    "test_classify_n_sp2_nonaromatic_nh_maps_to_ng21_not_nin": (
        "test_atom_typing_regressions",
        "test_classify_n_sp2_nonaromatic_nh_maps_to_ng21_not_nin",
    ),
    "test_classify_n2_nonaromatic_tertiary_with_n_neighbor_maps_to_nad3": (
        "test_atom_typing_regressions",
        "test_classify_n2_nonaromatic_tertiary_with_n_neighbor_maps_to_nad3",
    ),
    "test_conjugated_single_bond_promotion_does_not_require_planarity_by_default": (
        "test_atom_typing_regressions",
        "test_conjugated_single_bond_promotion_does_not_require_planarity_by_default",
    ),
    "test_sanitize_tolerant_handles_nonring_aromatic_placeholders": (
        "test_atom_typing_regressions",
        "test_sanitize_tolerant_handles_nonring_aromatic_placeholders",
    ),
    "test_classify_n_pl3_protonated_maps_to_nam2": (
        "test_atom_typing_regressions",
        "test_classify_n_pl3_protonated_maps_to_nam2",
    ),
    "test_classify_n_am_protonated_tertiary_maps_to_nam2": (
        "test_atom_typing_regressions",
        "test_classify_n_am_protonated_tertiary_maps_to_nam2",
    ),
    "test_classify_n2_protonated_with_n_neighbor_maps_to_nam2": (
        "test_atom_typing_regressions",
        "test_classify_n2_protonated_with_n_neighbor_maps_to_nam2",
    ),
    "test_classify_n2_protonated_formal_charge_maps_to_nam2": (
        "test_atom_typing_regressions",
        "test_classify_n2_protonated_formal_charge_maps_to_nam2",
    ),
    "test_ligand_atom_array_allows_passthrough_unknown_bond_type": (
        "test_atom_typing_regressions",
        "test_ligand_atom_array_allows_passthrough_unknown_bond_type",
    ),
    "test_conjugated_single_bond_skips_biaryl_like_ring_pivot": (
        "test_atom_typing_regressions",
        "test_conjugated_single_bond_skips_biaryl_like_ring_pivot",
    ),
    "test_large_ring_bonds_keep_ring_flag_in_residue_type": (
        "test_atom_typing_regressions",
        "test_large_ring_bonds_keep_ring_flag_in_residue_type",
    ),
    "test_sanitize_tolerant_preserves_existing_double_bond_without_aromatic_rewrite": (
        "test_atom_typing_regressions",
        "test_sanitize_tolerant_preserves_existing_double_bond_without_aromatic_rewrite",
    ),
    "test_hbond_properties_derived_from_chemical_db": (
        "test_chemistry_tables",
        "test_hbond_properties_derived_from_chemical_db",
    ),
    "test_polar_and_sp2_classes_come_from_db_tables": (
        "test_chemistry_tables",
        "test_polar_and_sp2_classes_come_from_db_tables",
    ),
    "FIXTURE_DIR": ("test_cif_to_dg", "FIXTURE_DIR"),
    "test_cif_smiles_matches_reference": (
        "test_cif_to_dg",
        "test_cif_smiles_matches_reference",
    ),
    "test_prepare_ligand_from_cif_registers_residue": (
        "test_cif_to_dg",
        "test_prepare_ligand_from_cif_registers_residue",
    ),
    "test_cif_to_params_golden": ("test_cif_to_dg", "test_cif_to_params_golden"),
    "test_cif_ligand_pose_scores_finite": (
        "test_cif_to_dg",
        "test_cif_ligand_pose_scores_finite",
    ),
    "test_fused_purine_ligand_uses_openbabel_charges": (
        "test_cif_to_dg",
        "test_fused_purine_ligand_uses_openbabel_charges",
    ),
    "test_smiles_hard_fail_without_bond_table": (
        "test_cif_to_dg",
        "test_smiles_hard_fail_without_bond_table",
    ),
    "DUD_DIR": ("test_dud_ligands", "DUD_DIR"),
    "DUD_CASES": ("test_dud_ligands", "DUD_CASES"),
    "TestDUDScoring": ("test_dud_ligands", "TestDUDScoring"),
    "test_read_mol2_summary_counts": (
        "test_fixture_integrity",
        "test_read_mol2_summary_counts",
    ),
    "test_paired_fixture_passes": (
        "test_fixture_integrity",
        "test_paired_fixture_passes",
    ),
    "test_paired_fixture_accepts_params_path": (
        "test_fixture_integrity",
        "test_paired_fixture_accepts_params_path",
    ),
    "test_orphan_ref1_is_rejected_with_heavy_count_mismatch": (
        "test_fixture_integrity",
        "test_orphan_ref1_is_rejected_with_heavy_count_mismatch",
    ),
    "test_residue_name_mismatch_is_rejected": (
        "test_fixture_integrity",
        "test_residue_name_mismatch_is_rejected",
    ),
    "test_dud80_ace_1_is_a_valid_pair": (
        "test_fixture_integrity",
        "test_dud80_ace_1_is_a_valid_pair",
    ),
    "test_charge_model_mismatch_user_vs_mmff94_is_rejected": (
        "test_fixture_integrity",
        "test_charge_model_mismatch_user_vs_mmff94_is_rejected",
    ),
    "test_charge_model_mmff94_matches": (
        "test_fixture_integrity",
        "test_charge_model_mmff94_matches",
    ),
    "test_charge_model_unknown_expected_is_rejected": (
        "test_fixture_integrity",
        "test_charge_model_unknown_expected_is_rejected",
    ),
    "test_charge_model_no_charges_fails_auto": (
        "test_fixture_integrity",
        "test_charge_model_no_charges_fails_auto",
    ),
    "test_dud80_ace_1_charge_model_enforced": (
        "test_fixture_integrity",
        "test_dud80_ace_1_charge_model_enforced",
    ),
    "DATA_DIR": ("test_fragmented_ligand_scoring", "DATA_DIR"),
    "TARGET": ("test_fragmented_ligand_scoring", "TARGET"),
    "LIGAND_NAME": ("test_fragmented_ligand_scoring", "LIGAND_NAME"),
    "MULTI_CUTS": ("test_fragmented_ligand_scoring", "MULTI_CUTS"),
    "test_fragment_definition_connections_icoors_and_mapping": (
        "test_fragmented_ligand_scoring",
        "test_fragment_definition_connections_icoors_and_mapping",
    ),
    "test_fragmented_ligand_export_restores_original_residue": (
        "test_fragmented_ligand_scoring",
        "test_fragmented_ligand_export_restores_original_residue",
    ),
    "test_fragmentation_uses_ligand_already_in_parameter_database": (
        "test_fragmented_ligand_scoring",
        "test_fragmentation_uses_ligand_already_in_parameter_database",
    ),
    "test_fragment_interactions_validate_inputs": (
        "test_fragmented_ligand_scoring",
        "test_fragment_interactions_validate_inputs",
    ),
    "test_duplicate_ligand_names_require_same_fragment_layout": (
        "test_fragmented_ligand_scoring",
        "test_duplicate_ligand_names_require_same_fragment_layout",
    ),
    "test_prepare_ligands_rejects_too_small_fragment": (
        "test_fragmented_ligand_scoring",
        "test_prepare_ligands_rejects_too_small_fragment",
    ),
    "test_prepare_ligands_rejects_unassigned_fragment_atoms_public_path": (
        "test_fragmented_ligand_scoring",
        "test_prepare_ligands_rejects_unassigned_fragment_atoms_public_path",
    ),
    "test_fragment_validation_rejects_unsupported_layouts": (
        "test_fragmented_ligand_scoring",
        "test_fragment_validation_rejects_unsupported_layouts",
    ),
    "test_fragment_mapping_is_stable_for_atom_array_stack": (
        "test_fragmented_ligand_scoring",
        "test_fragment_mapping_is_stable_for_atom_array_stack",
    ),
    "test_fragmented_ligand_minimize_and_pack_e2e": (
        "test_fragmented_ligand_scoring",
        "test_fragmented_ligand_minimize_and_pack_e2e",
    ),
    "test_fragmented_ligand_ddg_and_total_pose_parity": (
        "test_fragmented_ligand_scoring",
        "test_fragmented_ligand_ddg_and_total_pose_parity",
    ),
    "DATA": ("test_ligand_entry_paths", "DATA"),
    "MOL2_DIR": ("test_ligand_entry_paths", "MOL2_DIR"),
    "test_prepare_ligand_from_mol2_registers_residue": (
        "test_ligand_entry_paths",
        "test_prepare_ligand_from_mol2_registers_residue",
    ),
    "test_prepare_ligand_from_cif_uses_default_db_when_omitted": (
        "test_ligand_entry_paths",
        "test_prepare_ligand_from_cif_uses_default_db_when_omitted",
    ),
    "test_nonstandard_residue_info_from_mol2_block_roundtrips": (
        "test_ligand_entry_paths",
        "test_nonstandard_residue_info_from_mol2_block_roundtrips",
    ),
    "test_write_params_from_mol2_both_formats": (
        "test_ligand_entry_paths",
        "test_write_params_from_mol2_both_formats",
    ),
    "test_write_params_file_rosetta_list_to_directory": (
        "test_ligand_entry_paths",
        "test_write_params_file_rosetta_list_to_directory",
    ),
    "test_tmol_params_roundtrip_and_inject": (
        "test_ligand_entry_paths",
        "test_tmol_params_roundtrip_and_inject",
    ),
    "test_tmol_loader_accepts_minor_version_difference": (
        "test_ligand_entry_paths",
        "test_tmol_loader_accepts_minor_version_difference",
    ),
    "test_tmol_loader_warns_when_no_charges": (
        "test_ligand_entry_paths",
        "test_tmol_loader_warns_when_no_charges",
    ),
    "test_tmol_loader_rejects_bad_files": (
        "test_ligand_entry_paths",
        "test_tmol_loader_rejects_bad_files",
    ),
    "test_prepare_ligand_from_smiles_registers": (
        "test_ligand_entry_paths",
        "test_prepare_ligand_from_smiles_registers",
    ),
    "CIF_INPUTS": ("test_ligand_entry_paths", "CIF_INPUTS"),
    "test_prepare_ligand_from_cif_inputs": (
        "test_ligand_entry_paths",
        "test_prepare_ligand_from_cif_inputs",
    ),
    "test_prepare_ligands_writes_params_output": (
        "test_ligand_entry_paths",
        "test_prepare_ligands_writes_params_output",
    ),
    "test_prepare_ligands_accepts_single_model_stack": (
        "test_ligand_entry_paths",
        "test_prepare_ligands_accepts_single_model_stack",
    ),
    "test_prepare_ligands_rejects_multi_model_stack": (
        "test_ligand_entry_paths",
        "test_prepare_ligands_rejects_multi_model_stack",
    ),
    "test_prepare_ligands_strict_raises_on_unpreparable_ligand": (
        "test_ligand_entry_paths",
        "test_prepare_ligands_strict_raises_on_unpreparable_ligand",
    ),
    "test_prepare_ligands_lenient_skips_unpreparable_ligand": (
        "test_ligand_entry_paths",
        "test_prepare_ligands_lenient_skips_unpreparable_ligand",
    ),
    "test_prepare_ligands_with_params_files_skips_reprep": (
        "test_ligand_entry_paths",
        "test_prepare_ligands_with_params_files_skips_reprep",
    ),
    "PLI_CIF_INPUT_DIR": ("test_ligand_pipeline", "PLI_CIF_INPUT_DIR"),
    "TestDetectFromCIF": ("test_ligand_pipeline", "TestDetectFromCIF"),
    "TestFullPipeline": ("test_ligand_pipeline", "TestFullPipeline"),
    "TestLigandScoringData": ("test_ligand_pipeline", "TestLigandScoringData"),
    "test_prepare_ligands_missing_ligand_atom_fails": (
        "test_ligand_pipeline",
        "test_prepare_ligands_missing_ligand_atom_fails",
    ),
    "test_ddg_from_cif_complex_with_onthefly_ligand_prep": (
        "test_ligand_pipeline",
        "test_ddg_from_cif_complex_with_onthefly_ligand_prep",
    ),
    "TestParamsRoundtrip": ("test_ligand_pipeline", "TestParamsRoundtrip"),
    "test_collect_new_atom_types_strict_mode_errors": (
        "test_ligand_pipeline",
        "test_collect_new_atom_types_strict_mode_errors",
    ),
    "test_protonate_mol_variants_produces_valid_mol": (
        "test_ligand_pipeline",
        "test_protonate_mol_variants_produces_valid_mol",
    ),
    "test_prepare_ligand_from_cif_helper_loads_reference_fixture": (
        "test_ligand_pipeline",
        "test_prepare_ligand_from_cif_helper_loads_reference_fixture",
    ),
    "TestCovalentDetection": ("test_ligand_pipeline", "TestCovalentDetection"),
    "test_btn_close_contact_ligand_not_dropped_as_covalent": (
        "test_ligand_pipeline",
        "test_btn_close_contact_ligand_not_dropped_as_covalent",
    ),
    "GROUND_TRUTH": ("test_ligand_unit_coverage", "GROUND_TRUTH"),
    "TestDetectHelpers": ("test_ligand_unit_coverage", "TestDetectHelpers"),
    "TestStructureToSmiles": ("test_ligand_unit_coverage", "TestStructureToSmiles"),
    "TestAuthoritativeCharges": (
        "test_ligand_unit_coverage",
        "TestAuthoritativeCharges",
    ),
    "TestEquivalenceElementFromName": (
        "test_ligand_unit_coverage",
        "TestEquivalenceElementFromName",
    ),
    "TestLigandAtomArrayToRdkitMol": (
        "test_ligand_unit_coverage",
        "TestLigandAtomArrayToRdkitMol",
    ),
    "TestPreparationHelpers": ("test_ligand_unit_coverage", "TestPreparationHelpers"),
    "TestParamsIo": ("test_ligand_unit_coverage", "TestParamsIo"),
    "test_charge_sidecar_length_matches_atom_records": (
        "test_params_reference",
        "test_charge_sidecar_length_matches_atom_records",
    ),
    "test_parse_captures_name_and_nbr_and_charges_are_floats": (
        "test_params_reference",
        "test_parse_captures_name_and_nbr_and_charges_are_floats",
    ),
    "test_reference_charges_accepts_path_and_object": (
        "test_params_reference",
        "test_reference_charges_accepts_path_and_object",
    ),
    "test_all_bond_pairs_are_hydrogen_inclusive": (
        "test_params_reference",
        "test_all_bond_pairs_are_hydrogen_inclusive",
    ),
    "test_legacy_dict_shape_is_preserved": (
        "test_params_reference",
        "test_legacy_dict_shape_is_preserved",
    ),
    "test_compare_charges_identical_passes": (
        "test_params_reference",
        "test_compare_charges_identical_passes",
    ),
    "test_compare_charges_perturbation_fails_beyond_tolerance": (
        "test_params_reference",
        "test_compare_charges_perturbation_fails_beyond_tolerance",
    ),
    "test_compare_charges_no_shared_atoms_is_not_ok": (
        "test_params_reference",
        "test_compare_charges_no_shared_atoms_is_not_ok",
    ),
    "test_compare_charges_missing_key_fails_by_default": (
        "test_params_reference",
        "test_compare_charges_missing_key_fails_by_default",
    ),
    "test_compare_charges_extra_key_fails_by_default": (
        "test_params_reference",
        "test_compare_charges_extra_key_fails_by_default",
    ),
    "test_compare_charges_subset_mode_allows_missing_keys": (
        "test_params_reference",
        "test_compare_charges_subset_mode_allows_missing_keys",
    ),
    "test_reference_params_is_frozen": (
        "test_params_reference",
        "test_reference_params_is_frozen",
    ),
    "test_strict_comparator_passes_on_matching_fields": (
        "test_params_reference",
        "test_strict_comparator_passes_on_matching_fields",
    ),
    "test_strict_comparator_flags_atom_type_change": (
        "test_params_reference",
        "test_strict_comparator_flags_atom_type_change",
    ),
    "test_strict_comparator_flags_removed_bond": (
        "test_params_reference",
        "test_strict_comparator_flags_removed_bond",
    ),
    "test_strict_comparator_flags_icoor_topology_change": (
        "test_params_reference",
        "test_strict_comparator_flags_icoor_topology_change",
    ),
    "test_strict_comparator_flags_nbr_atom_change": (
        "test_params_reference",
        "test_strict_comparator_flags_nbr_atom_change",
    ),
    "test_strict_comparator_flags_charge_perturbation": (
        "test_params_reference",
        "test_strict_comparator_flags_charge_perturbation",
    ),
    "test_semantic_comparator_equates_renamed_copy": (
        "test_params_reference",
        "test_semantic_comparator_equates_renamed_copy",
    ),
    "test_semantic_comparator_flags_type_change": (
        "test_params_reference",
        "test_semantic_comparator_flags_type_change",
    ),
    "test_semantic_comparator_handles_pdb_carbon_name_collision": (
        "test_params_reference",
        "test_semantic_comparator_handles_pdb_carbon_name_collision",
    ),
    "test_element_from_atom_type_prefixes": (
        "test_params_reference",
        "test_element_from_atom_type_prefixes",
    ),
    "test_seed_entries_are_smiles_only": (
        "test_parity_manifest",
        "test_seed_entries_are_smiles_only",
    ),
    "test_manifest_load_resolves_relative_paths": (
        "test_parity_manifest",
        "test_manifest_load_resolves_relative_paths",
    ),
    "test_manifest_accepts_bare_list": (
        "test_parity_manifest",
        "test_manifest_accepts_bare_list",
    ),
    "test_manifest_rejects_missing_expected_prot_smiles": (
        "test_parity_manifest",
        "test_manifest_rejects_missing_expected_prot_smiles",
    ),
    "test_manifest_rejects_missing_params_file": (
        "test_parity_manifest",
        "test_manifest_rejects_missing_params_file",
    ),
    "test_manifest_rejects_missing_mol2_file": (
        "test_parity_manifest",
        "test_manifest_rejects_missing_mol2_file",
    ),
    "test_missing_manifest_path_raises": (
        "test_parity_manifest",
        "test_missing_manifest_path_raises",
    ),
    "test_entry_count_grows_with_manifest": (
        "test_parity_manifest",
        "test_entry_count_grows_with_manifest",
    ),
    "test_dud80_manifest_loads_all_paired_entries": (
        "test_parity_manifest",
        "test_dud80_manifest_loads_all_paired_entries",
    ),
    "PLI_DATA_DIR": ("test_protein_ligand_ddg", "PLI_DATA_DIR"),
    "LIGAND_RES_NAME": ("test_protein_ligand_ddg", "LIGAND_RES_NAME"),
    "test_protein_ligand_cif_to_ddg_golden": (
        "test_protein_ligand_ddg",
        "test_protein_ligand_cif_to_ddg_golden",
    ),
    "seed_prep": ("test_serialization_consistency", "seed_prep"),
    "test_params_tmol_overlapping_fields_agree": (
        "test_serialization_consistency",
        "test_params_tmol_overlapping_fields_agree",
    ),
    "test_charge_perturbation_breaks_consistency": (
        "test_serialization_consistency",
        "test_charge_perturbation_breaks_consistency",
    ),
    "test_proton_chi_sample_corruption_breaks_consistency": (
        "test_serialization_consistency",
        "test_proton_chi_sample_corruption_breaks_consistency",
    ),
    "test_sample_proton_chi_setting_drives_emission": (
        "test_serialization_consistency",
        "test_sample_proton_chi_setting_drives_emission",
    ),
    "entry_prep": ("test_smiles_semantic", "entry_prep"),
    "test_smiles_prep_structural_equivalence": (
        "test_smiles_semantic",
        "test_smiles_prep_structural_equivalence",
    ),
    "test_smiles_prep_charge_equivalence": (
        "test_smiles_semantic",
        "test_smiles_prep_charge_equivalence",
    ),
    "test_changed_heavy_graph_is_detected": (
        "test_smiles_semantic",
        "test_changed_heavy_graph_is_detected",
    ),
    "test_changed_chi_axis_is_detected": (
        "test_smiles_semantic",
        "test_changed_chi_axis_is_detected",
    ),
    "_element_from_atom_type": ("equivalence", "_element_from_atom_type"),
    "_heavy_atom_name_mapping": ("equivalence", "_heavy_atom_name_mapping"),
    "_infer_element_from_name": ("equivalence", "_infer_element_from_name"),
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
