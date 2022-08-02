import pandas

from tmol.system.kinematics import KinematicDescription
from tmol.kinematics.metadata import DOFMetadata
from tmol.kinematics.datatypes import KinForest
from tmol.system.packed import PackedResidueSystem


def report_tree_coverage(sys: PackedResidueSystem, ktree: KinForest):
    kinematic_metadata = DOFMetadata.for_kinforest(ktree).to_frame()
    torsion_metadata = pandas.DataFrame.from_records(sys.torsion_metadata)
    torsion_coverage = pandas.merge(
        left=torsion_metadata.query(
            "atom_index_a >= 0 "
            "and atom_index_b >= 0 "
            "and atom_index_c >= 0 "
            "and atom_index_d >= 0 "
        ),
        left_on=["atom_index_b", "atom_index_c"],
        right=kinematic_metadata.query("dof_type == 'bond_torsion'"),
        right_on=["parent_id", "child_id"],
        how="left",
    )

    missing_torsions = torsion_coverage[pandas.isna(torsion_coverage["node_idx"])]

    return {"missing_torsions": missing_torsions}


def test_system_kinematics(ubq_system):
    tsys = ubq_system

    tsys_kinematics = KinematicDescription.for_system(
        ubq_system.system_size, ubq_system.bonds, (ubq_system.torsion_metadata,)
    )

    kinematic_tree_results = report_tree_coverage(tsys, tsys_kinematics.kinforest)

    assert len(kinematic_tree_results["missing_torsions"]) == 0, (
        f"Generated kinematic tree did not cover all named torsions.\n"
        f"torsions:\n{kinematic_tree_results['missing_torsions']}\n"
    )
