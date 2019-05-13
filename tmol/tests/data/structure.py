from tmol.utility.reactive import reactive_property, reactive_attrs


@reactive_attrs(auto_attribs=True)
class TestStructure:
    pdb: str

    @reactive_property
    def pose(pdb):
        import pyrosetta.distributed.io
        import pyrosetta.distributed.packed_pose
        import tmol.support.rosetta.init  # noqa

        return pyrosetta.distributed.io.pose_from_pdbstring(pdb)

    @reactive_property
    def stripped_pose(pose):
        """Load pdb via rosetta, strip to canonical, non-terminal amino acids."""
        import pyrosetta.distributed.io
        import pyrosetta.distributed.packed_pose

        target_pose = pyrosetta.distributed.packed_pose.to_pose(pose)

        for i in range(len(target_pose.residues)):
            assert target_pose.residues[i + 1].is_protein()

            pyrosetta.rosetta.core.pose.remove_lower_terminus_type_from_pose_residue(
                target_pose, i + 1
            )
            pyrosetta.rosetta.core.pose.remove_upper_terminus_type_from_pose_residue(
                target_pose, i + 1
            )

        return pyrosetta.distributed.packed_pose.to_packed(target_pose)

    @reactive_property
    def tmol_system(stripped_pose):
        import pyrosetta.distributed.io
        import tmol.system.io

        return tmol.system.io.read_pdb(
            pyrosetta.distributed.io.to_pdbstring(stripped_pose)
        )

    @reactive_property
    def tmol_coords(tmol_system):
        import tmol.system.score_support
        import torch

        return tmol.system.score_support.coords_for_system(
            tmol_system, device=torch.device("cpu"), requires_grad=False
        )["coords"][0]
