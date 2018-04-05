import unittest

from tmol.system.residue.io import read_pdb
from tmol.score.hbond import HBondElementAnalysis
from tmol.tests.data.pdb import data as test_pdbs


class TestHBond(unittest.TestCase):
    def test_backbone_detection(self):
        sys = read_pdb(test_pdbs["1ubq"])

        hbe = HBondElementAnalysis.setup(
            atom_types = sys.atom_types,
            bonds = sys.bonds,
        )

        bb_donor = set(hbe.donors["d"])
        assert len(bb_donor) == len(hbe.donors)
        for ri, r in zip(sys.start_ind, sys.residues):
            ni = r.residue_type.atom_to_idx["N"]
            n_atom = r.residue_type.atoms[ni]
            if n_atom.atom_type != "Nbb":
                assert r.residue_type.name3 == "PRO"
            else:
                assert ni + ri in bb_donor

        assert len(hbe.donors)    == len([r for r in sys.residues if r.residue_type.name3 != "PRO"])
        assert len(hbe.sp2_acceptors) == len(sys.residues)
