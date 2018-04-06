import yaml
import cattr
import pandas
import numpy
import unittest

import tmol.system.residue.restypes as restypes
import tmol.database

from tmol.system.residue.io import read_pdb
from tmol.score.hbond import HBondElementAnalysis
from tmol.tests.data.pdb import data as test_pdbs


class TestHBond(unittest.TestCase):
    def test_backbone_detection(self):
        sys = read_pdb(test_pdbs["1ubq"])

        bb_hbond_database = cattr.structure(yaml.load("""
        global_parameters:
          max_dis : 6.0
        atom_groups:
          donors:
            - { d: Nbb, h: HNbb, donor_type: hbdon_PBA }
          sp2_acceptors:
            - { a: OCbb, b: CObb, b0: CAbb, acceptor_type: hbacc_PBA }
          sp3_acceptors: []
          ring_acceptors: []
        chemical_types:
          donors:
            - hbdon_PBA
          sp2_acceptors:
            - hbacc_PBA
          sp3_acceptors: []
          ring_acceptors: []
        """), tmol.database.scoring.HBondDatabase)

        donors = []
        acceptors = []
        for ri, r in zip(sys.start_ind, sys.residues):
            ni = r.residue_type.atom_to_idx["N"]
            if r.residue_type.name3 != "PRO":
                donors.append({
                    "d": r.residue_type.atom_to_idx["N"] + ri,
                    "h": r.residue_type.atom_to_idx["H"] + ri,
                    "donor_type" : "hbdon_PBA",
                })

            acceptors.append({
                "a": r.residue_type.atom_to_idx["O"] + ri,
                "b": r.residue_type.atom_to_idx["C"] + ri,
                "b0": r.residue_type.atom_to_idx["CA"] + ri,
                "acceptor_type" : "hbacc_PBA",
            })

        hbe = (
            HBondElementAnalysis(
                hbond_database = bb_hbond_database, atom_types=sys.atom_types, bonds=sys.bonds)
            .setup()
        )

        pandas.testing.assert_frame_equal(
            pandas.DataFrame.from_records(donors, columns=hbe.donors.dtype.names).sort_values("d"),
            pandas.DataFrame.from_records(hbe.donors).sort_values("d")
        )

        pandas.testing.assert_frame_equal(
            pandas.DataFrame.from_records(acceptors, columns=hbe.sp2_acceptors.dtype.names).sort_values("a"),
            pandas.DataFrame.from_records(hbe.sp2_acceptors).sort_values("a")
        )

    def test_identification_by_ljlk_types(self):
        db_res = tmol.database.default.chemical.residues
        types = [
            cattr.structure(cattr.unstructure(r), restypes.ResidueType)
            for r in db_res
        ]
        assert len(types) == 21

        lj_types = { t.name : t for t in tmol.database.default.scoring.ljlk.atom_type_parameters }

        for t in types:
            atom_types=numpy.array([a.atom_type for a in t.atoms])
            bonds=t.bond_indicies

            hbe = HBondElementAnalysis(atom_types=atom_types, bonds=bonds).setup()
            identified_donors = set(hbe.donors["d"])
            identified_acceptors = set(
                list(hbe.sp2_acceptors["a"]) + list(hbe.sp3_acceptors["a"]) + list(hbe.ring_acceptors["a"]))

            for ai, at in enumerate(atom_types):
                if lj_types[at].is_donor:
                    assert ai in identified_donors, \
                        f"Unidentified donor. res: {t.name} atom:{t.atoms[ai]}"
                if lj_types[at].is_acceptor:
                    assert ai in identified_acceptors, \
                        f"Unidentified acceptor. res: {t.name} atom:{t.atoms[ai]}"
