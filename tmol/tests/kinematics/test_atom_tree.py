import unittest
from collections import Counter
import tmol.kinematics.AtomTree as atree
import tmol.kinematics.HTRefold as htrefold
from tmol.tests.data.pdb import data as test_pdbs
import tmol.system.residue.io as pdbio
import tmol.io.pdb_parsing as pdb_parsing
import numpy
import math


class temp_class:
    pass


def print_tree(residues, root, depth=0):
    print ( " " * depth, residues[ root.atomid.res ].residue_type.atoms[ root.atomid.atomno ].name, \
                root.xyz, root.phi, root.theta, root.d )
    for atom in root.children:
        print_tree(residues, atom, depth + 1)


def print_tree_no_names(root, depth=0):
    if root.is_jump:
        print ( " " * depth, root.atomid, \
                root.xyz, root.rb, root.rot_delta, root.rot )
    else:
        print ( " " * depth, root.atomid, \
                root.xyz, root.phi, root.theta, root.d )
    for atom in root.children:
        print_tree_no_names(atom, depth + 1)


def print_res1_tree(residues, root, depth=0):
    if root.atomid.res != 0:
        return
    print ( " " * depth, residues[ root.atomid.res ].residue_type.atoms[ root.atomid.atomno ].name, \
                root.xyz, root.phi, root.theta, root.d )
    for atom in root.children:
        print_res1_tree(residues, atom, depth + 1)


def faux_score(coords):
    # Frank's dummy score
    natoms = coords.shape[0]
    dists = numpy.sqrt(numpy.square(coords[:,numpy.newaxis]-coords).sum(axis=2))
    igraph = numpy.bitwise_and(numpy.triu(~numpy.eye(dists.shape[0],dtype=bool)),dists < 3.4).nonzero();
    score = (3.4-dists[igraph])*(3.4-dists[igraph])
    return numpy.sum(score)

def faux_score_derivs(coords):
    natoms = coords.shape[0]
    #print("faux score derivs, natoms=", natoms)
    dxs = coords[:,numpy.newaxis]-coords
    dists = numpy.sqrt(numpy.square(dxs).sum(axis=2))
    igraph = numpy.bitwise_and(numpy.triu(~numpy.eye(dists.shape[0],dtype=bool)),dists < 3.4).nonzero();

    dEdxs = numpy.zeros([natoms,natoms,3])
    dEdxs[igraph[0],igraph[1],:] = -2 * (3.4-dists[igraph].reshape(-1,1)) * dxs[igraph]/dists[igraph].reshape(-1,1);

    dEdx = numpy.zeros([natoms,3])
    dEdx = dEdxs.sum(axis=1) - dEdxs.sum(axis=0)

    return dEdx

class TestAtomTree(unittest.TestCase):
    def test_homogeneous_transform_default_ctor(self):
        ht = atree.HomogeneousTransform()
        self.assertEqual(ht.frame[0, 0], 1)
        self.assertEqual(ht.frame[1, 1], 1)
        self.assertEqual(ht.frame[2, 2], 1)
        self.assertEqual(ht.frame[3, 3], 1)
        self.assertEqual(ht.frame[0, 1], 0)
        self.assertEqual(ht.frame[0, 2], 0)
        self.assertEqual(ht.frame[0, 3], 0)
        self.assertEqual(ht.frame[1, 0], 0)
        self.assertEqual(ht.frame[1, 2], 0)
        self.assertEqual(ht.frame[1, 3], 0)
        self.assertEqual(ht.frame[2, 0], 0)
        self.assertEqual(ht.frame[2, 1], 0)
        self.assertEqual(ht.frame[2, 3], 0)
        self.assertEqual(ht.frame[3, 0], 0)
        self.assertEqual(ht.frame[3, 1], 0)
        self.assertEqual(ht.frame[3, 2], 0)

    def test_homogeneous_transform_from_three_points(self):
        p1 = numpy.array([1., 1., 1.])
        p2 = numpy.array([0., 1., 1.])
        p3 = numpy.array([1., 3., 1.])
        ht = atree.HomogeneousTransform.from_three_coords(p1, p2, p3)
        # we should get back something very close to the identiy matrix
        # xaxis
        self.assertAlmostEqual(ht.frame[0, 0], 1)
        self.assertAlmostEqual(ht.frame[1, 0], 0)
        self.assertAlmostEqual(ht.frame[2, 0], 0)
        self.assertAlmostEqual(ht.frame[3, 0], 0)
        # yaxis
        self.assertAlmostEqual(ht.frame[0, 1], 0)
        self.assertAlmostEqual(ht.frame[1, 1], 1)
        self.assertAlmostEqual(ht.frame[2, 1], 0)
        self.assertAlmostEqual(ht.frame[3, 1], 0)

        # zaxis
        self.assertAlmostEqual(ht.frame[0, 2], 0)
        self.assertAlmostEqual(ht.frame[1, 2], 0)
        self.assertAlmostEqual(ht.frame[2, 2], 1)
        self.assertAlmostEqual(ht.frame[3, 2], 0)

        # coordinate
        self.assertAlmostEqual(ht.frame[0, 3], 0)
        self.assertAlmostEqual(ht.frame[1, 3], 1)
        self.assertAlmostEqual(ht.frame[2, 3], 1)
        self.assertAlmostEqual(ht.frame[3, 3], 1)

    def test_homogeneous_transform_from_four_points(self):
        p1 = numpy.array([1., 1., 1.])
        p2 = numpy.array([0., 1., 1.])
        p3 = numpy.array([1., 3., 1.])
        c = numpy.array([2., 2., 2.])
        ht = atree.HomogeneousTransform.from_four_coords(c, p1, p2, p3)
        # we should get back something very close to the identiy matrix
        # xaxis
        self.assertAlmostEqual(ht.frame[0, 0], 1)
        self.assertAlmostEqual(ht.frame[1, 0], 0)
        self.assertAlmostEqual(ht.frame[2, 0], 0)
        self.assertAlmostEqual(ht.frame[3, 0], 0)
        # yaxis
        self.assertAlmostEqual(ht.frame[0, 1], 0)
        self.assertAlmostEqual(ht.frame[1, 1], 1)
        self.assertAlmostEqual(ht.frame[2, 1], 0)
        self.assertAlmostEqual(ht.frame[3, 1], 0)

        # zaxis
        self.assertAlmostEqual(ht.frame[0, 2], 0)
        self.assertAlmostEqual(ht.frame[1, 2], 0)
        self.assertAlmostEqual(ht.frame[2, 2], 1)
        self.assertAlmostEqual(ht.frame[3, 2], 0)

        # coordinate
        self.assertAlmostEqual(ht.frame[0, 3], 2)
        self.assertAlmostEqual(ht.frame[1, 3], 2)
        self.assertAlmostEqual(ht.frame[2, 3], 2)
        self.assertAlmostEqual(ht.frame[3, 3], 1)

    def test_homogeneous_transform_x_axis_rotation(self):
        ''' Positive rotation about the x axis swings y up into positive z, and
        z down into negative y'''
        ht = atree.HomogeneousTransform()
        ht.set_rotation_x(60 / 180 * numpy.pi)
        self.assertEqual(ht.frame[0, 0], 1)
        self.assertEqual(ht.frame[1, 0], 0)
        self.assertEqual(ht.frame[2, 0], 0)
        self.assertEqual(ht.frame[3, 0], 0)
        self.assertEqual(ht.frame[0, 1], 0)
        self.assertAlmostEqual(ht.frame[1, 1], 0.5)
        self.assertAlmostEqual(ht.frame[2, 1], math.sqrt(3) / 2)
        self.assertEqual(ht.frame[3, 1], 0)
        self.assertEqual(ht.frame[0, 2], 0)
        self.assertAlmostEqual(ht.frame[1, 2], -math.sqrt(3) / 2)
        self.assertAlmostEqual(ht.frame[2, 2], 0.5)
        self.assertEqual(ht.frame[3, 2], 0)
        self.assertEqual(ht.frame[0, 3], 0)
        self.assertEqual(ht.frame[1, 3], 0)
        self.assertEqual(ht.frame[2, 3], 0)
        self.assertEqual(ht.frame[3, 3], 1)

    def test_homogeneous_transform_y_axis_rotation(self):
        ''' Positive rotation about the y axis swings z into positive x and x into negative z'''
        ht = atree.HomogeneousTransform()
        ht.set_rotation_y(60 / 180 * numpy.pi)
        half = 0.5
        root_three_over_two = math.sqrt(3) / 2
        self.assertAlmostEqual(ht.frame[0, 0], half)
        self.assertEqual(ht.frame[1, 0], 0)
        self.assertAlmostEqual(ht.frame[2, 0], -root_three_over_two)
        self.assertEqual(ht.frame[3, 0], 0)
        self.assertEqual(ht.frame[0, 1], 0)
        self.assertEqual(ht.frame[1, 1], 1)
        self.assertEqual(ht.frame[2, 1], 0)
        self.assertEqual(ht.frame[3, 1], 0)
        self.assertAlmostEqual(ht.frame[0, 2], root_three_over_two)
        self.assertEqual(ht.frame[1, 2], 0)
        self.assertAlmostEqual(ht.frame[2, 2], half)
        self.assertEqual(ht.frame[3, 2], 0)
        self.assertEqual(ht.frame[0, 3], 0)
        self.assertEqual(ht.frame[1, 3], 0)
        self.assertEqual(ht.frame[2, 3], 0)
        self.assertEqual(ht.frame[3, 3], 1)

    def test_homogeneous_transform_z_axis_rotation(self):
        ''' Positive rotation about the z axis sends x into positive y and y into negative x'''
        ht = atree.HomogeneousTransform()
        ht.set_rotation_z(60 / 180 * numpy.pi)
        half = 0.5
        root_three_over_two = math.sqrt(3) / 2
        self.assertAlmostEqual(ht.frame[0, 0], half)
        self.assertAlmostEqual(ht.frame[1, 0], root_three_over_two)
        self.assertEqual(ht.frame[2, 0], 0)
        self.assertEqual(ht.frame[3, 0], 0)
        self.assertAlmostEqual(ht.frame[0, 1], -root_three_over_two)
        self.assertAlmostEqual(ht.frame[1, 1], half)
        self.assertEqual(ht.frame[2, 1], 0)
        self.assertEqual(ht.frame[3, 1], 0)
        self.assertEqual(ht.frame[0, 2], 0)
        self.assertEqual(ht.frame[1, 2], 0)
        self.assertEqual(ht.frame[2, 2], 1)
        self.assertEqual(ht.frame[3, 2], 0)
        self.assertEqual(ht.frame[0, 3], 0)
        self.assertEqual(ht.frame[1, 3], 0)
        self.assertEqual(ht.frame[2, 3], 0)
        self.assertEqual(ht.frame[3, 3], 1)

    def test_walk_via_ht_multiplication(self):
        origin = atree.HomogeneousTransform()
        # let's put p2 in yz plane
        ht_xrot1 = atree.HomogeneousTransform.xrot(-30 / 180 * numpy.pi)
        #print("ht_xrot1\n",ht_xrot1)
        ht_transz1 = atree.HomogeneousTransform.ztrans(2.5)
        #print("ht_transz1\n",ht_transz1)
        ht2 = origin * ht_xrot1 * ht_transz1
        #print("ht2")
        #print(ht2)
        p2 = ht2.frame[0:3, 3]

        self.assertEqual(p2[0], 0)
        self.assertAlmostEqual(p2[1], 2.5 * 0.5)
        self.assertAlmostEqual(p2[2], 2.5 * math.sqrt(3) / 2)

    def test_measure_internal_coords_simple_system(self):
        p1 = (0, 0, 0)
        p2 = (0, 0, 2.5)
        p3 = (0, 1.25, 3.75)
        p4 = (1.5, 2.75, 4.25)

        root = atree.BondedAtom()
        root.xyz = p1
        child1 = atree.BondedAtom()
        child1.parent = root
        child1.xyz = p2
        root.children.append(child1)
        child2 = atree.BondedAtom()
        child2.parent = child1
        child2.xyz = p3
        child1.children.append(child2)
        child3 = atree.BondedAtom()
        child3.parent = child2
        child3.xyz = p4
        child2.children.append(child3)

        root.update_internal_coords()

        self.assertEqual(
            child3.d, numpy.linalg.norm(numpy.array(p4) - numpy.array(p3))
        )

    def test_construct_residue_tree(self):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        root, atom_pointers = atree.create_residue_tree( res_reader.chemical_db, \
                                                             residues[0].residue_type, 0, "N" )
        atree.set_coords(residues[0], atom_pointers)
        root.update_internal_coords()
        #print_tree( residues, root )

    def test_construct_whole_structure_atom_tree(self):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        tree = atree.tree_from_residues(res_reader.chemical_db, residues)

        # now set a new chi1 for residue 1 and refold
        indCG = residues[0].residue_type.atom_to_idx["CG"]
        tree.atom_pointer_list[0][indCG].phi = math.pi
        tree.update_xyz()

        cgat = tree.atom_pointer_list[0][indCG]
        #print( "chi1 end: ", cgat.phi, cgat.theta, cgat.d, cgat.xyz )
        final_cg_ideal = numpy.array([24.01077925, 25.87729449, 3.88653434])
        final_cg_actual = numpy.array(tree.atom_pointer_list[0][indCG].xyz)
        #print_res1_tree( residues, tree.root )
        self.assertAlmostEqual(
            numpy.linalg.norm(final_cg_actual - final_cg_ideal), 0.0
        )

        # dump the pdb to look at it
        #atom_records = pdb_parsing.parse_pdb( test_pdbs[ "1UBQ" ] )
        #for at_node in tree.atom_pointer_list[0] :
        #    #at_node = tree.atom_pointer_list[0][ atname ]
        #    atname = residues[ at_node.atomid.res ].residue_type.atoms[ at_node.atomid.atomno ].name
        #    found = False
        #    for ind in range(len(atom_records)) :
        #        if atom_records.loc[ ind, "resi" ] == 1 and atom_records.loc[ ind, "chain" ] == "A" \
        #                and atom_records.loc[ ind, "atomn" ] == atname :
        #            atom_records.loc[ ind, [ "x", "y", "z" ]] = at_node.xyz
        #            found = True
        #            break
        #    assert( found )
        #with open( "test_refold3.pdb", "w" ) as fid :
        #    fid.writelines( pdb_parsing.to_pdb( atom_records ) )

    def test_atomtree_refold_info_setup1(self):
        residues, nodes, tree, coords, bas, jas = self.create_franks_multi_jump_atom_tree()
        #nodes, coords = self.create_franks_multi_jump_atom_tree()

        atom_node_list = [[nodes[0],nodes[1],nodes[2]], \
                          [nodes[3],nodes[4],nodes[5],nodes[6],nodes[7]], \
                          [nodes[5+3],nodes[5+4],nodes[5+5],nodes[5+6],nodes[5+7]], \
                          [nodes[10+3],nodes[10+4],nodes[10+5],nodes[10+6],nodes[10+7]], \
                          [nodes[15+3],nodes[15+4],nodes[15+5],nodes[15+6],nodes[15+7]]]

        tree = atree.AtomTree(nodes[0], atom_node_list)

        # faux residues object
        residues = [ temp_class() for x in range(5) ]
        residues[0].coords = numpy.zeros((3,3))
        residues[1].coords = numpy.zeros((5,3))
        residues[2].coords = numpy.zeros((5,3))
        residues[3].coords = numpy.zeros((5,3))
        residues[4].coords = numpy.zeros((5,3))

        dofs = numpy.zeros((23,9))

        # the bonded atom indices in Frank's example
        bas = [0,1,2,4,5,6,7,9,10,11,12,14,15,16,17,19,20,21,22]
        for ba in bas:
            dofs[ba,0] = nodes[ba].d
            dofs[ba,1] = nodes[ba].theta
            dofs[ba,2] = nodes[ba].phi
        jas = [3,8,13,18]
        for ja in jas:
            for i in range(3):
                dofs[ja,i+0] = nodes[ja].rb[i]
                dofs[ja,i+3] = nodes[ja].rot_delta[i]
                dofs[ja,i+6] = nodes[ja].rot[i]

        ordered_roots, refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index = \
            htrefold.initialize_ht_refold_data( residues, tree )

        htrefold.cpu_htrefold_1( residues, tree, refold_data, atoms_for_controlling_torsions, \
                      refold_index_2_atomid, atomid_2_refold_index )

    def test_atomtree_refold_info_setup2(self):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        tree = atree.tree_from_residues(res_reader.chemical_db, residues)

        ordered_roots, refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index = \
            htrefold.initialize_ht_refold_data( residues, tree )

        htrefold.cpu_htrefold_1( residues, tree, refold_data, atoms_for_controlling_torsions, \
                      refold_index_2_atomid, atomid_2_refold_index )

        # let's dump the new coordinates
        #atom_records = pdb_parsing.parse_pdb( test_pdbs[ "1UBQ" ] )
        #for ii, res in enumerate( residues ) :
        #    for jj in range( res.coords.shape[0] ) :
        #        iijj_atname = res.residue_type.atoms[ jj ].name
        #        # oh, god, there has to be a better way!
        #        for kk in range( len( atom_records )) :
        #            if atom_records.loc[ kk, "resi" ] == ii+1 and \
        #                    atom_records.loc[ kk, "chain" ] == "A" and \
        #                    atom_records.loc[ kk, "atomn" ] == iijj_atname :
        #                atom_records.loc[ kk, [ "x", "y", "z" ]] = res.coords[jj]
        #with open( "test_refold2.pdb", "w" ) as fid :
        #    fid.writelines( pdb_parsing.to_pdb( atom_records ) )

    def create_franks_multi_jump_atom_tree(self):
        NATOMS = 23

        BOND = 1
        JUMP = 2

        nodes = [None] * NATOMS

        # kintree = numpy.empty( NATOMS, dtype=kintree_node_dtype);
        nodes[0] = atree.BondedAtom()
        nodes[0].atomid = atree.AtomID(
            0, 0
        )  # kintree[0]  = ("ORIG", 0, BOND,  0,  (1, 0, 2))
        nodes[1] = atree.BondedAtom()
        nodes[1].atomid = atree.AtomID(
            0, 1
        )  # kintree[1]  = (" X  ", 0, BOND,  0,  (1, 0, 2))
        nodes[2] = atree.BondedAtom()
        nodes[2].atomid = atree.AtomID(
            0, 2
        )  # kintree[2]  = (" Y  ", 0, BOND,  0,  (2, 0, 1))
        nodes[0].children.append(nodes[1])
        nodes[1].parent = nodes[0]
        nodes[1].children.append(nodes[2])
        nodes[2].parent = nodes[1]

        nodes[3] = atree.JumpAtom()
        nodes[3].atomid = atree.AtomID(
            1, 0
        )  # kintree[3]  = (" N  ", 1, JUMP,  0,  (4, 3, 5))
        nodes[4] = atree.BondedAtom()
        nodes[4].atomid = atree.AtomID(
            1, 1
        )  # kintree[4]  = (" CA ", 1, BOND,  3,  (4, 3, 5))
        nodes[5] = atree.BondedAtom()
        nodes[5].atomid = atree.AtomID(
            1, 2
        )  # kintree[5]  = (" CB ", 1, BOND,  4,  (5, 4, 3))
        nodes[6] = atree.BondedAtom()
        nodes[6].atomid = atree.AtomID(
            1, 3
        )  # kintree[6]  = (" C  ", 1, BOND,  4,  (6, 4, 3))
        nodes[7] = atree.BondedAtom()
        nodes[7].atomid = atree.AtomID(
            1, 4
        )  # kintree[7]  = (" O  ", 1, BOND,  6,  (7, 6, 4))
        nodes[0].children.insert(0, nodes[3])
        nodes[3].parent = nodes[0]
        nodes[3].children.append(nodes[4])
        nodes[4].parent = nodes[3]
        nodes[4].children.append(nodes[5])
        nodes[5].parent = nodes[4]
        nodes[4].children.append(nodes[6])
        nodes[6].parent = nodes[4]
        nodes[6].children.append(nodes[7])
        nodes[7].parent = nodes[6]

        nodes[8] = atree.JumpAtom()
        nodes[8].atomid = atree.AtomID(
            2, 0
        )  # kintree[8]  = (" N  ", 2, JUMP,  3,  (9, 8, 10))
        nodes[9] = atree.BondedAtom()
        nodes[9].atomid = atree.AtomID(
            2, 1
        )  # kintree[9]  = (" CA ", 2, BOND,  8,  (9, 8, 10))
        nodes[10] = atree.BondedAtom()
        nodes[10].atomid = atree.AtomID(
            2, 2
        )  # kintree[10] = (" CB ", 2, BOND,  9, (10, 9, 8))
        nodes[11] = atree.BondedAtom()
        nodes[11].atomid = atree.AtomID(
            2, 3
        )  # kintree[11] = (" C  ", 2, BOND,  9, (11, 9, 8))
        nodes[12] = atree.BondedAtom()
        nodes[12].atomid = atree.AtomID(
            2, 4
        )  # kintree[12] = (" O  ", 2, BOND, 11, (12,11, 9))
        nodes[3].children.insert(0, nodes[8])
        nodes[8].parent = nodes[3]
        nodes[5 + 3].children.append(nodes[5 + 4])
        nodes[5 + 4].parent = nodes[5 + 3]
        nodes[5 + 4].children.append(nodes[5 + 5])
        nodes[5 + 5].parent = nodes[5 + 4]
        nodes[5 + 4].children.append(nodes[5 + 6])
        nodes[5 + 6].parent = nodes[5 + 4]
        nodes[5 + 6].children.append(nodes[5 + 7])
        nodes[5 + 7].parent = nodes[5 + 6]

        nodes[13] = atree.JumpAtom()
        nodes[13].atomid = atree.AtomID(
            3, 0
        )  # kintree[13] = (" N  ", 3, JUMP,  3, (14, 13, 15))
        nodes[14] = atree.BondedAtom()
        nodes[14].atomid = atree.AtomID(
            3, 1
        )  # kintree[14] = (" CA ", 3, BOND, 13, (14, 13, 15))
        nodes[15] = atree.BondedAtom()
        nodes[15].atomid = atree.AtomID(
            3, 2
        )  # kintree[15] = (" CB ", 3, BOND, 14, (15, 14, 13))
        nodes[16] = atree.BondedAtom()
        nodes[16].atomid = atree.AtomID(
            3, 3
        )  # kintree[16] = (" C  ", 3, BOND, 14, (16, 14, 13))
        nodes[17] = atree.BondedAtom()
        nodes[17].atomid = atree.AtomID(
            3, 4
        )  # kintree[17] = (" O  ", 3, BOND, 16, (17, 16, 14))
        nodes[3].children.insert(1, nodes[13])
        nodes[13].parent = nodes[3]
        nodes[10 + 3].children.append(nodes[10 + 4])
        nodes[10 + 4].parent = nodes[10 + 3]
        nodes[10 + 4].children.append(nodes[10 + 5])
        nodes[10 + 5].parent = nodes[10 + 4]
        nodes[10 + 4].children.append(nodes[10 + 6])
        nodes[10 + 6].parent = nodes[10 + 4]
        nodes[10 + 6].children.append(nodes[10 + 7])
        nodes[10 + 7].parent = nodes[10 + 6]

        nodes[18] = atree.JumpAtom()
        nodes[18].atomid = atree.AtomID(
            4, 0
        )  # kintree[18] = (" N  ", 4, JUMP,  3, (19, 18, 20))
        nodes[19] = atree.BondedAtom()
        nodes[19].atomid = atree.AtomID(
            4, 1
        )  # kintree[19] = (" CA ", 4, BOND, 18, (19, 18, 20))
        nodes[20] = atree.BondedAtom()
        nodes[20].atomid = atree.AtomID(
            4, 2
        )  # kintree[20] = (" CB ", 4, BOND, 19, (20, 19, 18))
        nodes[21] = atree.BondedAtom()
        nodes[21].atomid = atree.AtomID(
            4, 3
        )  # kintree[21] = (" C  ", 4, BOND, 19, (21, 19, 18))
        nodes[22] = atree.BondedAtom()
        nodes[22].atomid = atree.AtomID(
            4, 4
        )  # kintree[22] = (" O  ", 4, BOND, 21, (22, 21, 19))
        nodes[3].children.insert(2, nodes[18])
        nodes[18].parent = nodes[3]
        nodes[15 + 3].children.append(nodes[15 + 4])
        nodes[15 + 4].parent = nodes[15 + 3]
        nodes[15 + 4].children.append(nodes[15 + 5])
        nodes[15 + 5].parent = nodes[15 + 4]
        nodes[15 + 4].children.append(nodes[15 + 6])
        nodes[15 + 6].parent = nodes[15 + 4]
        nodes[15 + 6].children.append(nodes[15 + 7])
        nodes[15 + 7].parent = nodes[15 + 6]

        coords = numpy.empty([NATOMS, 3])
        coords[0, :] = [0.000, 0.000, 0.000]
        coords[1, :] = [1.000, 0.000, 0.000]
        coords[2, :] = [0.000, 1.000, 0.000]

        coords[3, :] = [2.000, 2.000, 2.000]
        coords[4, :] = [3.458, 2.000, 2.000]
        coords[5, :] = [3.988, 1.222, 0.804]
        coords[6, :] = [4.009, 3.420, 2.000]
        coords[7, :] = [3.383, 4.339, 1.471]

        coords[8, :] = [5.184, 3.594, 2.596]
        coords[9, :] = [5.821, 4.903, 2.666]
        coords[10, :] = [5.331, 5.667, 3.888]
        coords[11, :] = [7.339, 4.776, 2.690]
        coords[12, :] = [7.881, 3.789, 3.186]

        coords[13, :] = [7.601, 2.968, 5.061]
        coords[14, :] = [6.362, 2.242, 4.809]
        coords[15, :] = [6.431, 0.849, 5.419]
        coords[16, :] = [5.158, 3.003, 5.349]
        coords[17, :] = [5.265, 3.736, 6.333]

        coords[18, :] = [4.011, 2.824, 4.701]
        coords[19, :] = [2.785, 3.494, 5.115]
        coords[20, :] = [2.687, 4.869, 4.470]
        coords[21, :] = [1.559, 2.657, 4.776]
        coords[22, :] = [1.561, 1.900, 3.805]

        for ii, node in enumerate(nodes):
            node.xyz = coords[ii, :]

        nodes[0].update_internal_coords()

        # faux residues object
        residues = [ temp_class() for x in range(5) ]
        residues[0].coords = numpy.zeros((3,3))
        residues[1].coords = numpy.zeros((5,3))
        residues[2].coords = numpy.zeros((5,3))
        residues[3].coords = numpy.zeros((5,3))
        residues[4].coords = numpy.zeros((5,3))

        atom_node_list = [[nodes[0],nodes[1],nodes[2]], \
                          [nodes[3],nodes[4],nodes[5],nodes[6],nodes[7]], \
                          [nodes[5+3],nodes[5+4],nodes[5+5],nodes[5+6],nodes[5+7]], \
                          [nodes[10+3],nodes[10+4],nodes[10+5],nodes[10+6],nodes[10+7]], \
                          [nodes[15+3],nodes[15+4],nodes[15+5],nodes[15+6],nodes[15+7]]]

        tree = atree.AtomTree(nodes[0], atom_node_list)

        # the bonded atoms
        bas = [0,1,2,4,5,6,7,9,10,11,12,14,15,16,17,19,20,21,22]
        # the jump atoms
        jas = [3,8,13,18]

        return residues, nodes, tree, coords, bas, jas

    def dofs_for_franks_multi_jump_atom_tree( self, nodes, bas, jas ):
        dofs = numpy.zeros((23,9))

        # the bonded atom indices in Frank's example
        #bas = [0,1,2,4,5,6,7,9,10,11,12,14,15,16,17,19,20,21,22]
        for ba in bas:
            dofs[ba,0] = nodes[ba].d
            dofs[ba,1] = nodes[ba].theta
            dofs[ba,2] = nodes[ba].phi
        #jas = [3,8,13,18]
        for ja in jas:
            for i in range(3):
                dofs[ja,i+0] = nodes[ja].rb[i]
                dofs[ja,i+3] = nodes[ja].rot_delta[i]
                dofs[ja,i+6] = nodes[ja].rot[i]
        return dofs


    def test_atomtree_w_jump_atoms(self):
        residues, nodes, tree, coords, bas, jas = self.create_franks_multi_jump_atom_tree()
        #nodes, coords = self.create_franks_multi_jump_atom_tree()
        #print_tree_no_names( nodes[0] )

        nodes[0].update_xyz()
        #print_tree_no_names( nodes[0] )

        for ii, node in enumerate(nodes):
            self.assertAlmostEqual(
                numpy.linalg.norm(numpy.array(node.xyz) - coords[ii, :]), 0
            )

    def count_nodes_in_ag_tree(self, root):
        count = 0
        if root is not None:
            count += 1
            count += self.count_nodes_in_ag_tree(root.first_child)
            if root.first_child: self.assertIs(root, root.first_child.parent)
            for child in root.other_children:
                count += self.count_nodes_in_ag_tree(child)
                self.assertIs(root, child.parent)
            count += self.count_nodes_in_ag_tree(root.younger_sibling)
            if root.younger_sibling:
                self.assertIs(root, root.younger_sibling.older_sibling)
        return count

    def test_create_abe_go_tree(self):
        # create the UBQ atom tree
        # follow with the Abe-Go tree
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        tree = atree.tree_from_residues(res_reader.chemical_db, residues)

        ag_root, ag_nodes = htrefold.abe_and_go_tree_from_atom_tree(tree)
        count_ag_nodes = sum([sum([len(x) for x in y]) for y in ag_nodes])
        count_at_nodes = sum([len(x) for x in tree.atom_pointer_list])

        # there are two nodes for every bonded atom
        self.assertEqual(count_at_nodes * 2, count_ag_nodes)

        # make sure all nodes have either a parent or an older sibling but not both
        for res_nodes in ag_nodes:
            for atom_nodes in res_nodes:
                for node in atom_nodes:
                    self.assertTrue(
                        node.parent is not None
                        or node.older_sibling is not None or node is ag_root
                    )
                    self.assertTrue(
                        node.parent is None or node.older_sibling is None
                    )

        # make sure a traversal starting at the root hits all children
        self.assertEqual(count_ag_nodes, self.count_nodes_in_ag_tree(ag_root))

    def test_create_derivsum_indices(self):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        tree = atree.tree_from_residues(res_reader.chemical_db, residues)

        ag_root, ag_nodes = htrefold.abe_and_go_tree_from_atom_tree(tree)
        path_roots = htrefold.create_abe_and_go_paths(ag_root)
        root_depth = htrefold.find_abe_go_path_depths(ag_root, path_roots)

        derivsum_index_2_ag_id, ag_id_2_derivsum_index, depth_start_inds = \
            htrefold.create_derivsum_indices( ag_nodes, path_roots )

        for ii, res in enumerate(ag_id_2_derivsum_index):
            for jj, at in enumerate(res):
                for kk, derivsum_id in enumerate(at):
                    self.assertEqual(
                        derivsum_index_2_ag_id[derivsum_id],
                        htrefold.AbeGoID(atree.AtomID(ii, jj), kk)
                    )

        # derivsum_index_2_ag_id[ ag_id.atomid.res ][ ag_id.atomid.atomno ][ ag_id.nodeid ], derivsum_id ) )

    def create_example_abe_go_tree(self, atom_f1f2s):
        # let's imagine a tree with the following structure:
        # 0 JumpAtom
        # |          \
        # 1 JumpAtom   6 BondedAtom
        # |        \          \
        # 2 Bonded  3 Bonded   7 BondedAtom
        # |                    |          \
        # 4 Bonded             8 BondedAtom 9 BondedAtom
        # |
        # 5 Bonded
        #
        # then
        # we will have 2*nbonded + 1*njump = 18 AbeGo nodes in the resulting AbeGo tree
        # 17 JA0 ----------------
        #  |                     \
        # 16 JA1                  \
        #  |                       \
        # 15 BA2Phi -- 1 BA3Phi     9 BA6Phi
        #  |           |            |
        # 14 BA2TD     0 BA3TD      8 BA6TD
        #  |                        |
        # 13 BA4Phi                 7 BA76Phi
        #  |                        |
        # 12 BA4TD                  6 BA7TD
        #  |                        |
        # 11 BA5Phi                 5 BA8Phi  --  3 BA9Phi
        #  |                        |             |
        # 10 BA5TD                  4 BA8TD       2 BA9TD
        #
        ag_derivsum_nodes = temp_class()
        ag_derivsum_nodes.nnodes = 18
        # note 19 entries; 18th entry corresponds to fictional entry
        ag_derivsum_nodes.has_initial_f1f2 = numpy.array( [     True, True, True, True, \
                                                                True, True, True, True, \
                                                                True, True, True, True, \
                                                                True, True, True, True, \
                                                                True, True, True ]
                                                          )
        # note an index of natoms (10) means "no children" and will index into the atomf1f2
        # matrix to a row of zeros.
        ag_derivsum_nodes.atom_indices = numpy.array( \
            [ 3, 10, 9, 10, 8, 10, 7, 10, 6, 10, 5, 10, 4, 10, 2, 10, 1, 0, 10 ] )
        ag_derivsum_nodes.is_leaf = numpy.array( \
            [ True, False, True, False, True, False, False, False, False, False, \
                  True, False, False, False, False, False, False, False ] )
        ag_derivsum_nodes.is_leaf_working = ag_derivsum_nodes.is_leaf.copy()

        # note an index of n-nodes (18) means "no children nodes" and will index into the f1f2sum
        # matrix to a row of zeros.
        ag_derivsum_nodes.prior_children = numpy.reshape( numpy.array( [ \
                18, 18, \
                     18, 18, \
                     18,  3, \
                     18, 18, \
                     18, 18, \
                     18, 18, \
                     18, 18, \
                     18,  1, \
                     18,  9 ] ), ( 18, 1 ) )
        ag_derivsum_nodes.lookback_inds = numpy.arange(18)
        ag_derivsum_nodes.atom_range_for_depth = [
            None
        ] * 3  #numpy.full( (3, 18), False, dtype=bool )

        ag_derivsum_nodes.atom_range_for_depth[0] = [0, 4]
        ag_derivsum_nodes.atom_range_for_depth[1] = [4, 10]
        ag_derivsum_nodes.atom_range_for_depth[2] = [10, 18]

        ag_derivsum_nodes.natoms_at_depth = numpy.array([4, 6, 8])

        gold_ancestors = [None] * 18
        gold_ancestors[0] = numpy.array([3])
        gold_ancestors[1] = numpy.array([3])
        gold_ancestors[2] = numpy.array([9])
        gold_ancestors[3] = numpy.array([9])
        gold_ancestors[4] = numpy.array([8])
        gold_ancestors[5] = numpy.array([8, 9])
        gold_ancestors[6] = numpy.array([8, 9, 7])
        gold_ancestors[7] = numpy.array([8, 9, 7])
        gold_ancestors[8] = numpy.array([8, 9, 7, 6])
        gold_ancestors[9] = numpy.array([8, 9, 7, 6])
        gold_ancestors[10] = numpy.array([5])
        gold_ancestors[11] = numpy.array([5])
        gold_ancestors[12] = numpy.array([5, 4])
        gold_ancestors[13] = numpy.array([5, 4])
        gold_ancestors[14] = numpy.array([5, 4, 2])
        gold_ancestors[15] = numpy.array([5, 4, 2, 3])
        gold_ancestors[16] = numpy.array([5, 4, 2, 3, 1])
        gold_ancestors[17] = numpy.array([5, 4, 2, 3, 1, 8, 9, 7, 6, 0])

        #print( "ancestors13" )
        #print( atom_f1f2s[ gold_ancestors[13], : ] )
        #print( numpy.sum( atom_f1f2s[ gold_ancestors[13], : ], 0 ) )

        f1f2sum_gold = numpy.zeros((18, 6))
        for ii, ancestors in enumerate(gold_ancestors):
            #print( "ii?", ii )
            #print(ii, ancestors)
            #print(ii, atom_f1f2s[ancestors,:])
            #print(ii,f1f2sum_gold[ ii:ii+1 ])
            #print(ii,f1f2sum_gold[ ii:ii+1 ].shape)
            f1f2sum_gold[ii:ii + 1] = numpy.sum(atom_f1f2s[ancestors, :], 0)
        return ag_derivsum_nodes, gold_ancestors, f1f2sum_gold

    def test_f1f2_summation_1(self):
        atom_f1f2s = numpy.arange(66).reshape((11, 6))
        atom_f1f2s[10, :] = 0.
        # last row in the table, natoms+1, is all zeros
        ag_derivsum_nodes, gold_ancestors, f1f2sum_gold = self.create_example_abe_go_tree(
            atom_f1f2s
        )
        f1f2sum = htrefold.cpu_f1f2_summation1(atom_f1f2s, ag_derivsum_nodes)

        for ii in range(18):
            #print(ii, "gold", f1f2sum_gold[ii] )
            #print(ii, "computed", f1f2sum[ii] )
            for jj in range(6):
                self.assertAlmostEqual(f1f2sum[ii, jj], f1f2sum_gold[ii, jj])

    def test_f1f2_summation_2(self):
        atom_f1f2s = numpy.arange(66).reshape((11, 6))
        atom_f1f2s[10, :] = 0.
        # last row in the table, natoms+1, is all zeros
        ag_derivsum_nodes, gold_ancestors, f1f2sum_gold = self.create_example_abe_go_tree(
            atom_f1f2s
        )
        f1f2sum = htrefold.cpu_f1f2_summation2(atom_f1f2s, ag_derivsum_nodes)

        for ii in range(18):
            #print(ii, "gold", f1f2sum_gold[ii] )
            #print(ii, "computed", f1f2sum[ii] )
            for jj in range(6):
                self.assertAlmostEqual(f1f2sum[ii, jj], f1f2sum_gold[ii, jj])

    def test_create_f1f2_tree_for_1ubq(self):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        atom_tree = atree.tree_from_residues(res_reader.chemical_db, residues)
        refold_data = htrefold.initialize_whole_structure_refold_data( residues, atom_tree )
        ag_tree = htrefold.create_abe_go_f1f2sum_tree_for_structure(
            residues, atom_tree, 
            refold_data.coalesced_ind_2_refold_index, refold_data.refold_index_2_coalesced_ind
        )

        atom_f1f2s = numpy.random.random((ag_tree.natoms + 1, 6))
        atom_f1f2s[ag_tree.natoms, :] = 0.
        f1f2sum = htrefold.cpu_f1f2_summation2(atom_f1f2s, ag_tree)


    def test_numpy_ht_refold(self):
        residues, nodes, tree, coords, bas, jas = self.create_franks_multi_jump_atom_tree()
        dofs = self.dofs_for_franks_multi_jump_atom_tree(nodes,bas,jas)
        refold_data = htrefold.initialize_whole_structure_refold_data( residues, tree )
        coords_out = coords.copy()
        htrefold.cpu_htrefold_2( dofs, refold_data, coords_out )
        #print( "coords"); print( coords )
        #print( "coords_out"); print( coords_out )

        for ii, node in enumerate(nodes):
            self.assertAlmostEqual(
                numpy.linalg.norm(coords_out[ii,:] - coords[ii, :]), 0
            )

    def test_numpy_ht_refold_2(self):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        tree = atree.tree_from_residues(res_reader.chemical_db, residues)

        refold_data = htrefold.initialize_whole_structure_refold_data( residues, tree )

        dofs = numpy.zeros((refold_data.natoms,9))
        count = 0
        for res_ptrs in tree.atom_pointer_list:
            for at in res_ptrs :
                if not at.is_jump :
                    dofs[count,0] = at.d
                    dofs[count,1] = at.theta
                    dofs[count,2] = at.phi
                else :
                    for i in range(3):
                        dofs[count,i+0] = at.rb[i]
                        dofs[count,i+3] = at.rot_delta[i]
                        dofs[count,i+6] = at.rot[i]
                count += 1
        coords_out = numpy.zeros( (refold_data.natoms,3))
        #print("dofs");print(dofs)
        htrefold.cpu_htrefold_2( dofs, refold_data, coords_out )
        #print("coords_out");print(coords_out)

        count = 0
        for res_ptrs in tree.atom_pointer_list:
            for at in res_ptrs :
                #print(coords_out[count,:],"vs",at.xyz)
                self.assertAlmostEqual(
                    numpy.linalg.norm(coords_out[count,:] - at.xyz), 0
                    )
                count += 1


    def test_dof_derivative_calculations( self ):
        residues, nodes, atom_tree, coords, bas, jas = self.create_franks_multi_jump_atom_tree()
        refold_data = htrefold.initialize_whole_structure_refold_data( residues, atom_tree )

        dofs = self.dofs_for_franks_multi_jump_atom_tree( nodes, bas, jas )
        ag_tree = htrefold.create_abe_go_f1f2sum_tree_for_structure(residues, atom_tree, \
             refold_data.coalesced_ind_2_refold_index, refold_data.refold_index_2_coalesced_ind )
        dofs_working = dofs.copy()
        delta_coords = coords.copy()
        htrefold.cpu_htrefold_2( dofs, refold_data, delta_coords )

        score = faux_score( coords )
        cart_derivs = faux_score_derivs( coords )

        # Debugging: trying to make sure that the f2s that were recursively summed
        # matched the f2s that should have been summed
        #
        #gold_descendants = [ None ] * 23
        #gold_descendants[0] = [ i for i in range(23) ]
        #gold_descendants[1] = [ 1, 2 ]
        #gold_descendants[2] = [ 2 ]
        #gold_descendants[3] = [ i for i in range(3,23) ]
        #gold_descendants[4] = [ i for i in range(4,8) ]
        #gold_descendants[5] = [ 5 ]
        #gold_descendants[6] = [ 6, 7 ]
        #gold_descendants[7] = [ 7 ]
        #gold_descendants[8] = [ i for i in range(8,13) ]
        #gold_descendants[9] = [ i for i in range(9,13) ]
        #gold_descendants[10] = [ 10 ]
        #gold_descendants[11] = [ 11, 12 ]
        #gold_descendants[12] = [ 12 ]
        #gold_descendants[13] = [ i for i in range(13,18) ]
        #gold_descendants[14] = [ i for i in range(14,18) ]
        #gold_descendants[15] = [ 15 ]
        #gold_descendants[16] = [ 16, 17 ]
        #gold_descendants[17] = [ 17 ]
        #gold_descendants[18] = [ i for i in range(18,23) ]
        #gold_descendants[19] = [ i for i in range(19,23) ]
        #gold_descendants[20] = [ 20 ]
        #gold_descendants[21] = [ 21, 22 ]
        #gold_descendants[22] = [ 22 ]
        #print("gold_f2s"); #print(gold_f2s)
        #for ii,desc_list in enumerate(gold_descendants):
        #    print(ii,numpy.sum(cart_derivs[numpy.array(desc_list)],0))

        #print("cart_derivs"); print(cart_derivs)
        dscore_ddofs_analytic = htrefold.compute_dscore_ddofs( coords, dofs, ag_tree, refold_data.hts, cart_derivs )
        #print("dscore_ddofs_analytic"); print( dscore_ddofs_analytic )
        dscore_ddofs_numeric = numpy.zeros((23,6))
        count_ci = 0
        for ii in range(23) :
            ndofs = 3 if ii in bas else 6
            for jj in range(ndofs) :
                delta = 1e-8
                dofs_working[ii,jj] = dofs[ii,jj] + delta
                htrefold.cpu_htrefold_2(dofs_working, refold_data, delta_coords)
                score_pdelta = faux_score(delta_coords)
                dofs_working[ii,jj] = dofs[ii,jj] - delta
                htrefold.cpu_htrefold_2(dofs_working, refold_data, delta_coords)
                score_mdelta = faux_score(delta_coords)
                dscore_ddofs_numeric[count_ci,jj] = ( score_pdelta - score_mdelta ) / ( 2*delta)
            count_ci += 1
        #print( "dscore_ddofs_numeric"); print(dscore_ddofs_numeric)
        for ii in range(23):
            ndofs = 3 if ii in bas else 6
            for jj in range(ndofs):
                self.assertAlmostEqual(dscore_ddofs_numeric[ii,jj], dscore_ddofs_analytic[ii,jj],5)
                dofs_working[ii,jj] = dofs[ii,jj]
                

    def do_not_test_dof_derivative_calculations_2( self ):
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb(test_pdbs["1UBQ"])
        atom_tree = atree.tree_from_residues(res_reader.chemical_db, residues)

        #residues, nodes, atom_tree, coords, bas, jas = self.create_franks_multi_jump_atom_tree()
        refold_data = htrefold.initialize_whole_structure_refold_data( residues, atom_tree )

        #dofs = self.dofs_for_franks_multi_jump_atom_tree( nodes, bas, jas )
        dofs = numpy.zeros((refold_data.natoms,9))
        count = 0
        for res_ptrs in atom_tree.atom_pointer_list:
            for at in res_ptrs :
                if not at.is_jump :
                    dofs[count,0] = at.d
                    dofs[count,1] = at.theta
                    dofs[count,2] = at.phi
                else :
                    for i in range(3):
                        dofs[count,i+0] = at.rb[i]
                        dofs[count,i+3] = at.rot_delta[i]
                        dofs[count,i+6] = at.rot[i]
                count += 1

        ag_tree = htrefold.create_abe_go_f1f2sum_tree_for_structure(residues, atom_tree, \
             refold_data.coalesced_ind_2_refold_index, refold_data.refold_index_2_coalesced_ind )
        dofs_working = dofs.copy()
        coords = numpy.zeros((refold_data.natoms,3))
        delta_coords = coords.copy()
        htrefold.cpu_htrefold_2( dofs, refold_data, coords )

        score = faux_score( coords )
        cart_derivs = faux_score_derivs( coords )

        #print("cart_derivs"); print(cart_derivs)
        dscore_ddofs_analytic = htrefold.compute_dscore_ddofs( coords, dofs, ag_tree, refold_data.hts, cart_derivs )
        #print("dscore_ddofs_analytic"); print( dscore_ddofs_analytic )
        dscore_ddofs_numeric = numpy.zeros((refold_data.natoms,6))
        count_ci = 0
        for ii in range(refold_data.natoms) :
            ndofs = 6 #3 if ii in bas else 6
            for jj in range(ndofs) :
                delta = 1e-8
                dofs_working[ii,jj] = dofs[ii,jj] + delta
                htrefold.cpu_htrefold_2(dofs_working, refold_data, delta_coords)
                score_pdelta = faux_score(delta_coords)
                dofs_working[ii,jj] = dofs[ii,jj] - delta
                htrefold.cpu_htrefold_2(dofs_working, refold_data, delta_coords)
                score_mdelta = faux_score(delta_coords)
                dscore_ddofs_numeric[count_ci,jj] = ( score_pdelta - score_mdelta ) / ( 2*delta)
            count_ci += 1
        #print( "dscore_ddofs_numeric"); print(dscore_ddofs_numeric)
        for ii in range(refold_data.natoms):
            ndofs = 6 # if ii in bas else 6
            for jj in range(ndofs):
                self.assertAlmostEqual(dscore_ddofs_numeric[ii,jj], dscore_ddofs_analytic[ii,jj],5)
                dofs_working[ii,jj] = dofs[ii,jj]
                
