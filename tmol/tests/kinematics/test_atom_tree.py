import unittest
from collections import Counter
import tmol.kinematics.AtomTree as atree
from tmol.tests.data.pdb import data as test_pdbs
import tmol.system.residue.io as pdbio
import tmol.io.pdb_parsing as pdb_parsing
import numpy
import math

def print_tree( root, depth = 0 ) :
    print ( " " * depth, root.name, root.xyz, root.phi, root.theta, root.d )
    for atom in root.children :
        print_tree( atom, depth+1 )


class TestAtomTree(unittest.TestCase):
    def test_homogeneous_transform_default_ctor(self):
        ht = atree.HomogeneousTransform()
        self.assertEqual( ht.frame[0,0], 1 )
        self.assertEqual( ht.frame[1,1], 1 )
        self.assertEqual( ht.frame[2,2], 1 )
        self.assertEqual( ht.frame[3,3], 1 )
        self.assertEqual( ht.frame[0,1], 0 )
        self.assertEqual( ht.frame[0,2], 0 )
        self.assertEqual( ht.frame[0,3], 0 )
        self.assertEqual( ht.frame[1,0], 0 )
        self.assertEqual( ht.frame[1,2], 0 )
        self.assertEqual( ht.frame[1,3], 0 )
        self.assertEqual( ht.frame[2,0], 0 )
        self.assertEqual( ht.frame[2,1], 0 )
        self.assertEqual( ht.frame[2,3], 0 )
        self.assertEqual( ht.frame[3,0], 0 )
        self.assertEqual( ht.frame[3,1], 0 )
        self.assertEqual( ht.frame[3,2], 0 )

    def test_homogeneous_transform_from_three_points( self ) :
        p1 = numpy.array( [0., 2.,  0. ] )
        p2 = numpy.array( [0., 0., -1. ] )
        p3 = numpy.array( [0., 0.,  0. ] )
        ht = atree.HomogeneousTransform.from_coords( p1, p2, p3 )
        # we should get back something very close to the identiy matrix
        self.assertAlmostEqual( ht.frame[0,0], 1 )
        self.assertAlmostEqual( ht.frame[1,1], 1 )
        self.assertAlmostEqual( ht.frame[2,2], 1 )
        self.assertAlmostEqual( ht.frame[3,3], 1 )
        self.assertAlmostEqual( ht.frame[0,1], 0 )
        self.assertAlmostEqual( ht.frame[0,2], 0 )
        self.assertAlmostEqual( ht.frame[0,3], 0 )
        self.assertAlmostEqual( ht.frame[1,0], 0 )
        self.assertAlmostEqual( ht.frame[1,2], 0 )
        self.assertAlmostEqual( ht.frame[1,3], 0 )
        self.assertAlmostEqual( ht.frame[2,0], 0 )
        self.assertAlmostEqual( ht.frame[2,1], 0 )
        self.assertAlmostEqual( ht.frame[2,3], 0 )
        self.assertAlmostEqual( ht.frame[3,0], 0 )
        self.assertAlmostEqual( ht.frame[3,1], 0 )
        self.assertAlmostEqual( ht.frame[3,2], 0 )

    def test_homogeneous_transform_x_axis_rotation( self ) :
        ''' Positive rotation about the x axis swings y up into positive z, and 
        z down into negative y'''
        ht = atree.HomogeneousTransform()
        ht.set_rotation_x( 60 / 180 * numpy.pi )
        self.assertEqual( ht.frame[0,0], 1 )
        self.assertEqual( ht.frame[1,0], 0 )
        self.assertEqual( ht.frame[2,0], 0 )
        self.assertEqual( ht.frame[3,0], 0 )
        self.assertEqual( ht.frame[0,1], 0 )
        self.assertAlmostEqual( ht.frame[1,1], 0.5 )
        self.assertAlmostEqual( ht.frame[2,1], math.sqrt(3)/2 )
        self.assertEqual( ht.frame[3,1], 0 )
        self.assertEqual( ht.frame[0,2], 0 )
        self.assertAlmostEqual( ht.frame[1,2], -math.sqrt(3)/2 )
        self.assertAlmostEqual( ht.frame[2,2], 0.5 )
        self.assertEqual( ht.frame[3,2], 0 )
        self.assertEqual( ht.frame[0,3], 0 )
        self.assertEqual( ht.frame[1,3], 0 )
        self.assertEqual( ht.frame[2,3], 0 )
        self.assertEqual( ht.frame[3,3], 1 )

    def test_homogeneous_transform_y_axis_rotation( self ) :
        ''' Positive rotation about the y axis swings z into positive x and x into negative z'''
        ht = atree.HomogeneousTransform()
        ht.set_rotation_y( 60 / 180 * numpy.pi )
        half = 0.5
        root_three_over_two = math.sqrt(3)/2
        self.assertAlmostEqual( ht.frame[0,0], half )
        self.assertEqual( ht.frame[1,0], 0 )
        self.assertAlmostEqual( ht.frame[2,0], -root_three_over_two )
        self.assertEqual( ht.frame[3,0], 0 )
        self.assertEqual( ht.frame[0,1], 0 )
        self.assertEqual( ht.frame[1,1], 1 )
        self.assertEqual( ht.frame[2,1], 0 )
        self.assertEqual( ht.frame[3,1], 0 )
        self.assertAlmostEqual( ht.frame[0,2], root_three_over_two )
        self.assertEqual( ht.frame[1,2], 0 )
        self.assertAlmostEqual( ht.frame[2,2], half )
        self.assertEqual( ht.frame[3,2], 0 )
        self.assertEqual( ht.frame[0,3], 0 )
        self.assertEqual( ht.frame[1,3], 0 )
        self.assertEqual( ht.frame[2,3], 0 )
        self.assertEqual( ht.frame[3,3], 1 )

    def test_homogeneous_transform_z_axis_rotation( self ) :
        ''' Positive rotation about the z axis sends x into positive y and y into negative x'''
        ht = atree.HomogeneousTransform()
        ht.set_rotation_z( 60 / 180 * numpy.pi )
        half = 0.5
        root_three_over_two = math.sqrt(3)/2
        self.assertAlmostEqual( ht.frame[0,0], half )
        self.assertAlmostEqual( ht.frame[1,0], root_three_over_two )
        self.assertEqual( ht.frame[2,0], 0 )
        self.assertEqual( ht.frame[3,0], 0 )
        self.assertAlmostEqual( ht.frame[0,1], -root_three_over_two )
        self.assertAlmostEqual( ht.frame[1,1], half )
        self.assertEqual( ht.frame[2,1], 0 )
        self.assertEqual( ht.frame[3,1], 0 )
        self.assertEqual( ht.frame[0,2], 0 )
        self.assertEqual( ht.frame[1,2], 0 )
        self.assertEqual( ht.frame[2,2], 1 )
        self.assertEqual( ht.frame[3,2], 0 )
        self.assertEqual( ht.frame[0,3], 0 )
        self.assertEqual( ht.frame[1,3], 0 )
        self.assertEqual( ht.frame[2,3], 0 )
        self.assertEqual( ht.frame[3,3], 1 )

    def test_walk_via_ht_multiplication( self ) :
        origin = atree.HomogeneousTransform()
        # let's put p2 in yz plane
        ht_xrot1 = atree.HomogeneousTransform.xrot( -30 / 180 * numpy.pi )
        #print("ht_xrot1\n",ht_xrot1)
        ht_transz1 = atree.HomogeneousTransform.ztrans(2.5)
        #print("ht_transz1\n",ht_transz1)
        ht2 = origin * ht_xrot1 * ht_transz1
        #print("ht2")
        #print(ht2)
        p2 = ht2.frame[0:3,3]
        
        self.assertEqual( p2[0], 0 )
        self.assertAlmostEqual( p2[1], 2.5 * 0.5 )
        self.assertAlmostEqual( p2[2], 2.5 * math.sqrt(3)/2 )

    def test_measure_internal_coords_simple_system( self ) :
        p1 = ( 0, 0, 0 )
        p2 = ( 0, 0, 2.5 )
        p3 = ( 0, 1.25, 3.75 )
        p4 = ( 1.5, 2.75, 4.25 )

        root = atree.BondedAtom()
        root.xyz = p1
        child1 = atree.BondedAtom()
        child1.parent = root
        child1.xyz = p2
        root.children.append( child1 )
        child2 = atree.BondedAtom()
        child2.parent = child1
        child2.xyz = p3
        child1.children.append( child2 )
        child3 = atree.BondedAtom()
        child3.parent = child2
        child3.xyz = p4
        child2.children.append( child3 )

        root.update_internal_coords()

        self.assertEqual( child3.d, numpy.linalg.norm( numpy.array(p4) - numpy.array(p3) ) )


    def test_construct_residue_tree( self ) :
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb( test_pdbs[ "1UBQ" ] )
        root, atom_pointers = atree.create_residue_tree( res_reader.chemical_db, \
                                                             residues[0].residue_type, 0, "N" )
        atree.set_coords( residues[0], atom_pointers )
        root.update_internal_coords()
        #print_tree( root )
    
    def test_construct_whole_structure_atom_tree( self ) :
        res_reader = pdbio.ResidueReader()
        residues = res_reader.parse_pdb( test_pdbs[ "1UBQ" ] ) 
        root, atom_pointer_list = atree.tree_from_residues( res_reader.chemical_db, residues )

        # now set a new chi1 for residue 1 and refold
        indCG = residues[0].residue_type.atom_to_idx[ "CG" ]
        atom_pointer_list[ 0 ][ indCG ].phi = math.pi
        root.update_xyz()

        #print( "chi1 end: ", atom_pointer_list[ 0 ][ indCG ].phi, atom_pointer_list[ 0 ][ indCG ].xyz )
        final_cg_ideal = numpy.array( [ 24.01077925,  25.87729449, 3.88653434 ] )
        final_cg_actual = numpy.array( atom_pointer_list[ 0 ][ indCG ].xyz )
        self.assertAlmostEqual( numpy.linalg.norm( final_cg_actual - final_cg_ideal ), 0.0 )

        # dump the pdb to look at it
        #atom_records = pdb_parsing.parse_pdb( test_pdbs[ "1UBQ" ] )
        #for atname in atom_pointer_list[0] :
        #    at_node = atom_pointer_list[0][ atname ]
        #    found = False
        #    for ind in range(len(atom_records)) :
        #        if atom_records.loc[ ind, "resi" ] == 1 and atom_records.loc[ ind, "chain" ] == "A" \
        #                and atom_records.loc[ ind, "atomn" ] == atname :
        #            atom_records.loc[ ind, [ "x", "y", "z" ]] = at_node.xyz
        #            found = True
        #            break
        #    assert( found )
        #with open( "test_refold.pdb", "w" ) as fid :
        #    fid.writelines( pdb_parsing.to_pdb( atom_records ) )
            
                    
