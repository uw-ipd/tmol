import tmol.kinematics.AtomTree as atree
import attr
import typing
import math

# module for defining a parallelizable refold method in terms of segmented scan of
# matrix multiplications. The first phase of implementation is getting the CPU-based
# version of this function working, which was originally written in C++


@attr.s( auto_attribs=True, slots=True )
class AtomTreePathData :
    longest_subpath: int = 0
    root_of_subpath : bool = False
    child_on_subpath: atree.AtomID = atree.AtomID()
    parent : atree.AtomID = atree.AtomID()

@attr.s( auto_attribs=True, slots=True ) 
class AtomRefoldData :
    phi_offset : float = 0
    theta : float = 0
    d : float = 0
    controlling_torsion : int = -1
    parent_index : int = -1

def initialize_ht_refold_data( residues, tree ) :
    tree_path_data = []
    natoms = 0

    tree_path_data = [ [] for x in residues ]
    for i in range(len(residues)) :
        tree_path_data[i] = [ AtomTreePathData() for x in residues[i].coords ]
        natoms += len(residues[i].coords)

    recurse_and_fill_atomtree_path_data( tree.root, tree_path_data )
    # also mark the root of the atom tree as the root of a subpath, since that is not handled by
    # the recursive routine
    tree_path_data[ tree.root.atomid.res ][ tree.root.atomid.atomno ].root_of_subpath = True

    # now identify all roots of the subpaths; the very shallowest path's depth is depth 0 because
    # python is an index-by-0 language
    root_list = []
    dfs_identify_roots_and_depths( tree.root, 0, tree_path_data, root_list )
    sort_roots_by_depth( root_list )

    ordered_roots = create_ordered_root_set( root_list )
    
    refold_index_2_atomid, atomid_2_refold_index = renumber_atoms( tree_path_data, ordered_roots )
    atoms_for_controlling_torsions, torsion_ids_and_offsets = number_torsions( tree )
    refold_data = fill_atom_refold_data( tree, refold_index_2_atomid, atomid_2_refold_index, torsion_ids_and_offsets, tree_path_data )

    return ordered_roots, refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index 


def cpu_htrefold( residues, tree, refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index ) :
    torsions = torsions_from_tree( tree, atoms_for_controlling_torsions )
    hts = [ atree.HomogeneousTransform() for x in refold_data ]
    # root ht is in fact the identity transform
    for ii, iidat in enumerate( refold_data ) :
        iiatid = refold_index_2_atomid[ ii ]
        print( ii, iidat.controlling_torsion, 180.0 / math.pi * ( torsions[ iidat.controlling_torsion ] if iidat.controlling_torsion >= 0 else 0.0 ), residues[ iiatid.res ].residue_type.atoms[ iiatid.atomno ].name )
               
        phi = iidat.phi_offset + ( torsions[ iidat.controlling_torsion ] if iidat.controlling_torsion >= 0 else 0.0 )
        parent_ht = hts[ iidat.parent_index if iidat.parent_index >= 0 else ii-1 ]
        phi_ht = atree.HomogeneousTransform.zrot( phi )
        theta_ht = atree.HomogeneousTransform.xrot( -1 * iidat.theta )
        d_ht = atree.HomogeneousTransform.ztrans( iidat.d )
        #print( "parent_ht:" ); print( parent_ht );
        #print( "phi_ht:" ); print( phi_ht );
        #print( "theta_ht:" ); print( theta_ht );
        #print( "d_ht:" ); print( d_ht );
        hts[ ii ] = parent_ht * phi_ht * theta_ht * d_ht

    #print( "len(atomid_2_refold_index)", len(atomid_2_refold_index) )
    for ii, res in enumerate( residues ):
        for jj in range( res.coords.shape[0] ) :
            #print( "atomid_2_refold_index[", ii, "][", jj, "]",  atomid_2_refold_index[ ii ][ jj ] )
            res.coords[ jj ] = hts[ atomid_2_refold_index[ ii ][ jj ] ].frame[0:3,3]
        

def recurse_and_fill_atomtree_path_data( root_atom, tree_path_data ) :
    ''' Divide the atom tree into paths, where each atom chooses one child as part of its subpath, and the
    other children become roots of their own trees'''
    first = True
    childs_longest_subpath = 0
    child_w_longest_subpath = None
    for child in root_atom.children :
        recurse_and_fill_atomtree_path_data( child, tree_path_data ) 
        childs_subpath_length = tree_path_data[ child.atomid.res ][ child.atomid.atomno ].longest_subpath
        if first or childs_subpath_length > childs_longest_subpath :
            childs_longest_subpath = childs_subpath_length
            child_w_longest_subpath = child.atomid
            first = False
    if first :
        tree_path_data[ root_atom.atomid.res ][ root_atom.atomid.atomno ].longest_subpath = 1
    else :
        tree_path_data[ root_atom.atomid.res ][ root_atom.atomid.atomno ].longest_subpath = childs_longest_subpath + 1
        tree_path_data[ root_atom.atomid.res ][ root_atom.atomid.atomno ].child_on_subpath = child_w_longest_subpath
        for child in root_atom.children :
            tree_path_data[ child.atomid.res ][ child.atomid.atomno ].parent = root_atom.atomid
            if child.atomid != child_w_longest_subpath :
                tree_path_data[ child.atomid.res ][ child.atomid.atomno ].root_of_subpath = True


def dfs_identify_roots_and_depths( root_atom, depth, tree_path_data, root_list ) :
    if tree_path_data[ root_atom.atomid.res ][ root_atom.atomid.atomno ].root_of_subpath :
        root_list.append( (root_atom.atomid, depth) )
        depth += 1
    for child in root_atom.children :
        dfs_identify_roots_and_depths( child, depth, tree_path_data, root_list )

def sort_roots_by_depth( root_list ) :
    root_list.sort( key=lambda x : x[1] )

def create_ordered_root_set( root_list ) :
    ordered_roots = []
    if len( root_list ) == 0 :
        return
    ndepths = root_list[-1][1]+1
    ordered_roots = [ [] for x in range(ndepths) ]

    for root, depth in root_list :
        ordered_roots[ depth ].append( root )
    return ordered_roots
    
def renumber_atoms( tree_path_data, ordered_roots ) :
    natoms = sum( [ len(x) for x in tree_path_data ] )
    atomid_2_refold_index = [ [natoms] * len(x) for x in tree_path_data ] #natoms is out of bounds
    refold_index_2_atomid = [ None ] * natoms
    
    new_atid = 0
    for ii, roots in enumerate( ordered_roots ) :
        for jj, next_atom in enumerate( roots ) :
            while True :
                refold_index_2_atomid[ new_atid ] = next_atom
                atomid_2_refold_index[ next_atom.res ][ next_atom.atomno ] = new_atid
                new_atid += 1
                child_id = tree_path_data[ next_atom.res ][ next_atom.atomno ].child_on_subpath
                if child_id.res == -1 :
                    # we are done with this path
                    break
                next_atom = child_id

    # we should have hit every atom
    #print( "len(atomid_2_refold_index)", len(atomid_2_refold_index) )
    for ii in range( len( tree_path_data ) ) :
        for jj in range( len( tree_path_data[ ii ] ) ) :
            assert( atomid_2_refold_index[ ii ][ jj ] != natoms )
    
    return refold_index_2_atomid, atomid_2_refold_index

def number_torsions( tree ) :
    torsion_ids_and_offsets = [ [ None ] * len(x) for x in tree.atom_pointer_list ]
    torsion_ids_and_offsets[ tree.root.atomid.res ][ tree.root.atomid.atomno ] = ( -1, 0.0 )
    natoms = sum( [ len(x) for x in tree.atom_pointer_list ] )

    atom_ids_for_torsions = []
    torsion_ids_and_offsets[ tree.root.atomid.res ][ tree.root.atomid.atomno ] = ( 0, 0.0 )
    atom_ids_for_torsions.append( tree.root.atomid )
    torsion_index = number_torsions_recursive( tree.root, torsion_ids_and_offsets, 0, atom_ids_for_torsions )
    return atom_ids_for_torsions, torsion_ids_and_offsets
           


def number_torsions_recursive( root_node, torsion_ids_and_offsets, torsion_index, atom_ids_for_torsions ) :
    first = True
    accumulated_offset = 0.0
    for child in root_node.children :
        if first :
            atom_ids_for_torsions.append( child.atomid )
            torsion_index += 1
            first = False
            torsion_ids_and_offsets[ child.atomid.res ][ child.atomid.atomno ] = ( torsion_index, 0.0 )
        else :
            accumulated_offset += child.phi
            torsion_ids_and_offsets[ child.atomid.res ][ child.atomid.atomno ] = ( torsion_index, accumulated_offset )
    for child in root_node.children :
        torsion_index = number_torsions_recursive( child, torsion_ids_and_offsets, torsion_index, atom_ids_for_torsions )
    return torsion_index

    
def fill_atom_refold_data( tree, refold_index_2_atomid, atomid_2_refold_index, torsion_ids_and_offsets, tree_path_data ) :
    refold_data = [ None ] * len( refold_index_2_atomid )
    for ii, iiid in enumerate( refold_index_2_atomid ) :
        iinode = tree.node( iiid )
        torid, phi_offset = torsion_ids_and_offsets[ iiid.res ][ iiid.atomno ]
        ii_data = AtomRefoldData( phi_offset, iinode.theta, iinode.d, torid )
        if tree_path_data[ iiid.res ][ iiid.atomno ].root_of_subpath :
            parent_id = tree_path_data[ iiid.res ][ iiid.atomno ].parent
            if parent_id.res != -1 :
                ii_data.parent_index = atomid_2_refold_index[ parent_id.res ][ parent_id.atomno ]
        refold_data[ ii ] = ii_data
        #if ii == 0 :
        #    print( "root refold data:", ii_data, iinode.phi, iinode.theta, iinode.d )
    refold_data[ 0 ].parent_index = 0 # set the root as its own parent
    return refold_data

def torsions_from_tree( tree, atoms_for_controlling_torsions ) :
    return [ tree.node( atid ).phi for atid in atoms_for_controlling_torsions ]
        
    
