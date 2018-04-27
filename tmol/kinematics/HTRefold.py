import tmol.kinematics.AtomTree as atree
import attr
import typing
import math
import numpy

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
    jump_dofs : typing.Tuple[ float, float, float, float, float, float ] = attr.Factory( lambda : ( 0., 0., 0., 0., 0., 0. ) )
    is_jump : bool = False
    controlling_torsion : int = -1
    parent_index : int = -1

# grr -- recursive data structures are a PITA in attr
@attr.s( auto_attribs = True, slots=True )
class DerivNode :
    pass

@attr.s( auto_attribs=True, slots=True )
class AbeGoID :
    atomid : atree.AtomID = attr.Factory( lambda: atree.AtomID() )
    nodeid : int = 0

@attr.s( auto_attribs=True, slots=True )
class AbeGoNode( DerivNode ) :
    first_child : DerivNode = None
    other_children : typing.List[ DerivNode ] = attr.Factory( lambda: list() )
    parent : DerivNode = None
    older_sibling : DerivNode = None
    younger_sibling : DerivNode = None
    phi_node : bool = False
    theta_d_node : bool = False
    jump_node : bool = False
    #atomid : atree.AtomID = attr.Factory( lambda : atree.AtomID() )
    reverse_depth : int = -1
    path_root_index : int = -1
    id: AbeGoID = attr.Factory( lambda: AbeGoID() )

@attr.s( auto_attribs=True, slots=True )
class AbeGoPathRootData :
    node : AbeGoNode = None
    depth : int = -1
    path_length : int = -1

class AbeGoRecursiveSummationData :
    node : AbeGoNode = None
    children : typing.List[ int ] = attr.Factory( lambda: list() )
    


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


# This function relies on the data that is prepared for the GPU's version of the refold, but
# is iterative in nature -- not going to be as efficient as Frank's version writen in numpy
def cpu_htrefold( residues, tree, refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index ) :
    torsions = torsions_from_tree( tree, atoms_for_controlling_torsions )
    hts = [ atree.HomogeneousTransform() for x in refold_data ]
    # root ht is in fact the identity transform
    for ii, iidat in enumerate( refold_data ) :
        iiatid = refold_index_2_atomid[ ii ]
        #print( ii, iidat.controlling_torsion, 180.0 / math.pi * ( torsions[ iidat.controlling_torsion ] if iidat.controlling_torsion >= 0 else 0.0 ), residues[ iiatid.res ].residue_type.atoms[ iiatid.atomno ].name )

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

def cpu_f1f2_summation( atom_f1f2s, ag_derivsum_nodes ) :
    f1f2sum = numpy.zeros( (ag_derivsum_nodes.nnodes, 6) )
    print( f1f2sum.shape )
    f1f2sum[ ag_derivsum_nodes.has_initial_f1f2, : ] = atom_f1f2s[ ag_derivsum_nodes.atom_indices[ ag_derivsum_nodes.atom_indices != -1 ], : ]
    for ii in range( ag_derivsum_nodes.nnodes ) :
        jj = ag_derivsum_nodes.prior_children[ ii, : ]
        print( numpy.sum( f1f2sum[ jj[ jj != -1 ], : ], 1 ) )
        print( numpy.sum( f1f2sum[ jj[ jj != -1 ], : ], 1 ).shape )
        print( f1f2sum[ ii ] )
        f1f2sum[ ii ] += numpy.sum( f1f2sum[ jj[ jj != -1 ], : ], 1 ).reshape(6)
        if not ag_derivsum_nodes.is_leaf[ ii ] :
            f1f2sum[ ii ] += f1f2sum[ ii-1 ]
    return f1f2sum
    


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


def abe_and_go_tree_from_atom_tree( tree ) :
    nodes = [ [ list() for y in x ] for x in tree.atom_pointer_list ]
    ag_root = recursively_create_abe_and_go_tree( tree.root, nodes )
    return ag_root, nodes

def recursively_create_abe_and_go_tree( atom_tree_node, deriv_nodes ) :
    '''
    Define the tree used to sum f1/f2 vectors recursively from the leaves toward the root.
    The Tree has two nodes per BondedAtom: one for the d and theta DOFs, which
    only accumulate f1/f2s from the atom's children, and the second for the phi dihedral,
    which accumulates f1/f2s from the atom's younger siblings also. JumpAtoms have only a
    single node in this tree.

    Each atom either has an older sibling or a parent, but not both.
    An atom may have both a younger sibling and children.

    The first child is on the same summation path that the atom_tree_node is
    on. The other children are roots of other summation paths.

    The other_children list constructed for  will be empty unless atom_tree_node has
    JumpAtom children; due to their simple summation directly up the tree, JumpAtom
    children do not have the next-younger-sibling structure that BondedAtoms do, and
    thus there may be multiple jump children for a single node in addition to at most
    one bonded-atom child -- any additional bonded-atom children will end up in this tree
    as younger siblings of their next older siblings.
    '''
    id = atom_tree_node.atomid
    if atom_tree_node.is_jump :
        #deriv_node is the one that will be the parent to the children of atom_tree_node
        deriv_node = AbeGoNode()
        deriv_nodes[ id.res ][ id.atomno ].append( deriv_node )
        #deriv_node.atomid = id
        deriv_node.id.atomid = id
        deriv_node.id.nodeid = 0

        deriv_node.jump_node = True
        return_node = deriv_node
    else :
        # BondedAtom gets two nodes
        phi_node = AbeGoNode()
        deriv_nodes[ id.res ][ id.atomno ].append( phi_node )
        #phi_node.atomid = id
        phi_node.phi_node = True
        phi_node.id.atomid = id
        phi_node.id.nodeid = 0
        return_node = phi_node

        #deriv_node is the one that will be the parent to the children of atom_tree_node
        deriv_node = AbeGoNode()
        deriv_nodes[ id.res ][ id.atomno ].append( deriv_node )
        #deriv_node.atomid = id
        deriv_node.theta_d_node = True
        deriv_node.id.atomid = id
        deriv_node.id.nodeid = 1

        phi_node.first_child = deriv_node
        deriv_node.parent = phi_node

    # ok, now iterate of the atom_tree_node's children and for each child, establish its
    # connectivity with this node as the parent or its next-older sibling
    last_nonjump_child = None
    for child in atom_tree_node.children :
        abe_go_child = recursively_create_abe_and_go_tree( child, deriv_nodes )
        if deriv_node.first_child is None :
            deriv_node.first_child = abe_go_child
            abe_go_child.parent = deriv_node
        elif last_nonjump_child is None  :
            deriv_node.other_children.append( abe_go_child )
            abe_go_child.parent = deriv_node
        else : #if last_nonjump_child :
            last_nonjump_child.younger_sibling = abe_go_child
            abe_go_child.older_sibling = last_nonjump_child

        if not child.is_jump :
            last_nonjump_child = abe_go_child
        else :
            # all jump children should be at the front of the children list
            assert( last_nonjump_child is None )

    return return_node

def create_abe_and_go_paths( ag_root ) :
    ''' Define the list of root nodes that will then serve as starting
    points / ending points for the recursive summation of f1/f2s.'''
    ag_path_root_nodes = [ AbeGoPathRootData( ag_root ) ]
    ag_root.path_root_index = 0
    recursively_identify_ag_tree_path_roots( ag_root, ag_path_root_nodes )
    return ag_path_root_nodes

def recursively_identify_ag_tree_path_roots( ag_root, ag_path_root_nodes ) :
    if ag_root is not None :
        recursively_identify_ag_tree_path_roots( ag_root.first_child, ag_path_root_nodes )
        for child in ag_root.other_children :
            child.path_root_index = len( ag_path_root_nodes )
            ag_path_root_nodes.append( AbeGoPathRootData( child ) )
            recursively_identify_ag_tree_path_roots( child, ag_path_root_nodes )
        if ag_root.younger_sibling :
            ag_root.younger_sibling.path_root_index = len( ag_path_root_nodes )
            ag_path_root_nodes.append( AbeGoPathRootData( ag_root.younger_sibling ) )
            recursively_identify_ag_tree_path_roots( ag_root.younger_sibling, ag_path_root_nodes )

def find_abe_go_path_depths( ag_root, path_roots ) :
    '''Post-order traversal to identify the depths of my children and offshoots
    and then to record that depth in the path_roots list if I am a path root'''
    my_depth = 0
    if ag_root is not None :
        my_childs_depth = find_abe_go_path_depths( ag_root.first_child, path_roots )
        max_other_depths = -1
        for other_child in ag_root.other_children :
            other_depth = find_abe_go_path_depths( other_child, path_roots )
            if max_other_depths < other_depth :
                max_other_depths = other_depth
        if ag_root.younger_sibling :
            other_depth = find_abe_go_path_depths( ag_root.younger_sibling, path_roots )
            if max_other_depths < other_depth :
                max_other_depths = other_depth
        my_depth = my_childs_depth
        if max_other_depths >= 0 :
            if my_depth < max_other_depths + 1 :
                my_depth = max_other_depths + 1
        if ag_root.path_root_index >= 0 :
            #print( "Saving depth", my_depth, "for node:", ag_root.id, "at position", ag_root.path_root_index, "in array of size", len(path_roots) )
            path_roots[ ag_root.path_root_index ].depth = my_depth
    return my_depth


def create_derivsum_indices( ag_nodes, path_roots ) :
    derivsum_index_2_ag_id = [ None ] * sum([sum([len(x) for x in y ]) for y in ag_nodes ])
    ag_id_2_derivsum_index = [ [ [-1] * len(x) for x in y ] for y in ag_nodes ]
    sorted_path_roots = sorted( path_roots, key=lambda x : x.depth )

    ind = 0
    last_depth = sorted_path_roots[ 0 ].depth
    # root node is the "deepest" part of the tree and the root node is put into the
    # path_roots array first.
    depth_start_inds = [ 0 ] * (path_roots[0].depth+1)
    for path_root in sorted_path_roots :
        if path_root.depth != last_depth :
            depth_start_inds[ path_root.depth ] = ind
            last_depth = path_root.depth
        ind = recursively_set_deriv_sum_indices( path_root.node, ind, derivsum_index_2_ag_id, ag_id_2_derivsum_index )
    return derivsum_index_2_ag_id, ag_id_2_derivsum_index, depth_start_inds

def recursively_set_deriv_sum_indices( node, ind, dsi2agi, agi2dsi ) :
    if node is not None :
        ind = recursively_set_deriv_sum_indices( node.first_child, ind, dsi2agi, agi2dsi )
        dsi2agi[ ind ] = node.id
        agi2dsi[ node.id.atomid.res ][ node.id.atomid.atomno ][ node.id.nodeid ] = ind
        ind += 1
    return ind

def create_f1f2_path_segments( ag_root, path_roots ) :
    pass
