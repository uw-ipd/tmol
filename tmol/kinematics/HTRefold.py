import tmol.kinematics.AtomTree as atree
import attr
import typing
import math
import numpy

# module for defining a parallelizable refold method in terms of segmented scan of
# matrix multiplications. The first phase of implementation is getting the CPU-based
# version of this function working, which was originally written in C++

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold=numpy.nan)
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)


@attr.s(auto_attribs=True, slots=True)
class AtomTreePathData:
    longest_subpath: int = 0
    root_of_subpath: bool = False
    child_on_subpath: atree.AtomID = atree.AtomID()
    parent: atree.AtomID = atree.AtomID()
    depth: int = -1

@attr.s(auto_attribs=True, slots=True)
class AtomRefoldData:
    phi_offset: float = 0
    theta: float = 0
    d: float = 0
    jump_dofs: typing.Tuple[float, float, float, float, float, float
                            ] = attr.Factory(lambda: (0., 0., 0., 0., 0., 0.))
    is_jump: bool = False
    controlling_torsion: int = -1
    parent_index: int = -1
    depth: int = -1

@attr.s(auto_attribs=True, slots=True)
class WholeStructureRefoldData:
    ''' Represent the whole atom tree structured as numpy arrays and matrices so that
    a multi-pass segmented-scan algorithm can be used to compute the coordinates of
    a structure from a set of DOFs'''
    natoms: int = 0
    atomid_2_refold_index: typing.List[typing.List[int]] = None
    refold_index_2_atomid: typing.List[atree.AtomID] = None
    atomid_2_coalesced_ind: typing.List[typing.List[int]] = None
    coalesced_ind_2_atomid: typing.List[atree.AtomID] = None
    refold_index_2_coalesced_ind: numpy.array = None
    coalesced_ind_2_refold_index: numpy.array = None

    # data for constructing the HTs for bonded atoms
    dofs: numpy.matrix = None
    bonded_atoms: numpy.array = None
    bonded_atoms_pad: numpy.array = None
    remapped_phi: numpy.array = None
    n_sibling_phis: int = 0
    max_bonded_siblings: int = 0
    bonded_dof_remapping: numpy.array = None # map from
    bonded_atom_has_sibling: numpy.array = None # which bonded atoms (in refold order) have siblings
    which_phi_for_bonded_atom_w_sibling: numpy.array = None # what phi dofs in the remapped phi array go with which atoms in refold order
    is_eldest_child: numpy.array = None
    is_eldest_child_working: numpy.array = None
    cp: numpy.array = None # cos( phi )
    sp: numpy.array = None # sin( phi )
    ct: numpy.array = None # cos( theta )
    st: numpy.array = None # sin( theta )
    d: numpy.array = None # bond length

    # data for constructing the HTs for jump atoms
    jump_atoms: numpy.array = None
    jump_atoms_pad: numpy.array = None
    si: numpy.array = None # sin( z axis rotation )
    sj: numpy.array = None # sin( y axis rotation )
    sk: numpy.array = None # sin( x axis rotation )
    ci: numpy.array = None # cos( z axis rotation )
    cj: numpy.array = None # cos( y axis rotation )
    ck: numpy.array = None # cos( x axis rotation )
    cc: numpy.array = None
    cs: numpy.array = None
    sc: numpy.array = None
    ss: numpy.array = None
    Rdelta: numpy.matrix = None
    Rglobal: numpy.matrix = None

    is_root: numpy.array = None
    is_root_working: numpy.array = None
    parents: numpy.array = None
    hts: numpy.matrix = None
    ht_temps: numpy.matrix = None
    atom_range_for_depth: typing.List[typing.Tuple[int, int]] = None
    natoms_at_depth: typing.List[int] = None
    lookback_inds: numpy.array = None
    remapped_residue_inds: numpy.array = None # refold order 2 residue index
    remapped_atom_inds: numpy.array = None # refold order 2 atom index

# grr -- recursive data structures are a PITA in attr
@attr.s(auto_attribs=True, slots=True)
class DerivNode:
    pass


@attr.s(auto_attribs=True, slots=True)
class AbeGoID:
    atomid: atree.AtomID = attr.Factory(lambda: atree.AtomID())
    nodeid: int = 0


@attr.s(auto_attribs=True, slots=True)
class AbeGoNode(DerivNode):
    first_child: DerivNode = None
    other_children: typing.List[DerivNode] = attr.Factory(lambda: list())
    parent: DerivNode = None
    older_sibling: DerivNode = None
    younger_sibling: DerivNode = None
    phi_node: bool = False
    theta_d_node: bool = False
    jump_node: bool = False
    #atomid : atree.AtomID = attr.Factory( lambda : atree.AtomID() )
    reverse_depth: int = -1
    path_root_index: int = -1
    id: AbeGoID = attr.Factory(lambda: AbeGoID())


@attr.s(auto_attribs=True, slots=True)
class AbeGoPathRootData:
    node: AbeGoNode = None
    depth: int = -1
    path_length: int = -1


@attr.s(auto_attribs=True, slots=True)
class AbeGoRecursiveSummationData:
    node: AbeGoNode = None
    children: typing.List[int] = attr.Factory(lambda: list())


@attr.s(auto_attribs=True, slots=True)
class AbeGoDerivsumTree:
    natoms: int = 0
    nnodes: int = 0
    has_initial_f1f2: numpy.array = None
    atom_indices: numpy.array = None
    is_leaf: numpy.array = None
    is_leaf_working: numpy.array = None
    prior_children: numpy.matrix = None
    lookback_inds: numpy.array = None
    ndepths: int = 0
    atom_range_for_depth: list = None
    natoms_at_depth: numpy.array = None
    agid_2_dsi: typing.List[typing.List[typing.List[int]]] = None
    dsi_2_agid: typing.List[AbeGoID] = None


def initialize_ht_refold_data(residues, tree):
    tree_path_data = []
    natoms = 0

    tree_path_data = [[] for x in residues]
    for i in range(len(residues)):
        tree_path_data[i] = [AtomTreePathData() for x in residues[i].coords]
        natoms += len(residues[i].coords)

    recurse_and_fill_atomtree_path_data(tree.root, tree_path_data)
    # also mark the root of the atom tree as the root of a subpath, since that is not handled by
    # the recursive routine
    tree_path_data[tree.root.atomid.res][tree.root.atomid.atomno
                                         ].root_of_subpath = True


    # now identify all roots of the subpaths; the very shallowest path's depth is depth 0 because
    # python is an index-by-0 language
    root_list = []
    dfs_identify_roots_and_depths(tree.root, 0, tree_path_data, root_list)
    sort_roots_by_depth(root_list)

    ordered_roots = create_ordered_root_set(root_list)

    refold_index_2_atomid, atomid_2_refold_index = renumber_atoms(
        tree_path_data, ordered_roots
    )
    atoms_for_controlling_torsions, torsion_ids_and_offsets = number_torsions(
        tree
    )
    refold_data = fill_atom_refold_data(
        tree, refold_index_2_atomid, atomid_2_refold_index,
        torsion_ids_and_offsets, tree_path_data
    )

    return ordered_roots, refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index

def initialize_whole_structure_refold_data( residues, atom_tree ):
    ''' Create the WholeStructureRefoldData object full of the tensors needed to perform
    multi-layer segmented scans to compute the new atomic coordinates.
    Assumption: input vector of DOFs for refolding is a matrix of [natoms x 9]
    Assumption: the desired output tensor of coords is a matrix of [natoms x 3]
    Both input and output values are ordered s.t. the atoms for residue 0 are numbered
    in order from 0..n_0-1, and then the atoms for residue 1 are numbered in order from
    n_0..n_0+n_1-1, etc. Call this ordering/indexing of the atoms the coalesced ordering'''

    numpy.set_printoptions( precision=3, suppress=True )

    natoms = count_atom_tree_natoms( atom_tree )
    atomid_2_coalesced_ind, coalesced_ind_2_atomid = get_coalesced_ordering( residues )
    #print("coalesced_ind_2_atomid")
    #print(coalesced_ind_2_atomid)

    refold_data = WholeStructureRefoldData( natoms )
    ordered_roots, atom_refold_data, atoms_for_controlling_torsions, refold_index_2_atomid, atomid_2_refold_index = \
        initialize_ht_refold_data( residues, atom_tree )

    #print("atom_refold_data[756:759]");print(atom_refold_data[756:759])
    #print("refold_index_2_atomid"); print(refold_index_2_atomid);

    refold_data.refold_index_2_coalesced_ind, refold_data.coalesced_ind_2_refold_index = \
        get_coalesced_to_refold_mapping( \
        atomid_2_coalesced_ind, coalesced_ind_2_atomid, atomid_2_refold_index, refold_index_2_atomid )

    ba2aid, aid2ba, aids_of_eldest_siblings = create_bonded_atoms_with_siblings_order( atom_tree )
    num_bonded_atom_siblings = len(ba2aid)

    refold_data.hts = numpy.zeros( (natoms+1, 4, 4 ) )
    refold_data.hts[natoms] = numpy.eye(4)
    refold_data.ht_temps = refold_data.hts.copy()
    refold_data.dofs = numpy.zeros( (natoms, 9) )
    refold_data.bonded_atoms = numpy.fromiter((not atom_tree.atom_pointer_list[id.res][id.atomno].is_jump for id in refold_index_2_atomid ), dtype=bool)
    refold_data.bonded_atoms_pad = numpy.full((natoms+1), False, dtype=bool)
    refold_data.bonded_atoms_pad[:natoms] = refold_data.bonded_atoms
    num_bonded_atoms = numpy.sum( refold_data.bonded_atoms )
    refold_data.remapped_phi = numpy.zeros((num_bonded_atom_siblings))
    refold_data.n_sibling_phis = num_bonded_atom_siblings
    refold_data.max_bonded_siblings = count_max_bonded_atom_children( atom_tree )
    refold_data.bonded_dof_remapping = create_bonded_dof_remapping( ba2aid, atomid_2_coalesced_ind )
    refold_data.bonded_atom_has_sibling = create_bonded_atom_has_sibling_boolvect( aid2ba, refold_index_2_atomid )
    refold_data.which_phi_for_bonded_atom_w_sibling = create_refold_index_2_bonded_atom_mapping( aid2ba, refold_index_2_atomid )
    refold_data.is_eldest_child = create_eldest_child_array( ba2aid, aid2ba, aids_of_eldest_siblings )
    refold_data.is_eldest_child_working = refold_data.is_eldest_child.copy()
    refold_data.cp = numpy.zeros((num_bonded_atoms))
    refold_data.sp = numpy.zeros((num_bonded_atoms))
    refold_data.ct = numpy.zeros((num_bonded_atoms))
    refold_data.st = numpy.zeros((num_bonded_atoms))
    refold_data.d  = numpy.zeros((num_bonded_atoms))

    refold_data.jump_atoms = numpy.fromiter((atom_tree.atom_pointer_list[id.res][id.atomno].is_jump for id in refold_index_2_atomid ), dtype=bool)
    refold_data.jump_atoms_pad = numpy.full((natoms+1), False, dtype=bool)
    refold_data.jump_atoms_pad[:natoms] = refold_data.jump_atoms
    num_jump_atoms = numpy.sum(refold_data.jump_atoms)
    refold_data.si = numpy.zeros((num_jump_atoms))
    refold_data.sj = numpy.zeros((num_jump_atoms))
    refold_data.sk = numpy.zeros((num_jump_atoms))
    refold_data.ci = numpy.zeros((num_jump_atoms))
    refold_data.cj = numpy.zeros((num_jump_atoms))
    refold_data.ck = numpy.zeros((num_jump_atoms))
    refold_data.cc = numpy.zeros((num_jump_atoms))
    refold_data.cs = numpy.zeros((num_jump_atoms))
    refold_data.sc = numpy.zeros((num_jump_atoms))
    refold_data.ss = numpy.zeros((num_jump_atoms))
    refold_data.Rdelta  = numpy.zeros((num_jump_atoms,4,4))
    refold_data.Rglobal = numpy.zeros((num_jump_atoms,4,4))

    refold_data.is_root = numpy.full( (natoms+1), False, dtype=bool )
    refold_data.is_root[:natoms] = numpy.fromiter((atdat.parent_index != -1 for atdat in atom_refold_data), dtype=bool)
    refold_data.is_root[0] = True
    refold_data.is_root_working = numpy.full( (natoms+1), False, dtype=bool )
    
    refold_data.parents = numpy.full( (natoms+1), natoms, dtype=int )
    refold_data.parents[:natoms] = numpy.fromiter(((ard.parent_index if ard.parent_index != -1 else natoms) for ard in atom_refold_data), dtype=int)
    refold_data.parents[0] = natoms; # root is listed as its own parent?
    refold_data.atom_range_for_depth = determine_atom_ranges_for_depths(atom_refold_data)
    refold_data.natoms_at_depth = [ x[1]-x[0] for x in refold_data.atom_range_for_depth ]
    refold_data.lookback_inds = numpy.arange(natoms+1)
    refold_data.remapped_residue_inds = numpy.fromiter((id.res for id in refold_index_2_atomid),dtype=int)
    refold_data.remapped_atom_inds = numpy.fromiter((id.atomno for id in refold_index_2_atomid),dtype=int)

    return refold_data

def create_abe_go_f1f2sum_tree_for_structure(residues, atom_tree):
    abe_go_root, abe_go_nodes = abe_and_go_tree_from_atom_tree(atom_tree)
    atomid_2_atomindex, atomindex_2_atomid = create_atomid_mapping(residues)

    ag_path_root_nodes = create_abe_and_go_paths(abe_go_root)
    find_abe_go_path_depths(abe_go_root, ag_path_root_nodes)
    derivsum_index_2_ag_id, ag_id_2_derivsum_index, depth_start_inds = create_derivsum_indices(
        abe_go_nodes, ag_path_root_nodes
    )

    n_atoms = count_atom_tree_natoms(atom_tree)
    n_derivsum_nodes = count_abe_go_nodes(abe_go_nodes)
    max_branch = count_max_branch_node(abe_go_nodes)

    ag_tree = AbeGoDerivsumTree(n_atoms, n_derivsum_nodes)
    ag_tree.has_initial_f1f2 = numpy.zeros((n_derivsum_nodes + 1), dtype=bool)
    ag_tree.atom_indices = numpy.ones((n_derivsum_nodes + 1),
                                      dtype=numpy.int) * n_atoms
    ag_tree.is_leaf = numpy.zeros((n_derivsum_nodes), dtype=bool)
    ag_tree.is_leaf_working = numpy.zeros((n_derivsum_nodes), dtype=bool)
    ag_tree.prior_children = numpy.ones((n_derivsum_nodes + 1, max_branch),
                                        dtype=numpy.int) * n_atoms
    ag_tree.lookback_inds = numpy.arange(n_derivsum_nodes)
    ag_tree.ndepths = ag_path_root_nodes[abe_go_root.path_root_index].depth + 1
    ag_tree.atom_range_for_depth = [None] * ag_tree.ndepths
    ag_tree.natoms_at_depth = [0] * ag_tree.ndepths
    ag_tree.agid_2_dsi = ag_id_2_derivsum_index
    ag_tree.dsi_2_agid = derivsum_index_2_ag_id

    visit_all_abe_go_tree_nodes(abe_go_root, ag_tree, atomid_2_atomindex)

    # define the beginning and end points of the f1f2 sum depths
    sorted_path_roots = sorted(ag_path_root_nodes, key=lambda x: x.depth)
    last_depth = 0
    last_dsi = 0
    count_depths = 0
    for path_root in sorted_path_roots:
        if last_depth != path_root.depth:
            rid = leaf_of_derivsum_path(path_root.node).id
            dsi = ag_id_2_derivsum_index[rid.atomid.res][rid.atomid.atomno
                                                         ][rid.nodeid]
            ag_tree.atom_range_for_depth[count_depths] = (last_dsi, dsi)
            last_dsi = dsi
            count_depths += 1
            last_depth = path_root.depth
        ag_tree.natoms_at_depth[count_depths] += count_nodes_in_ag_path(
            path_root.node
        )
    ag_tree.atom_range_for_depth[count_depths] = (last_dsi, n_derivsum_nodes)

    assert (sum(ag_tree.natoms_at_depth) == n_derivsum_nodes)
    return ag_tree


# This function relies on the data that is prepared for the GPU's version of the refold, but
# is iterative in nature -- not going to be as efficient as Frank's version writen in numpy
def cpu_htrefold_1(
        residues, tree, refold_data, atoms_for_controlling_torsions,
        refold_index_2_atomid, atomid_2_refold_index
):
    torsions = torsions_from_tree(tree, atoms_for_controlling_torsions)
    hts = [atree.HomogeneousTransform() for x in refold_data]
    # root ht is in fact the identity transform
    for ii, iidat in enumerate(refold_data):
        iiatid = refold_index_2_atomid[ii]
        #print( ii, iidat.controlling_torsion, 180.0 / math.pi * ( torsions[ iidat.controlling_torsion ] if iidat.controlling_torsion >= 0 else 0.0 ), residues[ iiatid.res ].residue_type.atoms[ iiatid.atomno ].name )

        phi = iidat.phi_offset + (
            torsions[iidat.controlling_torsion]
            if iidat.controlling_torsion >= 0 else 0.0
        )
        parent_ht = hts[iidat.parent_index
                        if iidat.parent_index >= 0 else ii - 1]
        if iidat.is_jump:
            # TO DO!
            ht = atree.HomogeneousTransform()
            pass
        else:
            phi_ht = atree.HomogeneousTransform.xrot(phi)
            theta_ht = atree.HomogeneousTransform.zrot(iidat.theta)
            d_ht = atree.HomogeneousTransform.xtrans(iidat.d)
            #print( "phi", phi, "theta", iidat.theta, "d", iidat.d )
            ht = phi_ht * theta_ht * d_ht
            #print( "parent_ht:" ); print( parent_ht );
            #print( "phi_ht:" ); print( phi_ht );
            #print( "theta_ht:" ); print( theta_ht );
            #print( "d_ht:" ); print( d_ht );

        #print( ii, "ht before"); print( ht )
        #print( ii, "parent ht"); print( parent_ht )
        hts[ii] = parent_ht * ht
        #print( ii, "ht after"); print( ht )

    #print( "len(atomid_2_refold_index)", len(atomid_2_refold_index) )
    for ii, res in enumerate(residues):
        for jj in range(res.coords.shape[0]):
            #print( "atomid_2_refold_index[", ii, "][", jj, "]",  atomid_2_refold_index[ ii ][ jj ] )
            res.coords[jj] = hts[atomid_2_refold_index[ii][jj]].frame[0:3, 3]

def cpu_htrefold_2( dofs, refold_data, coords ):
    ''' Numpy version of the refold algorithm where segmented scan operations are performed on all of the
    atoms at the same depth'''

    #print("dofs"); print(dofs)

    natoms = refold_data.natoms
    hts = refold_data.hts
    ht_temps = refold_data.ht_temps
    compute_hts_for_bonded_atoms( dofs, refold_data )
    compute_hts_for_jump_atoms( refold_data )

    #print("ri2ci");print( refold_data.refold_index_2_coalesced_ind)
    #print("hts"); print(refold_data.hts)

    refold_data.is_root_working[:] = refold_data.is_root # in-place copy
    #print("refold_data.is_root"); print(refold_data.is_root)

    for ii, iirange in enumerate( refold_data.atom_range_for_depth ) :
        ii_view_ht = hts[iirange[0]:iirange[1]]
        ii_view_ht_temp = ht_temps[iirange[0]:iirange[1]]
        ii_parent = refold_data.parents[iirange[0]:iirange[1]]
        ii_is_root = refold_data.is_root_working[iirange[0]:iirange[1]]
        # initialize with my parent's transform multiplied by my transform from my parent
        #if ii==1:
        #    print( ii, "ii_parent" ); print( ii_parent )
        #    print( "iirange" ); print( iirange )
        #    print( "hts[ii_parent]" ); print( hts[ii_parent][-10:])
        #    print( "ii_view_ht" ); print( ii_view_ht[-10:] )
        ii_view_ht_temp[:] = numpy.matmul(hts[ii_parent],ii_view_ht)
        ii_view_ht[:] = ii_view_ht_temp
        ii_ind = refold_data.lookback_inds[:refold_data.natoms_at_depth[ii]]
        #print("ii_ind"); print( ii_ind )
        offset = 1
        #print("int(numpy.ceil(numpy.log2(ii_view_ht.shape[0])))", int(numpy.ceil(numpy.log2(ii_view_ht.shape[0]))) )
        for jj in range(int(numpy.ceil(numpy.log2(ii_view_ht.shape[0])))):
            #if ii==1:
            #    print( "ii, jj", ii, jj )
            #    print( "ii_view_ht" ); print( ii_view_ht[-10:] )
            #    print( "ii_is_root" ); print(ii_is_root[-10:])
            #    print( "(ii_ind >= offset) & (~ii_is_root)" ); print( ((ii_ind >= offset) & (~ii_is_root))[-10:]  )
            #    print( "ii_ind[(ii_ind >= offset) & (~ii_is_root) ] - offset" ); print( (ii_ind[(ii_ind >= offset) & (~ii_is_root) ] - offset )[-10:])
            ii_view_ht_temp[(ii_ind >= offset) & (~ii_is_root) ] = numpy.matmul( ii_view_ht[ ii_ind[(ii_ind >= offset) & (~ii_is_root) ] - offset ], ii_view_ht[ (ii_ind >= offset) & (~ii_is_root) ] )
            #if ii==1:
            #    print( "ii_view_ht_temp" ); print( ii_view_ht_temp[-10:])
            ii_view_ht[:] = ii_view_ht_temp
            ii_is_root[ii_ind >= offset] |= ii_is_root[ii_ind[ii_ind >= offset] - offset]
            offset *= 2

    #print( "hts final"); print( hts )
    coords[:] = hts[refold_data.coalesced_ind_2_refold_index,0:3,3]

def compute_hts_for_bonded_atoms( dofs_in, refold_data ):
    '''First scan the phi dofs for all the bonded-atom siblings, then construct the
    HTs for the bonded atoms'''
    D = 0
    THETA = 1
    PHI = 2

    # reorder the input dofs into the refold order
    #print("dofs_in");print(dofs_in)
    refold_data.dofs[:] = dofs_in[refold_data.refold_index_2_coalesced_ind]
    #print("refold_data.coalesced_ind_2_refold_index"); print(refold_data.coalesced_ind_2_refold_index)
    #print("refold_data.dofs");print(refold_data.dofs)

    refold_data.remapped_phi[:] = dofs_in[ refold_data.bonded_dof_remapping, PHI]
    refold_data.is_eldest_child_working[:] = refold_data.is_eldest_child
    is_eldest_child = refold_data.is_eldest_child_working
    inds = refold_data.lookback_inds[:refold_data.n_sibling_phis]
    offset = 1

    # segmented scan, but super short, since the segments are expected to be short (3 or 4)    
    for ii in range( int( numpy.ceil( numpy.log2( refold_data.max_bonded_siblings ) ) ) ) :
        refold_data.remapped_phi[(inds >= offset) & (~is_eldest_child)] += refold_data.remapped_phi[inds[(inds >= offset) & (~is_eldest_child)] - offset]
        is_eldest_child[inds >= offset] |= is_eldest_child[inds[inds>=offset]-offset]
        offset *= 2

    #now we'll remap these phis back into the per-atom phis
    #print("refold_data.which_phi_for_bonded_atom_w_sibling"); print(refold_data.which_phi_for_bonded_atom_w_sibling)
    refold_data.dofs[refold_data.bonded_atom_has_sibling,PHI] = refold_data.remapped_phi[refold_data.which_phi_for_bonded_atom_w_sibling]

    # now construct the hts -- code stolen from Frank
    cp = refold_data.cp
    sp = refold_data.sp
    ct = refold_data.ct
    st = refold_data.st
    d = refold_data.d

    cp[:] = numpy.cos(refold_data.dofs[refold_data.bonded_atoms,PHI])
    sp[:] = numpy.sin(refold_data.dofs[refold_data.bonded_atoms,PHI])
    ct[:] = numpy.cos(refold_data.dofs[refold_data.bonded_atoms,THETA])
    st[:] = numpy.sin(refold_data.dofs[refold_data.bonded_atoms,THETA])
    d[:] = refold_data.dofs[refold_data.bonded_atoms,D]

    hts = refold_data.hts

    hts[refold_data.bonded_atoms_pad,0,0] = ct
    hts[refold_data.bonded_atoms_pad,0,1] = -st
    hts[refold_data.bonded_atoms_pad,0,2] = 0
    hts[refold_data.bonded_atoms_pad,0,3] = d*ct
    hts[refold_data.bonded_atoms_pad,1,0] = cp*st
    hts[refold_data.bonded_atoms_pad,1,1] = cp*ct
    hts[refold_data.bonded_atoms_pad,1,2] = -sp
    hts[refold_data.bonded_atoms_pad,1,3] = d*cp*st
    hts[refold_data.bonded_atoms_pad,2,0] = sp*st
    hts[refold_data.bonded_atoms_pad,2,1] = sp*ct
    hts[refold_data.bonded_atoms_pad,2,2] = cp
    hts[refold_data.bonded_atoms_pad,2,3] = d*sp*st
    hts[refold_data.bonded_atoms_pad,3,0] = 0
    hts[refold_data.bonded_atoms_pad,3,1] = 0
    hts[refold_data.bonded_atoms_pad,3,2] = 0
    hts[refold_data.bonded_atoms_pad,3,3] = 1

    #print("root ht"); print(hts[0,:,:])

def compute_hts_for_jump_atoms(refold_data):
    # dofs_in should already have been copied over and remapped in the call
    # to compute_hts_for_bonded_atoms

    # create some aliases for existing arrays
    si = refold_data.si
    sj = refold_data.sj
    sk = refold_data.sk
    ci = refold_data.ci
    cj = refold_data.cj
    ck = refold_data.ck
    cc = refold_data.cc
    cs = refold_data.cs
    sc = refold_data.sc
    ss = refold_data.ss
    Rdelta = refold_data.Rdelta
    Rglobal = refold_data.Rglobal

    si[:] = numpy.sin(refold_data.dofs[refold_data.jump_atoms,3])
    sj[:] = numpy.sin(refold_data.dofs[refold_data.jump_atoms,4])
    sk[:] = numpy.sin(refold_data.dofs[refold_data.jump_atoms,5])
    ci[:] = numpy.cos(refold_data.dofs[refold_data.jump_atoms,3])
    cj[:] = numpy.cos(refold_data.dofs[refold_data.jump_atoms,4])
    ck[:] = numpy.cos(refold_data.dofs[refold_data.jump_atoms,5])
    cc[:] = ci*ck
    cs[:] = ci*sk
    sc[:] = si*ck
    ss[:] = si*sk
    Rdelta[:,0,0] = cj*ck
    Rdelta[:,0,1] = sj*sc-cs
    Rdelta[:,0,2] = sj*cc+ss
    Rdelta[:,0,3] = refold_data.dofs[refold_data.jump_atoms,0]
    Rdelta[:,1,0] = cj*sk
    Rdelta[:,1,1] = sj*ss+cc
    Rdelta[:,1,2] = sj*cs-sc
    Rdelta[:,1,3] = refold_data.dofs[refold_data.jump_atoms,1]
    Rdelta[:,2,0] = -sj
    Rdelta[:,2,1] = cj*si
    Rdelta[:,2,2] = cj*ci
    Rdelta[:,2,3] = refold_data.dofs[refold_data.jump_atoms,2]
    Rdelta[:,3,0] = 0
    Rdelta[:,3,1] = 0
    Rdelta[:,3,2] = 0
    Rdelta[:,3,3] = 1


    si = numpy.sin(refold_data.dofs[refold_data.jump_atoms,6])
    sj = numpy.sin(refold_data.dofs[refold_data.jump_atoms,7])
    sk = numpy.sin(refold_data.dofs[refold_data.jump_atoms,8])
    ci = numpy.cos(refold_data.dofs[refold_data.jump_atoms,6])
    cj = numpy.cos(refold_data.dofs[refold_data.jump_atoms,7])
    ck = numpy.cos(refold_data.dofs[refold_data.jump_atoms,8])
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    Rglobal[:,0,0] = cj*ck
    Rglobal[:,0,1] = sj*sc-cs
    Rglobal[:,0,2] = sj*cc+ss
    Rglobal[:,0,3] = 0
    Rglobal[:,1,0] = cj*sk
    Rglobal[:,1,1] = sj*ss+cc
    Rglobal[:,1,2] = sj*cs-sc
    Rglobal[:,1,3] = 0
    Rglobal[:,2,0] = -sj
    Rglobal[:,2,1] = cj*si
    Rglobal[:,2,2] = cj*ci
    Rglobal[:,2,3] = 0
    Rglobal[:,3,0] = 0
    Rglobal[:,3,1] = 0
    Rglobal[:,3,2] = 0
    Rglobal[:,3,3] = 1

    #print( "jump dofs"); print( refold_data.dofs[ refold_data.jump_atoms ] )
    #print( "refold_data.jump_atoms_pad" ); print(refold_data.jump_atoms_pad)
    #print( "jump hts" ); print( numpy.matmul( Rdelta, Rglobal ) )
    refold_data.hts[ refold_data.jump_atoms_pad,:,: ] = numpy.matmul( Rdelta, Rglobal )

def cpu_f1f2_summation1(atom_f1f2s, ag_derivsum_nodes):
    ''' Numpy version of the f1f2 recursive summation function which iterates across the abe-go
    nodes in the proper order up the tree (they have to already have been put in the correct order)
    and sums the f1f2 of the all the not-immediately prior children and the prior child'''
    f1f2sum = numpy.zeros((ag_derivsum_nodes.nnodes + 1, 6))
    #print( f1f2sum.shape )
    f1f2sum[:] = atom_f1f2s[ag_derivsum_nodes.atom_indices, :]
    #print( "f1f2sum start:" ); print( f1f2sum )
    for ii in range(ag_derivsum_nodes.nnodes):
        jj = ag_derivsum_nodes.prior_children[ii, :]
        #print( ii, "prior children" ); print( jj );
        #print( ii, "f1f2sum prior children" ); print( f1f2sum[ jj ] )
        #print( numpy.sum( f1f2sum[ jj[ jj != -1 ], : ], 1 ) )
        #print( numpy.sum( f1f2sum[ jj[ jj != -1 ], : ], 1 ).shape )
        #print( f1f2sum[ ii ] )
        f1f2sum[ii] += numpy.sum(f1f2sum[jj], 0).reshape(6)
        #print( ii, "f1f2sum[ ii ]" ); print( f1f2sum[ ii ] )
        if not ag_derivsum_nodes.is_leaf[ii]:
            f1f2sum[ii] += f1f2sum[ii - 1]
    return f1f2sum


# try and write a segmented-scan version in numpy
def cpu_f1f2_summation2(atom_f1f2s, ag_derivsum_nodes):
    ''' Numpy version of the f1f2 recursive summation function which uses something like a segmented
    scan over each of the abe-go depths.'''
    f1f2sum = numpy.zeros((ag_derivsum_nodes.nnodes + 1, 6))
    f1f2sum[:] = atom_f1f2s[ag_derivsum_nodes.atom_indices, :]
    ag_derivsum_nodes.is_leaf_working[:
                                      ] = ag_derivsum_nodes.is_leaf  # in-place copy

    for ii, iirange in enumerate(ag_derivsum_nodes.atom_range_for_depth):
        ii_view_f1f2 = f1f2sum[iirange[0]:iirange[1]]
        ii_children = ag_derivsum_nodes.prior_children[iirange[0]:iirange[1]]
        ii_view_f1f2 += numpy.sum(f1f2sum[ii_children], 1)
        ii_is_leaf = ag_derivsum_nodes.is_leaf_working[iirange[0]:iirange[1]]
        offset = 1
        ii_ind = ag_derivsum_nodes.lookback_inds[:ag_derivsum_nodes.
                                                 natoms_at_depth[ii]]
        for jj in range(int(numpy.ceil(numpy.log2(ii_view_f1f2.shape[0])))):
            ii_view_f1f2[(ii_ind >= offset) & (~ii_is_leaf)] += ii_view_f1f2[
                ii_ind[(ii_ind >= offset) & (~ii_is_leaf)] - offset
            ]
            ii_is_leaf[ii_ind >= offset
                       ] |= ii_is_leaf[ii_ind[ii_ind >= offset] - offset]
            offset *= 2

    return f1f2sum


def recurse_and_fill_atomtree_path_data(root_atom, tree_path_data):
    ''' Divide the atom tree into paths, where each atom chooses one child as part of its subpath, and the
    other children become roots of their own trees'''
    first = True
    childs_longest_subpath = 0
    child_w_longest_subpath = None
    for child in root_atom.children:
        recurse_and_fill_atomtree_path_data(child, tree_path_data)
        childs_subpath_length = tree_path_data[child.atomid.res][
            child.atomid.atomno
        ].longest_subpath
        if first or childs_subpath_length > childs_longest_subpath:
            childs_longest_subpath = childs_subpath_length
            child_w_longest_subpath = child.atomid
            first = False
    if first:
        tree_path_data[root_atom.atomid.res][root_atom.atomid.atomno
                                             ].longest_subpath = 1
    else:
        tree_path_data[root_atom.atomid.res][
            root_atom.atomid.atomno
        ].longest_subpath = childs_longest_subpath + 1
        tree_path_data[root_atom.atomid.res][
            root_atom.atomid.atomno
        ].child_on_subpath = child_w_longest_subpath
        for child in root_atom.children:
            tree_path_data[child.atomid.res][child.atomid.atomno
                                             ].parent = root_atom.atomid
            if child.atomid != child_w_longest_subpath:
                tree_path_data[child.atomid.res][child.atomid.atomno
                                                 ].root_of_subpath = True


def dfs_identify_roots_and_depths(root_atom, depth, tree_path_data, root_list):
    root_id = root_atom.atomid
    root_path_data = tree_path_data[root_id.res][root_id.atomno]
    if root_path_data.root_of_subpath:
        root_path_data.depth = depth
        root_list.append((root_atom.atomid, depth))
        depth += 1

    #print("root_path_data",root_path_data)

    # assign my depth to my child that's on the path, if I have one
    path_child_id = root_path_data.child_on_subpath
    if path_child_id.res != -1 :
        tree_path_data[ path_child_id.res ][ path_child_id.atomno ].depth = root_path_data.depth

    for child in root_atom.children:
        dfs_identify_roots_and_depths(child, depth, tree_path_data, root_list)


def sort_roots_by_depth(root_list):
    root_list.sort(key=lambda x: x[1])


def create_ordered_root_set(root_list):
    ordered_roots = []
    if len(root_list) == 0:
        return
    ndepths = root_list[-1][1] + 1
    ordered_roots = [[] for x in range(ndepths)]

    for root, depth in root_list:
        ordered_roots[depth].append(root)
    return ordered_roots


def renumber_atoms(tree_path_data, ordered_roots):
    natoms = sum([len(x) for x in tree_path_data])
    atomid_2_refold_index = [[natoms] * len(x)
                             for x in tree_path_data]  #natoms is out of bounds
    refold_index_2_atomid = [None] * natoms

    new_atid = 0
    for ii, roots in enumerate(ordered_roots):
        for jj, next_atom in enumerate(roots):
            while True:
                refold_index_2_atomid[new_atid] = next_atom
                atomid_2_refold_index[next_atom.res][next_atom.atomno
                                                     ] = new_atid
                new_atid += 1
                child_id = tree_path_data[next_atom.res][next_atom.atomno
                                                         ].child_on_subpath
                if child_id.res == -1:
                    # we are done with this path
                    break
                next_atom = child_id

    # we should have hit every atom
    for ii in range(len(tree_path_data)):
        for jj in range(len(tree_path_data[ii])):
            assert (atomid_2_refold_index[ii][jj] != natoms)

    return refold_index_2_atomid, atomid_2_refold_index


def number_torsions(tree):
    torsion_ids_and_offsets = [[None] * len(x) for x in tree.atom_pointer_list]
    torsion_ids_and_offsets[tree.root.atomid.res][tree.root.atomid.atomno
                                                  ] = (-1, 0.0)
    natoms = sum([len(x) for x in tree.atom_pointer_list])

    atom_ids_for_torsions = []
    torsion_ids_and_offsets[tree.root.atomid.res][tree.root.atomid.atomno
                                                  ] = (0, 0.0)
    atom_ids_for_torsions.append(tree.root.atomid)
    torsion_index = number_torsions_recursive(
        tree.root, torsion_ids_and_offsets, 0, atom_ids_for_torsions
    )
    return atom_ids_for_torsions, torsion_ids_and_offsets


def number_torsions_recursive(
        root_node, torsion_ids_and_offsets, torsion_index,
        atom_ids_for_torsions
):
    first = True
    accumulated_offset = 0.0
    for child in root_node.children:
        if child.is_jump : continue
        if first:
            atom_ids_for_torsions.append(child.atomid)
            torsion_index += 1
            first = False
            torsion_ids_and_offsets[child.atomid.res][child.atomid.atomno
                                                      ] = (torsion_index, 0.0)
        else:
            accumulated_offset += child.phi
            torsion_ids_and_offsets[child.atomid.res][
                child.atomid.atomno
            ] = (torsion_index, accumulated_offset)
    for child in root_node.children:
        torsion_index = number_torsions_recursive(
            child, torsion_ids_and_offsets, torsion_index,
            atom_ids_for_torsions
        )
    return torsion_index


def fill_atom_refold_data(
        tree, refold_index_2_atomid, atomid_2_refold_index,
        torsion_ids_and_offsets, tree_path_data
):
    refold_data = [None] * len(refold_index_2_atomid)
    for ii, iiid in enumerate(refold_index_2_atomid):
        iinode = tree.node(iiid)
        if iinode.is_jump :
            ii_data = AtomRefoldData()
            ii_data.is_jump = True
        else :
            torid, phi_offset = torsion_ids_and_offsets[iiid.res][iiid.atomno]
            ii_data = AtomRefoldData(phi_offset, iinode.theta, iinode.d, torid)
        if tree_path_data[iiid.res][iiid.atomno].root_of_subpath:
            parent_id = tree_path_data[iiid.res][iiid.atomno].parent
            if parent_id.res != -1:
                ii_data.parent_index = atomid_2_refold_index[parent_id.res][
                    parent_id.atomno
                ]
        ii_data.depth = tree_path_data[iiid.res][iiid.atomno].depth
        #print( ii, "ii_data", ii_data )
        refold_data[ii] = ii_data
        #if ii == 0 :
        #    print( "root refold data:", ii_data, iinode.phi, iinode.theta, iinode.d )
    refold_data[0].parent_index = 0  # set the root as its own parent
    return refold_data


def torsions_from_tree(tree, atoms_for_controlling_torsions):
    return [tree.node(atid).phi for atid in atoms_for_controlling_torsions]


def abe_and_go_tree_from_atom_tree(tree):
    nodes = [[list() for y in x] for x in tree.atom_pointer_list]
    ag_root = recursively_create_abe_and_go_tree(tree.root, nodes)
    return ag_root, nodes


def recursively_create_abe_and_go_tree(atom_tree_node, deriv_nodes):
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
    if atom_tree_node.is_jump:
        #deriv_node is the one that will be the parent to the children of atom_tree_node
        deriv_node = AbeGoNode()
        deriv_nodes[id.res][id.atomno].append(deriv_node)
        #deriv_node.atomid = id
        deriv_node.id.atomid = id
        deriv_node.id.nodeid = 0

        deriv_node.jump_node = True
        return_node = deriv_node
    else:
        # BondedAtom gets two nodes
        phi_node = AbeGoNode()
        deriv_nodes[id.res][id.atomno].append(phi_node)
        #phi_node.atomid = id
        phi_node.phi_node = True
        phi_node.id.atomid = id
        phi_node.id.nodeid = 0
        return_node = phi_node

        #deriv_node is the one that will be the parent to the children of atom_tree_node
        deriv_node = AbeGoNode()
        deriv_nodes[id.res][id.atomno].append(deriv_node)
        #deriv_node.atomid = id
        deriv_node.theta_d_node = True
        deriv_node.id.atomid = id
        deriv_node.id.nodeid = 1

        phi_node.first_child = deriv_node
        deriv_node.parent = phi_node

    # ok, now iterate of the atom_tree_node's children and for each child, establish its
    # connectivity with this node as the parent or its next-older sibling
    last_nonjump_child = None
    for child in atom_tree_node.children:
        abe_go_child = recursively_create_abe_and_go_tree(child, deriv_nodes)
        if deriv_node.first_child is None:
            deriv_node.first_child = abe_go_child
            abe_go_child.parent = deriv_node
        elif last_nonjump_child is None:
            deriv_node.other_children.append(abe_go_child)
            abe_go_child.parent = deriv_node
        else:  #if last_nonjump_child :
            last_nonjump_child.younger_sibling = abe_go_child
            abe_go_child.older_sibling = last_nonjump_child

        if not child.is_jump:
            last_nonjump_child = abe_go_child
        else:
            # all jump children should be at the front of the children list
            assert (last_nonjump_child is None)

    return return_node


def create_abe_and_go_paths(ag_root):
    ''' Define the list of root nodes that will then serve as starting
    points / ending points for the recursive summation of f1/f2s.'''
    ag_path_root_nodes = [AbeGoPathRootData(ag_root)]
    ag_root.path_root_index = 0
    recursively_identify_ag_tree_path_roots(ag_root, ag_path_root_nodes)
    return ag_path_root_nodes


def recursively_identify_ag_tree_path_roots(ag_root, ag_path_root_nodes):
    if ag_root is not None:
        recursively_identify_ag_tree_path_roots(
            ag_root.first_child, ag_path_root_nodes
        )
        for child in ag_root.other_children:
            child.path_root_index = len(ag_path_root_nodes)
            ag_path_root_nodes.append(AbeGoPathRootData(child))
            recursively_identify_ag_tree_path_roots(child, ag_path_root_nodes)
        if ag_root.younger_sibling:
            ag_root.younger_sibling.path_root_index = len(ag_path_root_nodes)
            ag_path_root_nodes.append(
                AbeGoPathRootData(ag_root.younger_sibling)
            )
            recursively_identify_ag_tree_path_roots(
                ag_root.younger_sibling, ag_path_root_nodes
            )


def find_abe_go_path_depths(ag_root, path_roots):
    '''Post-order traversal to identify the depths of my children and offshoots
    and then to record that depth in the path_roots list if I am a path root'''
    my_depth = 0
    if ag_root is not None:
        my_childs_depth = find_abe_go_path_depths(
            ag_root.first_child, path_roots
        )
        max_other_depths = -1
        for other_child in ag_root.other_children:
            other_depth = find_abe_go_path_depths(other_child, path_roots)
            if max_other_depths < other_depth:
                max_other_depths = other_depth
        if ag_root.younger_sibling:
            other_depth = find_abe_go_path_depths(
                ag_root.younger_sibling, path_roots
            )
            if max_other_depths < other_depth:
                max_other_depths = other_depth
        my_depth = my_childs_depth
        if max_other_depths >= 0:
            if my_depth < max_other_depths + 1:
                my_depth = max_other_depths + 1
        if ag_root.path_root_index >= 0:
            #print( "Saving depth", my_depth, "for node:", ag_root.id, "at position", ag_root.path_root_index, "in array of size", len(path_roots) )
            path_roots[ag_root.path_root_index].depth = my_depth
    return my_depth


def create_derivsum_indices(ag_nodes, path_roots):
    derivsum_index_2_ag_id = [None] * sum([
        sum([len(x) for x in y]) for y in ag_nodes
    ])
    ag_id_2_derivsum_index = [[[-1] * len(x) for x in y] for y in ag_nodes]
    sorted_path_roots = sorted(path_roots, key=lambda x: x.depth)

    ind = 0
    last_depth = sorted_path_roots[0].depth
    # root node is the "deepest" part of the tree and the root node is put into the
    # path_roots array first.
    depth_start_inds = [0] * (path_roots[0].depth + 1)
    for path_root in sorted_path_roots:
        if path_root.depth != last_depth:
            depth_start_inds[path_root.depth] = ind
            last_depth = path_root.depth
        ind = recursively_set_deriv_sum_indices(
            path_root.node, ind, derivsum_index_2_ag_id, ag_id_2_derivsum_index
        )
    return derivsum_index_2_ag_id, ag_id_2_derivsum_index, depth_start_inds


def recursively_set_deriv_sum_indices(node, ind, dsi2agi, agi2dsi):
    if node is not None:
        ind = recursively_set_deriv_sum_indices(
            node.first_child, ind, dsi2agi, agi2dsi
        )
        #print( "recusively setting derivsum index", node.id, ind )
        dsi2agi[ind] = node.id
        agi2dsi[node.id.atomid.res][node.id.atomid.atomno][node.id.nodeid
                                                           ] = ind
        ind += 1
    return ind


def create_atomid_mapping(residues):
    atomid_2_atomindex = [[] for res in residues]
    atomindex_2_atomid = [None
                          ] * sum([res.coords.shape[0] for res in residues])
    count = 0
    for ii, res in enumerate(residues):
        atomid_2_atomindex[ii] = [0] * res.coords.shape[0]
        for jj in range(res.coords.shape[0]):
            atomid_2_atomindex[ii][jj] = count
            atomindex_2_atomid[count] = atree.AtomID(ii, jj)
            count += 1
    return atomid_2_atomindex, atomindex_2_atomid


def count_atom_tree_natoms(atom_tree):
    return sum([len(res) for res in atom_tree.atom_pointer_list])


def count_abe_go_nodes(abe_go_nodes):
    return sum([
        sum([len(atnodes)
             for atnodes in residue_nodes])
        for residue_nodes in abe_go_nodes
    ])


def count_max_branch_node(abe_go_nodes):
    return max([
        max([
            max([
                len(n.other_children) + (1 if n.younger_sibling else 0)
                for n in atnodes
            ])
            for atnodes in residue
        ])
        for residue in abe_go_nodes
    ])


def visit_all_abe_go_tree_nodes(root, ag_tree, atomid_2_atomindex):
    dsi2ai, ai2dsi = ag_tree.dsi_2_agid, ag_tree.agid_2_dsi
    dsi = ai2dsi[root.id.atomid.res][root.id.atomid.atomno][root.id.nodeid]
    ag_tree.is_leaf[dsi] = root.first_child is None
    ag_tree.has_initial_f1f2[dsi] = True
    count_other_children = 0
    if root.younger_sibling:
        sibid = root.younger_sibling.id
        child_dsi = ai2dsi[sibid.atomid.res][sibid.atomid.atomno][sibid.nodeid]
        assert (child_dsi < dsi)
        ag_tree.prior_children[dsi, count_other_children] = child_dsi
        count_other_children += 1
    for child_node in root.other_children:
        childid = child_node.id
        child_dsi = aid2dsi[childid.atomid.res][childid.atomid.atomno
                                                ][childid.nodeid]
        ag_tree.prior_children[dsi, count_other_children] = child_dsi
        count_other_children += 1
    if root.theta_d_node or root.jump_node:
        ag_tree.atom_indices[dsi] = atomid_2_atomindex[root.id.atomid.res
                                                       ][root.id.atomid.atomno]

    if root.younger_sibling:
        visit_all_abe_go_tree_nodes(
            root.younger_sibling, ag_tree, atomid_2_atomindex
        )
    if root.first_child:
        visit_all_abe_go_tree_nodes(
            root.first_child, ag_tree, atomid_2_atomindex
        )
    for child in root.other_children:
        visit_all_abe_go_tree_nodes(child, ag_tree, atomid_2_atomindex)


def leaf_of_derivsum_path(root):
    return leaf_of_derivsum_path(
        root.first_child
    ) if root.first_child else root


def count_nodes_in_ag_path(root):
    count = 0
    if root:
        count = 1 + count_nodes_in_ag_path(root.first_child)
    return count

def get_coalesced_ordering(residues):
    natoms = sum( [ res.coords.shape[0] for res in residues ] )
    atomid_2_coalesced_ind = [ [0] * res.coords.shape[0] for res in residues ]
    coalesced_ind_2_atomid = [ None ] * natoms

    count = 0
    for ii, res in enumerate(residues):
        for jj in range(res.coords.shape[0]):
            coalesced_ind_2_atomid[count] = atree.AtomID(ii, jj)
            atomid_2_coalesced_ind[ii][jj] = count
            count += 1

    return atomid_2_coalesced_ind, coalesced_ind_2_atomid


def get_coalesced_to_refold_mapping( aid2ci, ci2aid, aid2ri, ri2aid ):
    ri2ci = numpy.zeros( (len(ci2aid)), dtype=int )
    ci2ri = numpy.zeros( (len(ci2aid)), dtype=int )

    for ii, res2ci in enumerate(aid2ci):
        for jj, ci in enumerate(res2ci):
            ri = aid2ri[ii][jj]
            ri2ci[ri] = ci
            ci2ri[ci] = ri

    #print( "ri2ci"); print( ri2ci[0:10] )
    #print( "ci2ri");print(ci2ri[0:10])

    return ri2ci, ci2ri

def create_bonded_atoms_with_siblings_order( atom_tree ):
    aid2ba = [ [-1] * len( resnodes ) for resnodes in atom_tree.atom_pointer_list ]
    aids_of_eldest_siblings = []
    n_bonded_atoms = recursively_identify_bonded_atoms_with_siblings_order( 0, atom_tree.root, aid2ba, aids_of_eldest_siblings )
    ba2aid = [ None ] * n_bonded_atoms
    for ii,res2ba in enumerate(aid2ba) :
        for jj,ba_index in enumerate(res2ba) :
            if ba_index == -1 : continue
            ba2aid[ba_index] = atree.AtomID(ii,jj)

    #print( "ba2aid" ); print( ba2aid )
    #print( "aid2ba" ); print( aid2ba )
    return ba2aid, aid2ba, aids_of_eldest_siblings

def recursively_identify_bonded_atoms_with_siblings_order( next_ind, root, aid2ba, aids_of_eldest_siblings ):
    first_ba_child = None
    for child in root.children:
        if not child.is_jump:
            if first_ba_child is None :
                first_ba_child = child
                aids_of_eldest_siblings.append( child.atomid )
            aid2ba[ child.atomid.res ][ child.atomid.atomno ] = next_ind
            next_ind += 1
    for child in root.children:
        next_ind = recursively_identify_bonded_atoms_with_siblings_order( next_ind, child, aid2ba, aids_of_eldest_siblings )
    return next_ind


def count_max_bonded_atom_children( atom_tree ) :
    max_children = 0
    for res_atoms in atom_tree.atom_pointer_list :
        for atom in res_atoms :
            nchildren = 0
            for child in atom.children :
                if not child.is_jump :
                    nchildren += 1
            if max_children < nchildren :
                max_children = nchildren
    return max_children


def create_bonded_dof_remapping( ba2aid, aid2ci ) :
    '''Create the array of coalesced indices for every bonded atom with a sibling so that an array
    of DOFs in coalesced can be read from for each bonded atom and put into bonded-atom order'''
    #print( "ba2aid" ); print( ba2aid )
    #print( "aid2ci" ); print( aid2ci )
    
    return numpy.fromiter( (aid2ci[aid.res][aid.atomno] for aid in ba2aid ), dtype=int )

    #ba2ci = numpy.zeros( (ba2aid.shape[0]) )
    #for ii, aid in enumerate( ba2aid ) :
    #    ba2ci[ ii ] = aid2ci[ aid.res ][ aid.atomno ]
    #return ba2ci

def create_bonded_atom_has_sibling_boolvect( aid2ba, ri2aid ):
    '''The boolean values for all atoms in refold order of whether or not they are
    bonded atoms with more than one sibling'''
    return numpy.fromiter( (aid2ba[id.res][id.atomno] != -1 for id in ri2aid), dtype=bool )

def create_refold_index_2_bonded_atom_mapping( aid2ba, ri2aid ) :
    ri2ba = numpy.fromiter( (aid2ba[id.res][id.atomno] for id in ri2aid ), dtype=int)
    ri2ba = ri2ba[ ri2ba != -1]
    #print("ri2ba");print(ri2ba)
    return ri2ba

def create_eldest_child_array( ba2aid, aid2ba, aids_of_eldest_siblings ) :
    '''For each of the atoms in bonded-atom-with-sibling order, note which of them is the eldest
    sibling; i.e. which is the beginning of a segment for segmented scan.'''
    is_eldest_child = numpy.full( (len(ba2aid)), False, dtype=bool )
    is_eldest_child[numpy.fromiter((aid2ba[id.res][id.atomno] for id in aids_of_eldest_siblings), dtype=int)] = True
    return is_eldest_child


def determine_atom_ranges_for_depths(atom_refold_data):
    ranges = []
    last_depth=0
    start_of_last_depth=0
    for ii,ard in enumerate(atom_refold_data):
        if ard.depth != last_depth:
            ranges.append((start_of_last_depth, ii))
            last_depth = ard.depth
            start_of_last_depth=ii
    ranges.append((start_of_last_depth,len(atom_refold_data)))
    return ranges
