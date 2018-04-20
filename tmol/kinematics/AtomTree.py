import typing
import attr
import numpy
import math
import tmol.system.residue.restypes

class HomogeneousTransform :

    @staticmethod
    def from_three_coords( p1, p2, p3 ) :
        return HomogeneousTransform( p1=p1, p2=p2, p3=p3 )

    def from_four_coords( c, p1, p2, p3 ) :
        return HomogeneousTransform( p1=p1, p2=p2, p3=p3, c=c )

    @staticmethod
    def from_frame( frame ) :
        return HomogeneousTransform( input_frame=frame )

    @staticmethod
    def xrot( theta ) :
        ''' construct frame representing rotation about the x axis by theta radians'''
        ht = HomogeneousTransform()
        ht.set_rotation_x( theta )
        return ht

    @staticmethod
    def yrot( theta ) :
        ''' construct frame representing rotation about the y axis by theta radians'''
        ht = HomogeneousTransform()
        ht.set_rotation_y( theta )
        return ht

    @staticmethod
    def zrot( theta ) :
        ''' construct frame representing rotation about the z axis by theta radians'''
        ht = HomogeneousTransform()
        ht.set_rotation_z( theta )
        return ht

    @staticmethod
    def xtrans( d ) :
        ''' construct frame representing translation along the x axis by d distance'''
        ht = HomogeneousTransform()
        ht.set_translation_x( d )
        return ht

    @staticmethod
    def ytrans( d ) :
        ''' construct frame representing translation along the y axis by d distance'''
        ht = HomogeneousTransform()
        ht.set_translation_y( d )
        return ht

    @staticmethod
    def ztrans( d ) :
        ''' construct frame representing translation along the z axis by d distance'''
        ht = HomogeneousTransform()
        ht.set_translation_z( d )
        return ht

    def __init__( self, input_frame=None, p1=None, p2=None, p3=None, c=None ) :
        if input_frame is not None :
            assert input_frame.shape[ 0 ] == 4
            assert input_frame.shape[ 1 ] == 4
            self.frame = input_frame
        elif p1 is not None and p2 is not None and p3 is not None :
            # define a right handed coordinate frame located at either p2 or c
            # with the x axis pointing at p1 from p2, the y axis in the p1, p2, p3 plane
            # and p3 orthogonal to x and y
            assert len(p1) == 3
            assert len(p2) == 3
            assert len(p3) == 3
            self.frame = numpy.eye(4)
            v12 = p1 - p2
            xaxis = v12/numpy.linalg.norm( v12 )
            self.frame[0:3,0] = xaxis
            v31 = p3 - p1
            zaxis = numpy.cross( xaxis, v31 )
            zaxis = zaxis/numpy.linalg.norm(zaxis)
            self.frame[0:3,2] = zaxis
            yaxis = numpy.cross( zaxis, xaxis )
            self.frame[0:3,1] = yaxis
            if c is not None :
                self.frame[0:3,3] = c
            else :
                self.frame[0:3,3] = p2
        else :
            self.frame = numpy.eye(4)
        #print("frame"); print( self.frame )

    def set_rotation_x( self, theta ) :
        '''Set a rotation about the x axis; theta should be in radians'''
        self.frame.fill(0)
        ct = math.cos( theta )
        st = math.sin( theta )
        self.frame[0,0] = 1
        self.frame[1,1] = ct
        self.frame[2,2] = ct
        self.frame[1,2] = -st
        self.frame[2,1] = st
        self.frame[3,3] = 1

    def set_rotation_y( self, theta ) :
        '''Set a rotation about the y axis; theta should be in radians'''
        self.frame.fill(0)
        ct = math.cos( theta )
        st = math.sin( theta )
        self.frame[1,1] = 1
        self.frame[3,3] = 1
        self.frame[2,2] = ct
        self.frame[0,0] = ct
        self.frame[0,2] = st
        self.frame[2,0] = -st

    def set_rotation_z( self, theta ) :
        '''Set a rotation about the z axis; theta should be in radians'''
        self.frame.fill(0)
        ct = math.cos( theta )
        st = math.sin( theta )
        self.frame[2,2] = 1
        self.frame[3,3] = 1
        self.frame[0,0] = ct
        self.frame[1,1] = ct
        self.frame[0,1] = -st
        self.frame[1,0] = st

    def set_translation_x( self, dist ) :
        self.frame.fill( 0 )
        self.frame[0,0] = 1
        self.frame[1,1] = 1
        self.frame[2,2] = 1
        self.frame[3,3] = 1
        self.frame[0,3] = dist

    def set_translation_y( self, dist ) :
        self.frame.fill( 0 )
        self.frame[0,0] = 1
        self.frame[1,1] = 1
        self.frame[2,2] = 1
        self.frame[3,3] = 1
        self.frame[1,3] = dist

    def set_translation_z( self, dist ) :
        self.frame.fill( 0 )
        self.frame[0,0] = 1
        self.frame[1,1] = 1
        self.frame[2,2] = 1
        self.frame[3,3] = 1
        self.frame[2,3] = dist

    def __mul__( self, other ) :
        ht_res = HomogeneousTransform()
        ht_res.frame = numpy.matmul(self.frame, other.frame)
        return ht_res

    def __imul__( self, other ) :
        newframe = numpy.matmul( self.frame, other.frame )
        self.frame = newframe
        return self

    def __str__( self ) :
        buff = []
        buff.append( " ".join( [ "%10.3f" % x for x in self.frame[0,:] ] ) )
        buff.append( " ".join( [ "%10.3f" % x for x in self.frame[1,:] ] ) )
        buff.append( " ".join( [ "%10.3f" % x for x in self.frame[2,:] ] ) )
        buff.append( " ".join( [ "%10.3f" % x for x in self.frame[3,:] ] ) )
        return "\n".join(buff)

    def inverse( self ) :
        ht_res = HomogeneousTransform()
        ht_res.frame[0:3,0:3] = self.frame[0:3,0:3].transpose()
        ht_res.frame[0:3,3] = -1 * numpy.dot(self.frame[0:3,3],self.frame[0:3,0:3])
        return ht_res

@attr.s( auto_attribs=True, slots=True )
class AtomID :
    res :    int = -1
    atomno : int = -1

class TreeAtom :
    def update_xyz( self, parent_ht = None ) :
        raise NotImplementedError;

    def update_internal_coords( self, parent_ht = None ) :
        raise NotImplementedError;

@attr.s( auto_attribs=True, slots=True )
class BondedAtom( TreeAtom ) :
    atomid: AtomID = attr.Factory( AtomID )
    parent : TreeAtom = None
    phi : float = 0
    theta : float = 0
    d : float = 0
    xyz : typing.Tuple[ float, float, float ] = ( 0, 0, 0 )
    children : typing.List[ TreeAtom ] = attr.Factory(list)
    is_jump : bool = False

    def update_xyz( self, parent_ht = None ) :
        if parent_ht is None :
            parent_ht = HomogeneousTransform() # identity
        elif self.atomid.res == 1 :
            print( "Update XYZ", self.atomid )
            print( "parent ht:"); print( parent_ht );
        dihedral_rotation_ht = HomogeneousTransform.xrot( self.phi )
        # modify the parent_ht with the dihedral rotation here
        parent_ht *= dihedral_rotation_ht
        bond_angle_rotation_ht = HomogeneousTransform.zrot( self.theta )
        trans_ht = HomogeneousTransform.xtrans( self.d )
        new_ht = parent_ht * bond_angle_rotation_ht * trans_ht;
        self.xyz = new_ht.frame[0:3,3]
        for child in self.children :
            child.update_xyz( new_ht )


    def update_internal_coords( self, parent_ht = None ) :
        if parent_ht is None :
            parent_ht = HomogeneousTransform() # identity
        #print( "parent_ht" ); print( parent_ht )
        w = numpy.array( self.xyz ) - numpy.array( parent_ht.frame[0:3,3] )
        self.d = numpy.linalg.norm( w )
        if self.d <= 1e-6 :
            self.d = 0
            self.theta = 0
            self.phi = 0
        else :
            w /= self.d
            xdot = numpy.dot( w, parent_ht.frame[0:3,0] )
            # hacky version with coordinates in general position!
            # look at BondedAtom.cc for the three cases dependent on x's value (aka xdot here)
            if xdot >= 1 - 1e-6 :
                self.theta = 0
                self.phi = 0
            elif xdot < -1 + 1e-6 :
                self.theta = pi
                self.phi = 0
            else :
                self.theta = math.acos( xdot )
                # finally, we need the rotation about the x axis so that the coordinate will be in
                # the xy plane
                ydot = numpy.dot( w, parent_ht.frame[0:3,1] )
                zdot = numpy.dot( w, parent_ht.frame[0:3,2] )
                self.phi = math.atan2( zdot, ydot )

        if self.atomid.res == 1 :
            print( "Update internal coords" )
            print( "Atom: ", self.atomid )
            print( "parent ht:"); print( parent_ht );
            print( "d", self.d, "theta", self.theta, "phi", self.phi )
        # modify the parent ht so that younger siblings will have their offset phi readily
        # calculated
        parent_ht *= HomogeneousTransform.xrot( self.phi )
        #print( "parent_ht(again)" ); print( parent_ht )
        new_ht = parent_ht * HomogeneousTransform.zrot( self.theta ) * HomogeneousTransform.xtrans( self.d )

        #print( "d", self.d, "theta", self.theta, "phi", self.phi )

        for child in self.children :
            child.update_internal_coords( new_ht )

@attr.s( auto_attribs=True, slots=True )
class JumpAtom( TreeAtom ) :
    atomid: AtomID = attr.Factory( AtomID )
    parent: TreeAtom = None
    rb : typing.List[ float ] = [ 0, 0, 0 ]
    rot_delta : typing.List[ float ] = [ 0, 0, 0 ]
    rot : typing.List[ float ] = [ 0, 0, 0 ]
    xyz : typing.Tuple[ float, float, float ] = ( 0, 0, 0 )
    children : typing.List[ TreeAtom ] = attr.Factory(list)
    is_jump : bool = True
    
    def update_xyz( self, parent_ht = None ) :
        if parent_ht is None :
            parent_ht = HomogeneousTransform() # identity transform

        ht_deltas = HomogeneousTransform.zrot(self.rot_delta[2]) * \
            HomogeneousTransform.yrot(self.rot_delta[1]) * \
            HomogeneousTransform.xrot(self.rot_delta[0] ) * \
            HomogeneousTransform.xtrans(self.rb[0]) * \
            HomogeneousTransform.ytrans(self.rb[1]) * \
            HomogeneousTransform.ztrans(self.rb[2])

        ht_r_global = HomogeneousTransform.zrot(self.rot[2] ) * \
            HomogeneousTransform.yrot(self.rot[1] ) * \
            HomogeneousTransform.xrot(self.rot[0] )

        ht = parent_ht * ht_deltas * ht_r_global
        xyz = ht.frame[0:3,3]
        print("jump ht:" ); print(ht)
        for child in self.children :
            child.update_xyz( ht )

    def update_internal_coords( self, parent_ht = None ) :        
        if parent_ht is None :
            parent_ht = HomogeneousTransform()
        self.rot_delta[0] = 0; self.rot_delta[1] = 0; self.rot_delta[2] = 0;
        # build the HT at the downstream atoms:
        dest = self.get_ht_from_childrens_coords()
        transform = parent_ht.inverse() * dest;
        self.rb = transform.frame[0:3,3]
        
        df = transform.frame
        # code below lifted directly from Frank
        cys = numpy.sqrt(df[0,0]*df[0,0] + df[1,0]*df[1,0])
        print( "cys?", cys )
        if cys < 4*numpy.finfo(float).eps  :
            self.rot[0] = numpy.arctan2( -df[1,2], df[1,1] )
            self.rot[1] = numpy.arctan2( -df[2,0], cys     )
            self.rot[2] = 0;
        else :
            self.rot[0] = numpy.arctan2(  df[2,1], df[2,2] )
            self.rot[1] = numpy.arctan2( -df[2,0], cys     )
            self.rot[2] = numpy.arctan2(  df[1,0], df[0,0] )


        # TEMP -- let's make sure that the dofs we just measured give us back
        # the ht we are looking for:
        ht_deltas = HomogeneousTransform.zrot(self.rot_delta[2]) * \
            HomogeneousTransform.yrot(self.rot_delta[1]) * \
            HomogeneousTransform.xrot(self.rot_delta[0] ) * \
            HomogeneousTransform.xtrans(self.rb[0]) * \
            HomogeneousTransform.ytrans(self.rb[1]) * \
            HomogeneousTransform.ztrans(self.rb[2])

        ht_r_global = HomogeneousTransform.zrot(self.rot[2] ) * \
            HomogeneousTransform.yrot(self.rot[1] ) * \
            HomogeneousTransform.xrot(self.rot[0] )

        ht = parent_ht * ht_deltas * ht_r_global
        print( "Update internal coords", self.atomid )
        print( "target transform" )
        print( transform )
        print( "obtained transform" )
        print( ht_deltas * ht_r_global )
        print( "ht dest", self.atomid )
        print( dest )
        print( "self.rb", self.rb, "self.rot", self.rot )
        print( "ht from dofs:" )
        print( ht )
        print( "parent times parent inv?" )
        print( parent_ht * parent_ht.inverse() )


        for child in self.children :
            child.update_internal_coords( dest )
    
    def get_ht_from_childrens_coords( self ) :
        if self.stub_defined() :
            p1 = self.xyz
            nonjump_children = [ x for x in self.children if not x.is_jump ]
            p2 = nonjump_children[0].xyz
            p3 = nonjump_children[1].xyz if len( nonjump_children ) > 1 else [ child.xyz for child in nonjump_children[0].children if not child.is_jump ][0]
            return HomogeneousTransform.from_four_coords( self.xyz, p2, p1, p3 )

    def stub_defined( self ) :
        count_non_jump = 0
        for child in self.children :
            if not child.is_jump :
                count_non_jump += 1
        if count_non_jump < 2 :
            for child in self.children :
                if not child.is_jump :
                    for childs_child in child.children :
                        if not childs_child.is_jump :
                            count_non_jump += 1
        return count_non_jump >= 2

@attr.s( auto_attribs=True, slots=True )
class AtomTree :
    root : TreeAtom
    atom_pointer_list : typing.List[ typing.List[ TreeAtom ] ]
    def update_xyz( self, ) :
        self.root.update_xyz()
    def node( self, atid ) :
        return self.atom_pointer_list[ atid.res ][ atid.atomno ]

def create_links_simple( residue_type, links ) :
    
    for i in range(len(residue_type.bond_indices)) :
        # don't add cut-bond pairs
        at1,at2 = residue_type.bond_indices[i,:]
        if (at1,at2) in residue_type.cutbond : continue
        if (at2,at1) in residue_type.cutbond : continue
        links[ at1 ].append( at2 )

    return links

def is_chi( residue_type, atom ) :
    for chi in residue_type.chi :
        for chi_at in chi :
            if atom == chi_at : return True
    return False

# assumes that atom is a member of residue_type
def is_hydrogen( chem_db, residue_type, atom_index ) :
    atom_type = residue_type.atoms[ atom_index ].atom_type
    element = [ x[1] for x in chem_db.atom_types if x[0] == atom_type ][ 0 ]
    return element == "H"

def chi_continuation( residue_type, query_at1, query_at2 ) :
    ''' Return true if the query-at2 neighbor of currently-focused query-at1 continues a chi dihedral
    that query-at1 is part of. residue_type should be of type tmol.structure.restypes.ResidueType; query_at1
    and query_at2 should both be integers'''
    for at1, at2, at3, at4 in residue_type.chi_inds :
        if query_at1 == at3 and query_at2 == at4 : return True
        if query_at1 == at2 and query_at1 == at1 : return True
    return False

def chi_interruption( residue_type, done, query_at1, query_at2 ) :
    ''' Return true if the query-at2 neighbor of currently-focused query-at1 would interrupt the construction
    of a chi dihdedral. residue_type should be of type tmol.structure.restypes.ResidueType; query_at1
    and query_at2 should both be integers'''
    for at1, at2, at3, at4 in residue_type.chi_inds :
        if query_at1 == at3 and query_at2 == at4 : return False
        if query_at1 == at2 and query_at1 == at1 : return False
    for at1, at2, at3, at4 in residue_type.chi_inds :
        if query_at2 == at2 and query_at1 != at1 and qeury_at1 != at3 : return True
        if query_at2 == at3 and query_at1 != at2 and qeury_at1 != at4 : return True
        if query_at2 == at1 and done[ at4 ] : return True
        if query_at2 == at4 and done[ at1 ] : return True
    return False
    

def setup_atom_links( root_atom_index, full_links, done, chem_db, residue_type, new_links ) :
    ''' recursive function for deciding which atoms to put ino the atom tree in which order;
    ripped from R3'''
    done[ root_atom_index ] = True
    neighbs = full_links[root_atom_index]
    n_neighbs = len( neighbs )
    root_links = new_links[ root_atom_index ]
    if n_neighbs == 1 :
        if not done[ neighbs[0] ]  :
            root_links.append( neighbs[0] )
            setup_atom_links( neighbs[0], full_links, done, chem_db, residue_type, new_links )
        return
    # next priority: mainchain
    for neighb in neighbs :
        if done[ neighb ] : continue
        if neighb in residue_type.mainchain_inds :
            root_links.append( neighb )
            setup_atom_links( neighb, full_links, done, chem_db, residue_type, new_links )
    # next priority: within a chi angle and heavy
    root_is_chi = is_chi( residue_type, root_atom_index )
    for neighb in neighbs :
        if done[ neighb ] : continue
        if root_is_chi and is_chi( residue_type, neighb ) and chi_continuation( residue_type, root_atom_index, neighb ) and not is_hydrogen( chem_db, residue_type, neighb ) :
            root_links.append( neighb )
            setup_atom_links( neighb, full_links, done, chem_db, residue_type, new_links )

    # next priority: any heavyatom chi
    for neighb in neighbs :
        if done[ neighb ] : continue
        if is_chi( residue_type, neighb ) and not is_hydrogen( chem_db, residue_type, neighb ) :
            root_links.append( neighb )
            setup_atom_links( neighb, full_links, done, chem_db, residue_type, new_links )

    # next priority: any chi -- could be hydrogen
    for neighb in neighbs :
        if done[ neighb ] : continue
        if is_chi( residue_type, neighb ) :
            root_links.append( neighb )
            setup_atom_links( neighb, full_links, done, chem_db, residue_type, new_links )

    # next priority: heavy atoms
    for neighb in neighbs :
        if done[ neighb ] : continue
        if is_chi( residue_type, neighb ) and chi_interruption( residue_type, done, root_atom_index, neighb ) : continue
        if not is_hydrogen( chem_db, residue_type, neighb ) :
            root_links.append( neighb )
            setup_atom_links( neighb, full_links, done, chem_db, residue_type, new_links )

    # lowest priority: hydrogens
    for neighb in neighbs :
        if done[ neighb ] : continue
        if is_chi( residue_type, neighb ) and chi_interruption( residue_type, done, root_atom_index, neighb ) : continue
        root_links.append( neighb )
        setup_atom_links( neighb, full_links, done, chem_db, residue_type, new_links )        
        
            
def add_atom( root_atom_index, ordered_links, atom_pointers ) :
    ''' Creates a set of BondedAtoms (no JumpAtoms yet!) given the tree-structured
    ordered_links dictionary and puts them into the atom_pointers map'''
    tree_node = BondedAtom()
    atom_pointers[ root_atom_index ] = tree_node
    for child in ordered_links[ root_atom_index ] :
        if atom_pointers[ child ] is not None : continue
        tree_node.children.append( add_atom( child, ordered_links, atom_pointers ) )
    for child in tree_node.children :
        child.parent = tree_node
    return tree_node

def create_residue_tree( chem_db, residue_type, res_index, root_atom ) :
    full_links = [ [] for x in range(len(residue_type.atoms) ) ]
    create_links_simple( residue_type, full_links )
    ordered_links = [ [] for x in range(len(residue_type.atoms ) ) ]
    done = [ False ] * len( residue_type.atoms )
    root_atom_index = residue_type.atom_to_idx[ root_atom ]
    setup_atom_links( root_atom_index, full_links, done, chem_db, residue_type, ordered_links )

    atom_pointers = [ None ] * len(residue_type.atoms )
    root = add_atom( root_atom_index, ordered_links, atom_pointers )
    for atom_index, at in enumerate( atom_pointers ) :
        at.atomid = AtomID( res_index, atom_index )
    return ( root, atom_pointers )

def set_coords( residue, atom_pointers ) :
    for i in range(len( residue.coords )) :
        atom_pointers[ i ].xyz = residue.coords[ i ]


# does this belong in the kinematic layer, or perhaps instead, in the
# system layer?
def tree_from_residues( chem_db, residues ) :
    ''' Construct an atom tree from a list of residues, represented as a pair:
    The root of the tree, and a list of dictionaries pointing to the atoms in the tree'''
    atom_pointers_list = []
    last_residue = None
    first_root = None
    for ind, residue in enumerate( residues ) :
        #print( "tree from residues", ind )
        root, atom_pointers = create_residue_tree( chem_db, residue.residue_type, ind, \
                                                   residue.residue_type.lower_connect )
        set_coords( residue, atom_pointers )
        if last_residue :
            upper_connect_last = atom_pointers_list[ -1 ][ last_residue.residue_type.atom_to_idx[ last_residue.residue_type.upper_connect ] ]
            upper_connect_last.children.insert( 0, root )
            root.parent = upper_connect_last
        if not first_root :
            first_root = root
        last_residue = residue
        atom_pointers_list.append( atom_pointers )
    first_root.update_internal_coords()
    return AtomTree( first_root, atom_pointers_list )
    
