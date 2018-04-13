import typing
import attr
import numpy
import math

class HomogeneousTransform :

    @staticmethod
    def from_coords( p1, p2, p3 ) :
        return HomogeneousTransform( p1=p1, p2=p2, p3=p3 )

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

    def __init__( self, input_frame=None, p1=None, p2=None, p3=None ) :
        if input_frame is not None :
            assert input_frame.shape[ 0 ] == 4
            assert input_frame.shape[ 1 ] == 4
            self.frame = input_frame
        elif p1 is not None and p2 is not None and p3 is not None :
            # define a right handed coordinate frame located at p3 with the z along
            # the vector from p2 to p3, and p1 lying in the yz plane
            assert len(p1) == 3
            assert len(p2) == 3
            assert len(p3) == 3
            self.frame = numpy.eye(4)
            v23 = p3 - p2
            v23 = v23/numpy.linalg.norm( v23 )
            self.frame[0:3,2] = v23
            v21 = p1 - p2
            dotprod = numpy.dot( v21, v23 )
            v21 -= dotprod * v23;
            v21 = v21/numpy.linalg.norm(v21)
            self.frame[0:3,1] = v21
            xaxis = numpy.cross( v21, v23 )
            self.frame[0:3,0] = xaxis
        else :
            self.frame = numpy.eye(4)

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

class TreeAtom :
    def update_xyz( self, parent_ht = None ) :
        raise NotImplementedError;

    def update_internal_coords( self, parent_ht = None ) :
        raise NotImplementedError;

@attr.s( auto_attribs=True, slots=True )
class BondedAtom( TreeAtom ) :
    parent : TreeAtom = None
    phi : float = 0
    theta : float = 0
    d : float = 0
    xyz : typing.Tuple[ float, float, float ] = ( 0, 0, 0 )
    children : typing.List[ TreeAtom ] = attr.Factory(list)

    def update_xyz( self, parent_ht = None ) :
        if parent_ht is None :
            parent_ht = HomogeneousTransform() # identity
        dihedral_rotation_ht = HomogeneousTransform.zrot( self.phi )
        # modify the parent_ht with the dihedral rotation here
        parent_ht *= dihedral_rotation_ht
        bond_angle_rotation_ht = HomogeneousTransform.xrot( -1 * self.theta )
        trans_ht = HomogeneousTransform.ztrans( self.d )
        new_ht = parent_ht * bond_angle_rotation_ht * trans_ht;
        xyz = new_ht.frame[0:3,3]
        for child in self.children :
            update_xyz( new_ht )


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
            zdot = numpy.dot( w, parent_ht.frame[0:3,2] )
            # hacky version with coordinates in general position!
            # look at BondedAtom.cc for the three cases dependent on x's value (aka zdot here)
            self.theta = math.acos( zdot )

            # finally, we need the rotation about the z axis so that the coordinate will be in
            # the yz plane
            xdot = numpy.dot( w, parent_ht.frame[0:3,0] )
            ydot = numpy.dot( w, parent_ht.frame[0:3,1] )
            self.phi = math.atan2( -xdot, ydot )

        # modify the parent ht so that younger siblings will have their offset phi readily
        # calculated
        parent_ht *= HomogeneousTransform.zrot( self.phi )
        #print( "parent_ht(again)" ); print( parent_ht )
        new_ht = parent_ht * HomogeneousTransform.xrot( -self.theta ) * HomogeneousTransform.ztrans( self.d )

        print( "d", self.d, "theta", self.theta, "phi", self.phi )

        for child in self.children :
            child.update_internal_coords( new_ht )


