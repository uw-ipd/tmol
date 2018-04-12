import typing
import attr
import numpy

class HomogeneousTransform :

    @staticmethod
    def from_coords( p1, p2, p3 ) :
        return HomogeneousTransform( p1=p1, p2=p2, p3=p3 )

    @staticmethod
    def from_frame( frame ) :
        return HomogeneousTransform( input_frame=frame )

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
            v23 = v23/numpy.norm( v23 )
            self.frame[2,0:2] = v23
            v21 = p1 - p2
            dotprod = numpy.dot( v21. v23 )
            v21 -= dotprod * v 23;
            v21 = v21/numpy.norm(v21)
            self.frame[1,0:2] = v21
            xaxis = numpy.cross( v21, v23 )
            self.frame[0,0:2] = xaxis
        else :
            self.frame = numpy.eye(4)

    def set_rotation_x( self, theta ) :
        '''Set a rotation about the x axis; theta should be in radians'''
        self.frame.fill(0)
        ct = cos( theta )
        st = sin( theta )
        self.frame[0,0] = 1
        self.frame[1,1] = ct
        self.frame[2,2] = ct
        self.frame[1,2] = -st
        self.frame[2,1] = st
        self.frame[3,3] = 1

    def set_rotation_y( self, theta ) :
        '''Set a rotation about the y axis; theta should be in radians'''
        self.frame.fill(0)
        ct = cos( theta )
        st = sin( theta )
        self.frame[1,1] = 1
        self.frame[3,3] = 1
        self.frame[2,2] = ct
        self.frame[0,0] = ct
        self.frame[0,2] = st
        self.frame[2,0] = -st

    def set_rotation_z( self, theta ) :
        '''Set a rotation about the z axis; theta should be in radians'''
        self.frame.fill(0)
        ct = cos( theta )
        st = sin( theta )
        self.frame[2,2] = 1
        self.frame[3,3] = 1
        self.frame[0,0] = ct
        self.frame[1,1] = ct
        self.frame[0,1] = -st
        self.frame[1,0] = st

    def set_xaxis_translation( self, dist ) :
        self.frame.fill( 0 )
        self.frame[3,3] = 1
        self.frame[0,3] = dist

    def set_yaxis_translation( self, dist ) :
        self.frame.fill( 0 )
        self.frame[3,3] = 1
        self.frame[1,3] = dist

    def set_zaxis_translation( self, dist ) :
        self.frame.fill( 0 )
        self.frame[3,3] = 1
        self.frame[2,3] = dist

    def __mul__( self, other ) :
        ht_res = HomogeneousTransform()
        ht_res.frame = self.frame * other.frame
        return ht_res

    def __imul__( self, other ) :
        self.frame *= newframe

@attr.s( auto_attribs=True, slots=True )
class BondedAtom :
    parent : BondedAtom
    phi : float
    theta : float
    d : float
    children : typing.List[ BondedAtom ] = attr.Factory(list)
    xyz : typing.Tuple[ float, float, float ]

    def refold( parent_ht = None ) :
        pass
    
