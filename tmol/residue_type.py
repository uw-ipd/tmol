# a small class holding data created for us by Rosetta proper for the residue types
# that make up ubiquitin

class ResidueType :
    def __init__( self ) :
        self.name = ""
        self.natoms = 0
        self.atom_names = []
        self.atom_types = []
        self.connection_points = []
        self.path_dists = []
        self.donor_hydrogens = []
        self.acceptors = []
        self.partial_charges = []

class ResidueTypeSet :
    def __init__( self ) :
        self.residue_types = {}

class Residue :
    def __init__( self ) :
        self.residue_type = None
        self.seqpos = 0
        self.coords = []
        self.connection_parters = []
