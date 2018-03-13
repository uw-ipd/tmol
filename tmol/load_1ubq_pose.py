import pickle
from residue_type import *

def load_1ubq() :
    (rts,residues,resnames) = pickle.load( open( "ubq.pickle", "rb" ) )
    for i,res in enumerate( residues ) :
        res.residue_type = rts.residue_types[ resnames[ i ] ]
    return rts, residues

if __name__ == "__main__" :
    rts,residues = load_1ubq()
    print( len(residues), residues[0].residue_type.name )
