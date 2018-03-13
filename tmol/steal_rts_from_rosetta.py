import pyrosetta
from residue_type import *
import pickle

def residue_type_from_rosetta_restype( rose_rt ) :
    rt = ResidueType()
    rt.name = rose_rt.name()
    rt.natoms = rose_rt.natoms()
    rt.atom_names = [ rose_rt.atom_name( i ) for i in range(1,rose_rt.natoms()+1 ) ]
    rt.atom_types = [ rose_rt.atom_type( i ) for i in range(1,rose_rt.natoms()+1 ) ]
    rt.connection_points = [ rose_rt.residue_connection(i).atomno() - 1 for i in range(1,rose_rt.n_possible_residue_connections()+1) ]
    rt.path_dists = [ [ j for j in i ] for i in rose_rt.path_distances() ]
    rt.donor_hydrogens = [ i - 1 for i in rose_rt.Hpos_polar() ]
    rt.acceptors = [ i - 1 for i in rose_rt.accpt_pos() ]
    rt.partial_charges = [ rose_rt.atom(i).charge() for i in range(1,rose_rt.natoms()+1 ) ]
    return rt

def construct_rts_from_pose( pose ) :
    rts = ResidueTypeSet()
    for i in range(1,pose.total_residue()+1 ) :
        ires = pose.residue(i)
        if ires.name() not in rts.residue_types :
            rts.residue_types[ ires.name() ] = residue_type_from_rosetta_restype( ires.type() )
    return rts

def rts_and_res_from_pose( pose ) :
    rts = construct_rts_from_pose( pose )
    residues = []
    for i in range(1,pose.total_residue()+1) :
        pose_res = pose.residue(i)
        ires = Residue()
        print( pose_res.name() )
        ires.residue_type = rts.residue_types[ pose_res.name() ]
        for j in range(1,pose_res.natoms()) :
            jat = pose_res.atom(j)
            ires.coords.append( (jat.xyz()[0], jat.xyz()[1], jat.xyz()[2] ) )
        ires.seqpos = pose_res.seqpos()-1
        ires.connection_partners = [ (pose_res.connect_map(i).resid()-1, pose_res.connect_map(i).connid()-1) for i in range(1,pose_res.connect_map_size()+1) ]
        residues.append( ires )
    return rts,residues

if __name__ == "__main__" :
    pyrosetta.init()
    ubq = pyrosetta.rosetta.core.import_pose.pose_from_file( "../inputs/1ubq.pdb" )
    rts,residues = rts_and_res_from_pose( ubq )

    resnames = [ res.residue_type.name for res in residues ]
    for x in residues : x.residue_type = None
    pickle.dump( (rts, residues, resnames), open( "ubq.pickle", "wb" ) )
