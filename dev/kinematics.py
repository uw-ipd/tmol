#!/usr/bin/env python

# some code stolen from https://www.lfd.uci.edu/~gohlke/code/transformations.py.html

import numpy

# doftypes
BOND = 1
JUMP = 2

# data structure describing the atom-level kinematics of a molecular system
kintree_node_dtype = numpy.dtype([
   ("atom_name" , numpy.str, 4),
   ("resnum"    , numpy.int),
   ("doftype"   , numpy.int),
   ("parent"    , numpy.int),
   ("frame"     , numpy.int, 3)
])

######
# DEBUG1: score a conformation (dummy sfxn)
def score(kintree, coords):
	natoms = coords.shape[0]
	dists = numpy.sqrt(numpy.square(coords[:,numpy.newaxis]-coords).sum(axis=2))
	igraph = numpy.bitwise_and(numpy.triu(~numpy.eye(dists.shape[0],dtype=bool)),dists < 3.4).nonzero();
	score = (3.4-dists[igraph])*(3.4-dists[igraph])
	return numpy.sum(score)
######

######
# DEBUG2: cartesian derivs of a conformation (dummy sfxn)
def dscore(kintree, coords):
	natoms = coords.shape[0]
	dxs = coords[:,numpy.newaxis]-coords
	dists = numpy.sqrt(numpy.square(dxs).sum(axis=2))
	igraph = numpy.bitwise_and(numpy.triu(~numpy.eye(dists.shape[0],dtype=bool)),dists < 3.4).nonzero();

	dEdxs = numpy.zeros([natoms,natoms,3])
	dEdxs[igraph[0],igraph[1],:] = -2 * (3.4-dists[igraph].reshape(-1,1)) * dxs[igraph]/dists[igraph].reshape(-1,1);

	dEdx = numpy.zeros([natoms,3])
	dEdx = dEdxs.sum(axis=1) - dEdxs.sum(axis=0)

	return dEdx
######

# helper to quickly invert a HT
def HTinv (HTs):
	N = HTs.shape[0];
	HTinvs = numpy.tile(numpy.identity(4),(N,1,1))
	HTinvs[:,:3,:3] = numpy.transpose(HTs[:,:3,:3], (0,2,1))
	HTinvs[:,:3,3] = -numpy.einsum( 'aij,aj->ai', HTinvs[:,:3,:3], HTs[:,:3,3] )
	return HTinvs


## segmented scan code for passing:
##   - HT's down the atom tree
##   - derivs up the atom tree
def SegScan (data, parents, operator, verbose=False):
	nelts = data.shape[0]
	N = numpy.ceil(numpy.log2( nelts )) # this might result in several extra rounds...

	backPointers = parents
	prevBackPointers = numpy.arange(nelts)
	toCalc = (prevBackPointers!=backPointers)
	retval = data
	for i in numpy.arange(N):
		prevBackPointers = backPointers
		operator( retval, backPointers, toCalc )
		backPointers = prevBackPointers[prevBackPointers]
		toCalc = (prevBackPointers!=backPointers)

	return (retval)

## segmented scan "down" operator: aggregate HTs
def HTcollect( HTs, ptrs, toCalc ):
	HTs[toCalc] = numpy.matmul( HTs[ptrs][toCalc], HTs[toCalc] )

## segmented scan "up" operator: aggregate f1/f2s
def Fscollect( fs, ptrs, toCalc ):
	numpy.add.at(fs, ptrs[toCalc], fs[toCalc])

# JUMP dofs -> HTs
# jump dofs are _9_ parameters: 
#  - 3 translational
#  - 3 rotational deltas
#  - 3 rotational
# Only the rotational deltas are exposed to minimization
def JumpTransforms(dofs):
	natoms = dofs.shape[0]

	si = numpy.sin(dofs[:,3])
	sj = numpy.sin(dofs[:,4])
	sk = numpy.sin(dofs[:,5])
	ci = numpy.cos(dofs[:,3])
	cj = numpy.cos(dofs[:,4])
	ck = numpy.cos(dofs[:,5])
	cc = ci*ck
	cs = ci*sk
	sc = si*ck
	ss = si*sk
	Rdelta = numpy.zeros([natoms,4,4])
	Rdelta[:,0,0] = cj*ck
	Rdelta[:,0,1] = sj*sc-cs
	Rdelta[:,0,2] = sj*cc+ss
	Rdelta[:,1,0] = cj*sk
	Rdelta[:,1,1] = sj*ss+cc
	Rdelta[:,1,2] = sj*cs-sc
	Rdelta[:,2,0] = -sj
	Rdelta[:,2,1] = cj*si
	Rdelta[:,2,2] = cj*ci
	Rdelta[:,0,3] = dofs[:,0]
	Rdelta[:,1,3] = dofs[:,1]
	Rdelta[:,2,3] = dofs[:,2]
	Rdelta[:,3,3] = 1

	si = numpy.sin(dofs[:,6])
	sj = numpy.sin(dofs[:,7])
	sk = numpy.sin(dofs[:,8])
	ci = numpy.cos(dofs[:,6])
	cj = numpy.cos(dofs[:,7])
	ck = numpy.cos(dofs[:,8])
	cc = ci*ck
	cs = ci*sk
	sc = si*ck
	ss = si*sk
	Rglobal = numpy.zeros([natoms,4,4])
	Rglobal[:,0,0] = cj*ck
	Rglobal[:,0,1] = sj*sc-cs
	Rglobal[:,0,2] = sj*cc+ss
	Rglobal[:,1,0] = cj*sk
	Rglobal[:,1,1] = sj*ss+cc
	Rglobal[:,1,2] = sj*cs-sc
	Rglobal[:,2,0] = -sj
	Rglobal[:,2,1] = cj*si
	Rglobal[:,2,2] = cj*ci
	Rglobal[:,3,3] = 1

	Ms= numpy.matmul(Rdelta, Rglobal)

	return Ms

# HTs -> JUMP dofs
# this function will always assign rotational delta = 0
def InvJumpTransforms(Ms):
	njumpatoms = Ms.shape[0]

	dofs = numpy.empty( [njumpatoms,9] )

	dofs[:,:3] = Ms[:,:3,3]   # translation
	dofs[:,3:6] = 0           # rotational "delta"

	cys = numpy.sqrt(Ms[:,0,0]*Ms[:,0,0] + Ms[:,1,0]*Ms[:,1,0])

	problemSelector = (cys <= 4*numpy.finfo(float).eps)

	dofs[~problemSelector,6] = numpy.arctan2( Ms[~problemSelector,2,1], Ms[~problemSelector,2,2])
	dofs[~problemSelector,7] = numpy.arctan2(-Ms[~problemSelector,2,0], cys[~problemSelector])
	dofs[~problemSelector,8] = numpy.arctan2( Ms[~problemSelector,1,0], Ms[~problemSelector,0,0])

	dofs[problemSelector,6] = numpy.arctan2(-Ms[problemSelector,1,2], Ms[problemSelector,1,1])
	dofs[problemSelector,7] = numpy.arctan2(-Ms[problemSelector,2,0], cys[problemSelector])
	dofs[problemSelector,8] = 0.0

	return dofs

# compute JUMP derivatives from f1/f2
def JumpDerivatives( dofs, Ms, Mparents, f1s, f2s):
	# trans dofs
	njumpatoms = dofs.shape[0]
	dsc_ddofs = numpy.zeros([njumpatoms,6]);
	x_axes = Mparents[:,0:3,0]
	y_axes = Mparents[:,0:3,1]
	z_axes = Mparents[:,0:3,2]
	dsc_ddofs[:,0] = numpy.einsum('ij, ij->i', x_axes, f2s )
	dsc_ddofs[:,1] = numpy.einsum('ij, ij->i', y_axes, f2s )
	dsc_ddofs[:,2] = numpy.einsum('ij, ij->i', z_axes, f2s )

	end_pos = Ms[:,0:3,3]
	rotdof3_axes = -Mparents[:,0:3,2]
	
	zrots = numpy.zeros([njumpatoms,3,3]);
	zrots[:,0,0] =  numpy.cos(dofs[:,5])
	zrots[:,0,1] = -numpy.sin(dofs[:,5])
	zrots[:,1,0] =  numpy.sin(dofs[:,5])
	zrots[:,1,1] =  numpy.cos(dofs[:,5])
	zrots[:,2,2] =  1
	rotdof2_axes = -numpy.matmul(Mparents[:,0:3,0:3],zrots)[:,0:3,1]

	yrots = numpy.empty([njumpatoms,3,3]);
	yrots[:,0,0] =  numpy.cos(-dofs[:,4])
	yrots[:,0,2] = -numpy.sin(-dofs[:,4])
	yrots[:,1,1] =  1
	yrots[:,2,0] =  numpy.sin(-dofs[:,4])
	yrots[:,2,2] =  numpy.cos(-dofs[:,4])
	rotdof1_axes = -numpy.matmul(numpy.matmul(Mparents[:,0:3,0:3],zrots),yrots)[:,0:3,0]

	dsc_ddofs[:,3] = numpy.einsum('ij, ij->i', rotdof1_axes, f1s ) \
	             + numpy.einsum('ij, ij->i', numpy.cross( rotdof1_axes, end_pos ), f2s )
	dsc_ddofs[:,4] = numpy.einsum('ij, ij->i', rotdof2_axes, f1s ) \
	             + numpy.einsum('ij, ij->i', numpy.cross( rotdof2_axes, end_pos ), f2s )
	dsc_ddofs[:,5] = numpy.einsum('ij, ij->i', rotdof3_axes, f1s ) \
	             + numpy.einsum('ij, ij->i', numpy.cross( rotdof3_axes, end_pos ), f2s )

	return dsc_ddofs


# BOND dofs -> HTs
def BondTransforms( dofs ):
	natoms = dofs.shape[0]

	cp = numpy.cos(dofs[:,2]);
	sp = numpy.sin(dofs[:,2]);
	ct = numpy.cos(dofs[:,1]);
	st = numpy.sin(dofs[:,1]);
	d = dofs[:,0]

	Ms = numpy.zeros( [natoms,4,4] )
	Ms[:,0,0] = ct
	Ms[:,0,1] = -st
	Ms[:,0,3] = d*ct
	Ms[:,1,0] = cp*st
	Ms[:,1,1] = cp*ct
	Ms[:,1,2] = -sp
	Ms[:,1,3] = d*cp*st
	Ms[:,2,0] = sp*st
	Ms[:,2,1] = sp*ct
	Ms[:,2,2] = cp
	Ms[:,2,3] = d*sp*st
	Ms[:,3,3] = 1

	return Ms

# HTs -> BOND dofs
def InvBondTransforms( Ms ):
	nbondatoms = Ms.shape[0]

	dofs = numpy.empty( [nbondatoms,3] )
	dofs[:,0] = numpy.sqrt(numpy.square(Ms[:,:3,3]).sum(axis=1))
	dofs[:,1] = numpy.arctan2( -Ms[:,0,1],  Ms[:,0,0])
	dofs[:,2] = numpy.arctan2( -Ms[:,1,2],  Ms[:,2,2])

	return dofs

# compute BOND derivatives from f1/f2
def BondDerivatives( dofs, Ms, Mparents, f1s, f2s):
	nbondatoms = dofs.shape[0]

	end_pos = Mparents[:,0:3,3]
	phi_axes = Mparents[:,0:3,0]
	theta_axes = Ms[:,0:3,2]
	d_axes = Ms[:,0:3,0]

	dsc_ddofs = numpy.zeros([nbondatoms,3])

	dsc_ddofs[:,0] = numpy.einsum('ij, ij->i', d_axes, f2s )
	dsc_ddofs[:,1] = -numpy.sign(dofs[:,1]) * \
		( numpy.einsum('ij, ij->i', theta_axes, f1s ) + \
		  numpy.einsum('ij, ij->i', numpy.cross( theta_axes, end_pos ), f2s ) )
	dsc_ddofs[:,2] = \
		- numpy.einsum('ij, ij->i', phi_axes, f1s ) \
		- numpy.einsum('ij, ij->i', numpy.cross( phi_axes, end_pos ), f2s )

	return dsc_ddofs

# xyzs -> HTs
def HTs_from_frames(Cs,Xs,Ys,Zs):
	natoms = Cs.shape[0]

	Ms = numpy.zeros([natoms,4,4])

	Ms[:,:3,0] = Xs-Ys
	Ms[:,:3,0] = Ms[:,:3,0] / numpy.sqrt(numpy.square(Ms[:,:3,0]).sum(axis=1)[:,numpy.newaxis])
	Ms[:,:3,2] = numpy.cross( Ms[:,:3,0] , Zs-Xs )
	Ms[:,:3,2] = Ms[:,:3,2] / numpy.sqrt(numpy.square(Ms[:,:3,2]).sum(axis=1)[:,numpy.newaxis])
	Ms[:,:3,1] = numpy.cross( Ms[:,:3,2] , Ms[:,:3,0] )
	Ms[:,:3,3] = Cs
	Ms[:,3,3] = 1

	return (Ms)

# xyzs -> HTs, dofs
#  - "backward" kinematics
def backwardKin(kintree, coords):
	natoms = coords.shape[0]

	parents = kintree["parent"]
	frames = kintree["frame"]

	# 1) global HTs
	HTs = HTs_from_frames( coords, coords[frames[:,0],:], coords[frames[:,1],:], coords[frames[:,2],:] )

	# 2) local HTs
	localHTs = numpy.empty([natoms, 4, 4])
	localHTs[1:] = numpy.matmul( HTinv(HTs[parents[1:],:,:]), HTs[1:,:,:] )

	# 3) dofs
	dofs = numpy.zeros( [natoms, 9] )

	bondSelector = (kintree["doftype"] == BOND)
	bondSelector[0] = False
	dofs[bondSelector,:3] = InvBondTransforms( localHTs[bondSelector,:3] )

	jumpSelector = (kintree["doftype"] == JUMP)
	jumpSelector[0] = False
	dofs[jumpSelector,:9] = InvJumpTransforms( localHTs[jumpSelector,:9] )

	print (dofs[0:3,:])

	return (HTs,dofs)


# dofs -> HTs, xyzs
#  - "forward" kinematics
def forwardKin(kintree, dofs):
	natoms = dofs.shape[0]

	parents = kintree["parent"]

	# 1) local HTs
	HTs = numpy.empty([natoms,4,4])

	bondSelector = (kintree["doftype"] == BOND)
	HTs[bondSelector,:,:] = BondTransforms( dofs[bondSelector,0:3] )

	jumpSelector = (kintree["doftype"] == JUMP)
	HTs[jumpSelector,:,:] = JumpTransforms( dofs[jumpSelector,0:9] )

	# 2) global HTs (rewrite 1->N in-place)
	SegScan(HTs, parents, HTcollect)

	coords = numpy.zeros( [natoms, 3] )
	coords = numpy.matmul( HTs, [0,0,0,1] )[:,:3]
	return (HTs,coords)


# xyz derivs -> dof derivs
#  - derivative mapping using Abe and Go approach
def resolveDerivs(kintree, dofs, HTs, dsc_dx):
	natoms = coords.shape[0]

	parents = kintree["parent"]

	# 1) local f1/f2s
	Xs = HTs[:,0:3,3]
	f1s = numpy.cross(Xs, Xs-dsc_dx)
	f2s = dsc_dx

	# 2) pass f1/f2s up tree
	SegScan(f1s, parents, Fscollect)
	SegScan(f2s, parents, Fscollect)

	# 3) convert to dscore/dtors
	dsc_ddofs = numpy.zeros( [natoms, 9] )
	bondSelector = (kintree["doftype"] == BOND)
	dsc_ddofs[bondSelector,0:3] = BondDerivatives( \
		dofs[bondSelector,:], HTs[bondSelector,:,:], HTs[parents[bondSelector],:,:],  \
		f1s[bondSelector,:], f2s[bondSelector,:] \
	)
	jumpSelector = (kintree["doftype"] == JUMP)
	dsc_ddofs[jumpSelector,0:6] = JumpDerivatives( \
		dofs[jumpSelector,:], HTs[jumpSelector,:,:], HTs[parents[jumpSelector],:,:],  \
		f1s[jumpSelector,:], f2s[jumpSelector,:] \
	)

	return dsc_ddofs


# debugging: dump a PDB-like file
def writePDB( kintree, coords ):
	atom_record_format = "ATOM  {:5d} {:^4}{:^1}{:3s} {:1}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"	
	for i in numpy.arange(len(kintree)):
		print (atom_record_format.format( \
			i+1, kintree["atom_name"][i], " ", "ALA", "A", kintree["resnum"][i], " ", \
			coords[i,0], coords[i,1], coords[i,2], 1, 0));


if __name__ == '__main__':
	NATOMS=23

	# kinematics definition
	kintree = numpy.empty( NATOMS, dtype=kintree_node_dtype);
	kintree[0]  = ("ORIG", 0, BOND,  0,  (1, 0, 2))
	kintree[1]  = (" X  ", 0, BOND,  0,  (1, 0, 2))
	kintree[2]  = (" Y  ", 0, BOND,  0,  (2, 0, 1))

	kintree[3]  = (" N  ", 1, JUMP,  0,  (4, 3, 5))
	kintree[4]  = (" CA ", 1, BOND,  3,  (4, 3, 5))
	kintree[5]  = (" CB ", 1, BOND,  4,  (5, 4, 3))
	kintree[6]  = (" C  ", 1, BOND,  4,  (6, 4, 3))
	kintree[7]  = (" O  ", 1, BOND,  6,  (7, 6, 4))

	kintree[8]  = (" N  ", 2, JUMP,  3,  (9, 8, 10))
	kintree[9]  = (" CA ", 2, BOND,  8,  (9, 8, 10))
	kintree[10] = (" CB ", 2, BOND,  9, (10, 9, 8))
	kintree[11] = (" C  ", 2, BOND,  9, (11, 9, 8))
	kintree[12] = (" O  ", 2, BOND, 11, (12,11, 9))

	kintree[13] = (" N  ", 3, JUMP,  3, (14, 13, 15))
	kintree[14] = (" CA ", 3, BOND, 13, (14, 13, 15))
	kintree[15] = (" CB ", 3, BOND, 14, (15, 14, 13))
	kintree[16] = (" C  ", 3, BOND, 14, (16, 14, 13))
	kintree[17] = (" O  ", 3, BOND, 16, (17, 16, 14))

	kintree[18] = (" N  ", 4, JUMP,  3, (19, 18, 20))
	kintree[19] = (" CA ", 4, BOND, 18, (19, 18, 20))
	kintree[20] = (" CB ", 4, BOND, 19, (20, 19, 18))
	kintree[21] = (" C  ", 4, BOND, 19, (21, 19, 18))
	kintree[22] = (" O  ", 4, BOND, 21, (22, 21, 19))


	# coordinates
	coords = numpy.empty( [NATOMS,3] )
	coords[0,:]  = [0.000 ,  0.000 ,  0.000]
	coords[1,:]  = [1.000 ,  0.000 ,  0.000]
	coords[2,:]  = [0.000 ,  1.000 ,  0.000]

	coords[3,:]  = [2.000 ,  2.000 ,  2.000]
	coords[4,:]  = [3.458 ,  2.000 ,  2.000]
	coords[5,:]  = [3.988 ,  1.222 ,  0.804]
	coords[6,:]  = [4.009 ,  3.420 ,  2.000]
	coords[7,:]  = [3.383 ,  4.339 ,  1.471]

	coords[8,:]  = [5.184 ,  3.594 ,  2.596]
	coords[9,:]  = [5.821 ,  4.903 ,  2.666]
	coords[10,:] = [5.331 ,  5.667 ,  3.888]
	coords[11,:] = [7.339 ,  4.776 ,  2.690]
	coords[12,:] = [7.881 ,  3.789 ,  3.186]

	coords[13,:]  = [7.601 ,  2.968 ,  5.061] 
	coords[14,:]  = [6.362 ,  2.242 ,  4.809]
	coords[15,:]  = [6.431 ,  0.849 ,  5.419]
	coords[16,:]  = [5.158 ,  3.003 ,  5.349]
	coords[17,:]  = [5.265 ,  3.736 ,  6.333]

	coords[18,:]  = [4.011 ,  2.824 ,  4.701]
	coords[19,:]  = [2.785 ,  3.494 ,  5.115]
	coords[20,:]  = [2.687 ,  4.869 ,  4.470]
	coords[21,:]  = [1.559 ,  2.657 ,  4.776]
	coords[22,:]  = [1.561 ,  1.900 ,  3.805]

	sc = score(kintree[3:],coords[3:,:])

	(HTs,dofs) = backwardKin(kintree, coords)

	## test folding
	#(HTs,coords) = forwardKin(kintree, dofs)
	#writePDB( kintree, coords )

	## test perturb
	#dofs[8,3:6]  = [0.02,0.02,0.02]
	#dofs[13,3:6] = [0.01,0.01,0.02]
	#dofs[18,3:6] = [0.01,0.02,0.01]
	#(HTs,coords) = forwardKin(kintree, dofs)

	dsc_dx = numpy.zeros([NATOMS,3])
	dsc_dx[3:,:] = dscore(kintree[3:],coords[3:,:])
	dsc_dtors_a = resolveDerivs( kintree, dofs, HTs, dsc_dx )


	print ("-bond_deriv_analytic-")
	bonds = numpy.hstack([numpy.arange(4,8),numpy.arange(9,13),numpy.arange(14,18),numpy.arange(19,23)])

	print (dofs[bonds,:3])


	print (dsc_dtors_a[bonds,0:3])

	print ("-jump_deriv_analytic-")
	print (dsc_dtors_a[(8,13,18),0:6])

	dsc_dtors_n = numpy.zeros( [NATOMS,9])
	for i in numpy.arange(0,NATOMS):
		for j in numpy.arange(0,6):
			dofs[i,j] += 0.00001;
			(HTs,coordsAlt) = forwardKin(kintree, dofs)
			sc_p = score(kintree[3:],coordsAlt[3:,:])
			#writePDB( kintree, coordsAlt ) #print (HTs[i])
			dofs[i,j] -= 0.00002;
			(HTs,coordsAlt) = forwardKin(kintree, dofs)
			sc_m = score(kintree[3:],coordsAlt[3:,:])
			dofs[i,j] += 0.00001;
			#writePDB( kintree, coordsAlt ) #print (HTs[i])

			dsc_dtors_n[i,j] = (sc_p - sc_m)/0.00002

	print ("-bond_deriv_numeric-")
	print (dsc_dtors_n[bonds,0:3])

	print ("-jump_deriv_numeric-")
	print (dsc_dtors_n[(8,13,18),0:6])

