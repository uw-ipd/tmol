"""Distance-geometry 3D coordinate generation for ligands from a SMILES.

Replaces OpenBabel's ``make3D`` + rotor search. Pipeline: MMFF ideal geometry
(RDKit, read-only) -> distance-bounds matrix -> metric-matrix embedding -> a
torch stress refine (4D chiral annealing with random restarts, then a full-weight
3D pass) -> OpenBabel force-field cleanup. MMFF-untypeable but modelable ligands
(e.g. pentavalent phosphoranes) fall back to covalent-radius / hybridization
ideals with explicit trigonal-bipyramidal constraints.

The public entry :func:`generate_conformer` returns a coordinate-carrying pybel
``Molecule``; partial charges and atom names are assigned by the caller.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

# Stress-refine weights: exact (bond/angle/ring) distances tethered hard; bracket
# and vdW bounds are softer hinges (there are O(N^2) of them).
W_EXACT = 50.0
W_BOUND = 2.0
W_PLANE = 10.0
W_CHIRAL = 30.0
CHIRAL_MARGIN = 4.0  # target |chiral volume|; drives centers firmly pyramidal so
# the force-field cleanup cannot flip a near-planar stereocenter.
REFINE_ITERS = 200
STAGE_A_ITERS = 15  # LBFGS iters per 4D anneal level
STAGE_A_ANNEAL = (0.0, 0.5, 5.0, 50.0)  # 4th-dimension penalty ramp
N_RESTART = 5  # max random stage-A restarts; stop on all chiral signs correct
TORCH_DTYPE = torch.float32

_ELECTRONEGATIVITY = {
    1: 2.20,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    35: 2.96,
    53: 2.66,
}
_BOND_ORDER_SHORTENING = {1.0: 0.0, 1.5: 0.13, 2.0: 0.18, 3.0: 0.32}
_HYB_ANGLE = {
    Chem.HybridizationType.SP: 180.0,
    Chem.HybridizationType.SP2: 120.0,
    Chem.HybridizationType.SP3: 109.5,
}


def generate_conformer(
    smiles: str, *, minimize_steps: int = 50, seed: Optional[int] = None
):
    """Conformer generation script SMILES -> pybel.

    Uses a custom generation scheme similar to RDKit's ETKDG, where an embedding
    + minimization in the embedding space is used to generate a very good guess
    at the initial conformer.  Uses pytorch minimization machinery, followed
    by a very short MMFF minimization.

    Args:
        minimize_steps: OpenBabel conjugate-gradient steps for the final min.
        seed: Ooptional fixed RNG seed for reproducible coordinates.

    Raises:
        ValueError: Failures in parsing or final min.
    """
    _, pybel = _import_openbabel()
    obmol = _smiles_to_obmol(smiles)
    rd = _obmol_to_rdkit(obmol)
    if rd is None:
        raise ValueError(f"could not parse ligand chemistry for SMILES {smiles!r}")
    targets = _geometry_targets(rd)
    _set_coords(obmol, _embed_coordinates(targets, seed=seed))
    _forcefield_minimize(obmol, steps=minimize_steps, frozen=targets["frozen_atoms"])
    return pybel.Molecule(obmol)


# OpenBabel / RDKit interop
def _import_openbabel():
    from tmol.ligand.openbabel_compat import _import_openbabel as _imp

    return _imp()


def _smiles_to_obmol(smiles: str):
    _, pybel = _import_openbabel()
    pymol = pybel.readstring("smi", smiles)
    pymol.addh()
    return pymol.OBMol


def _obmol_to_rdkit(obmol) -> Optional[Chem.Mol]:
    """OBMol -> sanitized RDKit mol carrying stereo (via OB atom parity)."""
    from tmol.ligand.openbabel_compat import _obmol_to_rdkit_mol

    mol = _obmol_to_rdkit_mol(obmol, sanitize=True)
    if mol is not None:
        Chem.AssignAtomChiralTagsFromMolParity(mol)
    return mol


def _set_coords(obmol, coords: np.ndarray) -> None:
    for i in range(len(coords)):
        obmol.GetAtom(i + 1).SetVector(
            float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
        )


def _forcefield_minimize(obmol, *, steps: int, frozen=()) -> None:
    """In-place OpenBabel minimize:
    mmff94 by default
    uff fallback (for chemistries mmff94 cannot type)
    'frozen' atoms are held fixed"""
    openbabel, _ = _import_openbabel()
    for name in ("mmff94", "uff"):
        ff = openbabel.OBForceField.FindForceField(name)
        if ff is None:
            continue
        constraints = openbabel.OBFFConstraints()
        for i in frozen:
            constraints.AddAtomConstraint(int(i) + 1)  # OB is 1-indexed
        if ff.Setup(obmol, constraints):
            ff.ConjugateGradients(steps)
            ff.GetCoordinates(obmol)
            return
    raise ValueError("no force field could minimize the ligand")


# Geometry targets (ideal distances / angles / chirality)
def _ideal_13_distance(r12: float, r23: float, theta0_deg: float) -> float:
    t = math.radians(theta0_deg)
    return math.sqrt(r12 * r12 + r23 * r23 - 2 * r12 * r23 * math.cos(t))


def _chiral_volume_targets(rd: Chem.Mol):
    """Signed chiral-volume targets (c0,c1,c2,c3,sign), read off rd."""
    stable3 = {"S"}
    tag4 = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1.0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1.0,
    }
    tag3 = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    }
    out = []
    for a in rd.GetAtoms():
        tag = a.GetChiralTag()
        if tag not in tag4:
            continue
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) == 4:
            out.append(tuple(nb) + (tag4[tag],))
        elif len(nb) == 3 and a.GetSymbol() in stable3:
            out.append((a.GetIdx(),) + tuple(nb) + (tag3[tag],))
    return out


def _mmff_bonds_angles(rd: Chem.Mol, props):
    bonds = []
    for b in rd.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p = props.GetMMFFBondStretchParams(rd, i, j)
        if p:
            bonds.append((i, j, p[2]))
    angles = []
    for a in rd.GetAtoms():
        v = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        for m in range(len(nb)):
            for o in range(m + 1, len(nb)):
                p = props.GetMMFFAngleBendParams(rd, nb[m], v, nb[o])
                if p and p[2] > 0:
                    angles.append((nb[m], v, nb[o], p[2]))
    return bonds, angles, ()


def _fallback_bonds_angles(rd: Chem.Mol, elements):
    """MMFF-free ideals: covalent-radius bonds, hybridization angles. Degree-5
    centers get explicit trigonal-bipyramidal constraints (most electronegative
    substituents axial, by apicophilicity) and are frozen during the cleanup min.
    Returns (bonds, angles, frozen_atoms)."""
    pt = Chem.GetPeriodicTable()
    bonds = []
    for b in rd.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        r0 = (
            pt.GetRcovalent(elements[i])
            + pt.GetRcovalent(elements[j])
            - _BOND_ORDER_SHORTENING.get(b.GetBondTypeAsDouble(), 0.0)
        )
        bonds.append((i, j, r0))
    angles = []
    frozen = set()
    for a in rd.GetAtoms():
        v = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) == 5:  # trigonal bipyramidal
            ordered = sorted(
                nb, key=lambda k: -_ELECTRONEGATIVITY.get(elements[k], 2.5)
            )
            axial, equatorial = ordered[:2], ordered[2:]
            angles.append((axial[0], v, axial[1], 180.0))
            for x in axial:
                for y in equatorial:
                    angles.append((x, v, y, 90.0))
            for m in range(len(equatorial)):
                for o in range(m + 1, len(equatorial)):
                    angles.append((equatorial[m], v, equatorial[o], 120.0))
            frozen.update([v] + nb)
        elif len(nb) < 5:
            th = _HYB_ANGLE.get(a.GetHybridization(), 109.5)
            for m in range(len(nb)):
                for o in range(m + 1, len(nb)):
                    angles.append((nb[m], v, nb[o], th))
    return bonds, angles, tuple(sorted(frozen))


def _planar_systems(rd: Chem.Mol, bonds):
    """Fused planar atom sets: joined by an aromatic bond or a small (<=7) all-sp2
    ring. Returns (components, components extended with first-shell neighbors)."""
    n = rd.GetNumAtoms()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    sp2 = Chem.HybridizationType.SP2
    planar = set()
    for b in rd.GetBonds():
        if b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            union(i, j)
            planar.update((i, j))
    for ring in rd.GetRingInfo().AtomRings():
        if len(ring) <= 7 and all(
            rd.GetAtomWithIdx(a).GetIsAromatic()
            or rd.GetAtomWithIdx(a).GetHybridization() == sp2
            for a in ring
        ):
            for a in ring:
                union(a, ring[0])
                planar.add(a)
    groups: dict[int, list] = {}
    for a in planar:
        groups.setdefault(find(a), []).append(a)
    components = [tuple(sorted(v)) for v in groups.values() if len(v) >= 3]
    nbr: dict[int, set] = {}
    for i, j, _ in bonds:
        nbr.setdefault(i, set()).add(j)
        nbr.setdefault(j, set()).add(i)
    sets = [
        tuple(sorted(set(c) | {b for a in c for b in nbr.get(a, ())}))
        for c in components
    ]
    return components, sets


def _geometry_targets(rd: Chem.Mol) -> dict:
    elements = [a.GetAtomicNum() for a in rd.GetAtoms()]
    props = AllChem.MMFFGetMoleculeProperties(rd)
    if props is not None:
        bonds, angles, frozen = _mmff_bonds_angles(rd, props)
    else:
        bonds, angles, frozen = _fallback_bonds_angles(rd, elements)
    components, sets = _planar_systems(rd, bonds)
    pt = Chem.GetPeriodicTable()
    return dict(
        n=rd.GetNumAtoms(),
        elements=elements,
        vdw=[pt.GetRvdw(z) for z in elements],
        bonds=bonds,
        angles=angles,
        planar_components=components,
        planar_sets=sets,
        chirals=_chiral_volume_targets(rd),
        frozen_atoms=frozen,
    )


# Distance-bounds matrix + smoothing + embedding
def _planar_component_distances(atoms, r0, ang, max_iters: int = 150):
    """Given one planar ring system, compute the exact distance between
    _every_ pair of its atoms assuming it's perfectly flat.

    Do this using the same trick as the whole molecule (embed + min)
    but in 2D."""
    idx = {a: k for k, a in enumerate(atoms)}
    m = len(atoms)
    aset = set(atoms)
    tp = []
    for a in atoms:
        for b in atoms:
            if a < b and (a, b) in r0:
                tp.append((idx[a], idx[b], r0[(a, b)]))
    for (lo, v, hi), th0 in ang.items():
        if lo in aset and hi in aset and v in aset and (lo, v) in r0 and (v, hi) in r0:
            tp.append(
                (idx[lo], idx[hi], _ideal_13_distance(r0[(lo, v)], r0[(v, hi)], th0))
            )
    if len(tp) < m:
        return None
    Lc = np.zeros((m, m))
    Uc = np.full((m, m), 1e5)
    np.fill_diagonal(Uc, 0.0)
    for i, j, t in tp:
        Lc[i, j] = Lc[j, i] = Uc[i, j] = Uc[j, i] = t
    Lc, Uc = _smooth_bounds(Lc, Uc)
    Dm = 0.5 * (Lc + Uc)
    Jc = np.eye(m) - 1.0 / m
    w, V = np.linalg.eigh(-0.5 * (Jc @ (Dm**2) @ Jc))
    o = np.argsort(w)[::-1]
    Y0 = V[:, o[:2]] * np.sqrt(np.clip(w[o[:2]], 0.0, None))
    I = torch.tensor([p[0] for p in tp])
    J = torch.tensor([p[1] for p in tp])
    T = torch.tensor([p[2] for p in tp], dtype=torch.double)
    Y = torch.tensor(np.ascontiguousarray(Y0), dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([Y], max_iter=max_iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        d = ((Y[I] - Y[J]) ** 2).sum(1).clamp_min(1e-12).sqrt()
        loss = ((d - T) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    Yc = Y.detach().numpy()
    return {
        (atoms[k], atoms[l]): float(np.linalg.norm(Yc[k] - Yc[l]))
        for k in range(m)
        for l in range(k + 1, m)
    }


def _bounds_matrix(targets: dict):
    """Build the distance bounds matrix:
    * 1-2 and 1-3 atom distances exact given ideal geometry
    * planar pairs in same ring system exact given 2D embedding
    * lower bounded given VDW radii for all others"""
    n = targets["n"]
    vdw = np.array(targets["vdw"], float)
    L = vdw[:, None] + vdw[None, :]
    U = np.full((n, n), 1e5)
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(U, 0.0)
    exact = np.zeros((n, n), bool)

    def fix(i, j, d):
        L[i, j] = L[j, i] = U[i, j] = U[j, i] = d
        exact[i, j] = exact[j, i] = True

    r0 = {}
    for i, j, d in targets["bonds"]:
        r0[(i, j)] = r0[(j, i)] = d
    ang = {}
    for i, v, j, th0 in targets["angles"]:
        ang[(min(i, j), v, max(i, j))] = th0
    for i, j, d in targets["bonds"]:
        fix(i, j, d)
    for i, v, j, th0 in targets["angles"]:
        if not exact[i, j]:
            fix(i, j, _ideal_13_distance(r0[(i, v)], r0[(v, j)], th0))
    for comp in targets["planar_components"]:
        dists = _planar_component_distances(comp, r0, ang)
        if dists:
            for (i, j), d in dists.items():
                fix(i, j, d)
    L = np.minimum(L, U)
    return L, U


def _smooth_bounds(L, U, lower_passes: int = 2):
    """Triangle-inequality smoothing:
    * upper bound via Floyd-Warshall shortest path,
    * lower bound via the inverse triangle inequality."""
    n = len(U)
    U = U.copy()
    L = L.copy()
    for k in range(n):
        U = np.minimum(U, U[:, k, None] + U[None, k, :])
    for _ in range(lower_passes):
        for k in range(n):
            cand = L[:, k, None] - U[k, None, :]
            L = np.maximum(L, cand)
            L = np.maximum(L, cand.T)
    np.fill_diagonal(L, 0.0)
    L = np.minimum(L, U)
    return L, U


def _metric_embed(L, U, seed: int) -> np.ndarray:
    """The main embedding:
    - randomly sample distances in [L, U],
    - double-center to Gram mtx
    - eigendecomposition to get coordinates."""
    n = len(L)
    rng = np.random.default_rng(seed)
    D = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    D[iu] = rng.uniform(L[iu], U[iu])
    D = D + D.T
    J = np.eye(n) - 1.0 / n
    G = -0.5 * (J @ (D**2) @ J)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    return V[:, order[:3]] * np.sqrt(np.clip(w[order[:3]], 0.0, None))


def _pair_masks(L0, U0, L, U):
    tri = np.triu(np.ones_like(L0, bool), 1)
    exact = tri & np.isclose(L0, U0)
    nonex = tri & ~exact
    finite_u = nonex & (U < 1e4)
    T = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    ei, ej = np.nonzero(exact)
    ni, nj = np.nonzero(nonex)
    fi, fj = np.nonzero(finite_u)
    return (
        T(ei),
        T(ej),
        T(ni),
        T(nj),
        T(fi),
        T(fj),
        torch.as_tensor(L0[exact], dtype=TORCH_DTYPE),
        torch.as_tensor(L[nonex], dtype=TORCH_DTYPE),
        torch.as_tensor(U[finite_u], dtype=TORCH_DTYPE),
    )


def _chiral_ok(X: np.ndarray, chirals):
    """
    Check stereochemical agreement to target.

    returns (n_matching, n_total) chiral centers."""
    ch = np.array(chirals, float).reshape(-1, 5)
    if len(ch) == 0:
        return (0, 0)
    c = ch[:, :4].astype(int)
    v = np.einsum(
        "ij,ij->i",
        X[c[:, 1]] - X[c[:, 0]],
        np.cross(X[c[:, 2]] - X[c[:, 0]], X[c[:, 3]] - X[c[:, 0]]),
    )
    return (int(np.sum(np.sign(v) == np.sign(ch[:, 4]))), len(ch))


def _chiral_anneal(X0, L0, U0, L, U, components, chirals, rng_seed: int) -> np.ndarray:
    """The first stage of minimization.

    Following EKTDG, run in 4D to allow easier chirality flipping.
    Ramping constraint tethers 4th dim to 0.
    Return the 3D projection."""
    ei, ej, ni, nj, fi, fj, tgt, lo, hi = _pair_masks(L0, U0, L, U)
    comps = [torch.as_tensor(c, dtype=torch.long) for c in components if len(c) >= 4]
    ch = np.array(chirals, float).reshape(-1, 5)
    c0, c1, c2, c3 = (
        torch.as_tensor(ch[:, k].astype(int), dtype=torch.long) for k in range(4)
    )
    csgn = torch.as_tensor(ch[:, 4], dtype=TORCH_DTYPE)
    w0 = np.random.default_rng(rng_seed).normal(0, 0.5, (X0.shape[0], 1))
    X = torch.tensor(
        np.ascontiguousarray(np.hstack([X0, w0])), dtype=TORCH_DTYPE, requires_grad=True
    )

    def pd(i, j):
        return ((X[i] - X[j]) ** 2).sum(1).clamp_min(1e-12).sqrt()

    def chiral_penalty():
        P = X[:, :3]
        v = ((P[c1] - P[c0]) * torch.linalg.cross(P[c2] - P[c0], P[c3] - P[c0])).sum(1)
        return (torch.relu(CHIRAL_MARGIN - csgn * v) ** 2).sum()

    def run(iters, w_dim4):
        opt = torch.optim.LBFGS([X], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = W_EXACT * ((pd(ei, ej) - tgt) ** 2).sum()
            loss = loss + W_BOUND * (torch.relu(lo - pd(ni, nj)) ** 2).sum()
            loss = loss + W_BOUND * (torch.relu(pd(fi, fj) - hi) ** 2).sum()
            for c in comps:
                P = X[c][:, :3] - X[c][:, :3].mean(0)
                loss = loss + W_PLANE * torch.linalg.eigvalsh(P.t() @ P)[0]
            loss = loss + W_CHIRAL * chiral_penalty()
            if w_dim4 > 0:
                loss = loss + w_dim4 * (X[:, 3] ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)

    for w_dim4 in STAGE_A_ANNEAL:
        run(STAGE_A_ITERS, w_dim4)
    return X.detach().numpy()[:, :3]


def _stress_refine(X0, L0, U0, L, U, components, chirals) -> np.ndarray:
    """The 2nd stage of minimization.

    Full-weight 3D refine."""
    ei, ej, ni, nj, fi, fj, tgt, lo, hi = _pair_masks(L0, U0, L, U)
    X = torch.tensor(np.ascontiguousarray(X0), dtype=TORCH_DTYPE, requires_grad=True)
    comps = [torch.as_tensor(c, dtype=torch.long) for c in components if len(c) >= 4]
    ch = np.array(chirals, float).reshape(-1, 5)
    c0, c1, c2, c3 = (
        torch.as_tensor(ch[:, k].astype(int), dtype=torch.long) for k in range(4)
    )
    csgn = torch.as_tensor(ch[:, 4], dtype=TORCH_DTYPE)

    def pd(i, j):
        return ((X[i] - X[j]) ** 2).sum(1).clamp_min(1e-12).sqrt()

    def chiral_penalty():
        if len(csgn) == 0:
            return X.sum() * 0.0
        v = ((X[c1] - X[c0]) * torch.linalg.cross(X[c2] - X[c0], X[c3] - X[c0])).sum(1)
        return (torch.relu(CHIRAL_MARGIN - csgn * v) ** 2).sum()

    opt = torch.optim.LBFGS([X], max_iter=REFINE_ITERS, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = W_EXACT * ((pd(ei, ej) - tgt) ** 2).sum()
        loss = loss + W_BOUND * (torch.relu(lo - pd(ni, nj)) ** 2).sum()
        loss = loss + W_BOUND * (torch.relu(pd(fi, fj) - hi) ** 2).sum()
        for c in comps:
            P = X[c] - X[c].mean(0)
            loss = loss + W_PLANE * torch.linalg.eigvalsh(P.t() @ P)[0]
        loss = loss + W_CHIRAL * chiral_penalty()
        loss.backward()
        return loss

    opt.step(closure)
    return X.detach().numpy()


def _embed_coordinates(targets: dict, *, seed: Optional[int] = None) -> np.ndarray:
    """Outermost embedding loop:
    - run stage 1 from random conditions until chirals are all correct
      (at most N_RESTART times)
    - run stage 2
    - return final embedding
    """
    L0, U0 = _bounds_matrix(targets)
    L, U = _smooth_bounds(L0, U0)
    components = targets["planar_sets"]
    chirals = targets["chirals"]
    seeds = np.random.SeedSequence(seed).generate_state(N_RESTART)
    if chirals:
        best = None
        for s in seeds:
            X = _chiral_anneal(
                _metric_embed(L, U, int(s)), L0, U0, L, U, components, chirals, int(s)
            )
            n_ok, n_total = _chiral_ok(X, chirals)
            if best is None or n_ok > best[0]:
                best = (n_ok, X)
            if n_ok == n_total:
                break
        X0 = best[1]
    else:
        X0 = _metric_embed(L, U, int(seeds[0]))
    return _stress_refine(X0, L0, U0, L, U, components, chirals)
