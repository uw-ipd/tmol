"""Audit the prepared biotin amide's local parameter ownership and geometry.

This compares tmol's installed records with a topology-derived MMFF diagnostic;
it does not install a new force field or infer equilibrium geometry from a CIF.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import cattr
import numpy as np
import biotite.structure as struc
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.ligand._conjugation_patches import conjugated_chemistry
from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array
from tmol.ligand._detect import _dimorphite_protonate_smiles
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.score.genbonded._genbonded_energy_term import GenBondedEnergyTerm
from tmol.tests.data import data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    aa = atom_array_from_cif(data_path("covalent_fixtures", "lys_biotin_1bdo.cif"))
    pose, context = pose_stack_from_biotite(
        aa,
        torch.device(args.device),
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
        ligand_seed=20250828,
    )
    db = context.parameter_database
    bts = [
        pose.packed_block_types.active_block_types[t]
        for t in pose.block_type_ind[0].tolist()
    ]
    bi = next(i for i, bt in enumerate(bts) if bt.name.startswith("LYS:conj_NZ"))
    bt = bts[bi]
    conn = next(i for i, c in enumerate(bt.connections) if c.name == "conj_NZ")
    bj, cj = pose.inter_residue_connections[0, bi, conn].tolist()
    partner = bts[bj]
    nz = bt.atom_to_idx["NZ"]
    neighbors = sorted({int(b) for a, b in bt.bond_indices if int(a) == nz})
    element = {t.name: t.element for t in db.chemical.atom_types}
    hn = next(i for i in neighbors if element[bt.atoms[i].atom_type] == "H")
    ce = bt.atom_to_idx["CE"]
    partner_atom = int(partner.ordered_connection_atoms[cj])
    cart = CartBondedEnergyTerm(db, pose.device)
    generic = GenBondedEnergyTerm(db, pose.device)
    modules = {}
    for name, term in (("cartbonded", cart), ("genbonded", generic)):
        for block_type in pose.packed_block_types.active_block_types:
            term.setup_block_type(block_type)
        term.setup_packed_block_types(pose.packed_block_types)
        term.setup_poses(pose)
        modules[name] = term.render_whole_pose_scoring_module(pose)
    local_names = [bt.atoms[i].name for i in neighbors]
    types = [
        bt.atoms[nz].atom_type,
        *(bt.atoms[i].atom_type for i in neighbors),
        partner.atoms[partner_atom].atom_type,
    ]
    charge_rows = {
        (p.res, p.atom): p.charge for p in db.scoring.elec.atom_charge_parameters
    }
    charges = {
        a.name: charge_rows.get(
            (bt.name, a.name), charge_rows.get((bt.base_name, a.name))
        )
        for a in bt.atoms
    }
    result = {
        "device": str(pose.device),
        "block_type": bt.name,
        "partner_type": partner.name,
        "site_and_neighbor_types": types,
        "site_neighbors": local_names + ["+" + partner.atoms[partner_atom].name],
        "cart_improper_root_present": "NZ" in cart.improper_roots,
        "cart_torsions_containing_remaining_hydrogen": [
            cattr.unstructure(p)
            for p in db.scoring.cartbonded.residue_params["LYS"].torsion_parameters
            if bt.atoms[hn].name in (p.atm1, p.atm2, p.atm3, p.atm4)
        ],
        "gen_improper_parameter": cattr.unstructure(
            db.scoring.genbonded.find_improper_params(*types)
        ),
        "local_cart_angle_records": [
            cattr.unstructure(p)
            for p in db.scoring.cartbonded.residue_params["LYS"].angle_parameters
            if p.atm2 == "NZ" and {p.atm1, p.atm3} <= set(local_names)
        ],
        "effective_lysine_charges": charges,
        "effective_lysine_net_charge": sum(charges.values()),
    }
    xyz = pose.coords.double()
    start = int(pose.block_coord_offset[0, bi])
    other = int(pose.block_coord_offset[0, bj]) + partner_atom
    origin = xyz[0, start + nz]
    axis = xyz[0, start + ce] - origin
    axis /= axis.norm()
    plane = xyz[0, other] - origin
    plane -= plane.dot(axis) * axis
    plane /= plane.norm()
    normal = torch.linalg.cross(axis, plane)
    bond_length = float((xyz[0, start + hn] - origin).norm())
    samples = []
    for theta, phi in itertools.product(
        (90.0, 109.5, 120.0, 130.0), (0.0, 30.0, 60.0, 90.0, 180.0)
    ):
        t, p = math.radians(theta), math.radians(phi)
        direction = math.cos(t) * axis - math.sin(t) * (
            math.cos(p) * plane + math.sin(p) * normal
        )
        coords = xyz.clone()
        coords[0, start + hn] = origin + bond_length * direction
        values = {
            name: module(coords).detach().cpu().reshape(-1).tolist()
            for name, module in modules.items()
        }
        samples.append(dict(angle_deg=theta, out_of_plane_deg=phi, values=values))
    result["hydrogen_scans"] = samples

    heavy = aa[aa.element != "H"]
    boundaries = struc.get_residue_starts(heavy, add_exclusive_stop=True)
    indices = np.repeat(np.arange(len(boundaries) - 1), np.diff(boundaries))
    a, b = next(
        (int(a), int(b))
        for a, b, _ in heavy.bonds.as_array()
        if {str(heavy.atom_name[a]), str(heavy.atom_name[b])} == {"NZ", "C11"}
        and indices[a] != indices[b]
    )
    selected = np.flatnonzero(np.isin(indices, [indices[a], indices[b]]))
    pair = heavy[selected].copy()
    pair.coord[:] = np.nan
    smi = _dimorphite_protonate_smiles(
        ligand_smiles_from_atom_array(pair, with_atom_map=True), ph=7.4
    )
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    props = AllChem.MMFFGetMoleculeProperties(mol)
    nz_atom = next(
        at
        for at in mol.GetAtoms()
        if at.GetAtomMapNum() and pair.atom_name[at.GetAtomMapNum() - 1] == "NZ"
    )
    ni = nz_atom.GetIdx()
    nbs = [a.GetIdx() for a in nz_atom.GetNeighbors()]
    result["topology_reference"] = {
        "scope": "Uncapped residue pair: its formal charge includes artificial free backbone termini; it is not a net conjugate charge reference.",
        "smiles": smi,
        "formal_charge": Chem.GetFormalCharge(mol),
        "nz_mmff_type": props.GetMMFFAtomType(ni),
        "nz_mmff_charge": props.GetMMFFPartialCharge(ni),
        "nz_angles": [
            props.GetMMFFAngleBendParams(mol, i, ni, j)
            for i, j in itertools.combinations(nbs, 2)
        ],
        "nz_oop_parameter": props.GetMMFFOopBendParams(mol, nbs[0], ni, nbs[1], nbs[2]),
    }
    lys = pair[pair.res_name == "LYS"]
    btn = pair[pair.res_name == "BTN"]
    result["existing_conjugated_chemistry_helper"] = conjugated_chemistry(
        lys, "NZ", btn, "C11"
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
