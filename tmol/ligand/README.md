# tmol Ligand Preparation Pipeline

Automated preparation of non-standard residues (ligands, modified amino acids, etc.) for loading into tmol's PoseStack.

## Pipeline Overview

```
Biotite AtomArray (CIF/PDB input)
        │
        ▼
detect_nonstandard_residues()
  Filter residues not in ChemicalDatabase
  Classify via Biotite CCD (modified AA vs ligand)
        │
        ▼
perceive_smiles()
  OpenBabel SMILES from atom coordinates
        │
        ▼
protonate_ligand_smiles()
  dimorphite_dl at target pH (default 7.4)
        │
        ▼
smiles_to_obmol()
  OpenBabel 3D structure generation
  MMFF94 partial charges + energy minimization
        │
        ▼
assign_tmol_atom_types()
  Element + bonding → Rosetta generic_potential types
        │
        ▼
build_residue_type()
  RawResidueType with atoms, bonds, icoors, properties
        │
        ▼
register_ligand()
  Add to ChemicalDatabase + rebuild CanonicalOrdering
        │
        ▼
pose_stack_from_biotite()
  Normal PoseStack loading (ligands are now known residues)
```

## Quick Start

```python
from tmol.ligand import prepare_ligands
import biotite.structure.io

# Load a structure with a ligand
atom_array = biotite.structure.io.load_structure("protein_with_ligand.cif")

# Prepare all non-standard residues
chem_db, canonical_ordering = prepare_ligands(atom_array)

# The ligands are now registered and can be loaded normally
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `detect.py` | Detect non-standard residues, classify via CCD |
| `smiles.py` | SMILES perception (OpenBabel) + protonation (dimorphite_dl) |
| `mol3d.py` | 3D generation, MMFF94 charges, minimization |
| `atom_typing.py` | Map element + bonding to Rosetta atom types |
| `residue_builder.py` | Build RawResidueType (atoms, bonds, icoors) |
| `registry.py` | Register in ChemicalDatabase, rebuild CanonicalOrdering |
| `params_io.py` | Optional: read/write classic Rosetta .params files |

## Standalone Module Usage

Each module can be used independently:

```python
# Detect non-standard residues
from tmol.ligand.detect import detect_nonstandard_residues
ligands = detect_nonstandard_residues(atom_array, canonical_ordering)

# Perceive SMILES from coordinates
from tmol.ligand.smiles import perceive_smiles, protonate_ligand_smiles
smi = perceive_smiles(ligand_info)
protonated = protonate_ligand_smiles(smi, ph=7.4)

# Generate 3D structure with charges
from tmol.ligand.mol3d import smiles_to_obmol, get_partial_charges
mol = smiles_to_obmol(protonated)
charges = get_partial_charges(mol)

# Assign atom types
from tmol.ligand.atom_typing import assign_tmol_atom_types
atom_types = assign_tmol_atom_types(mol.OBMol)

# Build residue type
from tmol.ligand.residue_builder import build_residue_type
restype = build_residue_type(mol.OBMol, "LIG", atom_types)

# Write Rosetta params file (optional)
from tmol.ligand.params_io import write_params_file
write_params_file(restype, "LIG.params", partial_charges=charges)
```

## Supported Atom Types

The pipeline maps ligand atoms to Rosetta's generic_potential atom types.
All types have corresponding LJLK scoring parameters in `ljlk.yaml`.

**Carbon:** CS, CS1, CS2, CS3, CSp, CSQ, CD, CD1, CD2, CDp, CR, CRp, CT, CTp
**Hydrogen:** HC, HN (polar), HO (polar), HR, HG (polar)
**Nitrogen:** Nad, Nad3, Nam, Nam2, Ngu1, Ngu2, Nim, Nin, NG1-NG3, NG21, NG22
**Oxygen:** Oad, Oal, Oat, Oet, Ofu, Ohx, Ont, OG2, OG3, OG31
**Sulfur:** SR, Ssl, Sth, SG2, SG3, SG5
**Phosphorus:** PG3, PG5, Phos
**Halogens:** FR (F), ClR (Cl), BrR (Br), IR (I)

## Dependencies

- **biotite** - Structure I/O and CCD access
- **openbabel-wheel** - SMILES perception, 3D generation, MMFF94 charges
- **dimorphite_dl** - pH-dependent protonation

### System dependency: X11 libraries

`openbabel-wheel` bundles native format plugins that link against X11 shared
libraries (`libXrender`, `libXext`, `libX11`). These are present on most
desktop Linux systems but may be missing on headless servers, minimal Docker
containers, or NGC GPU containers.

If you see errors like `libXrender.so.1: cannot open shared object file` or
`libXext.so.6: cannot open shared object file`, install the X11 libraries:

```bash
# Ubuntu / Debian
sudo apt-get install libxrender1 libxext6

# Fedora / RHEL
sudo dnf install libXrender libXext

# Conda (any platform)
conda install -c conda-forge xorg-libxrender xorg-libxext
```

The ligand pipeline uses lazy imports so that `import tmol` works even without
these libraries installed. The error only surfaces when ligand preparation
functions are actually called.

## Limitations

- Metals and metallocenes are not currently handled
- Conformer sampling is single-conformer (best MMFF94 minimum)
- Modified amino acids are treated as standalone residues (no polymer connections)
- HBond donor/acceptor parameters for novel atom types require manual curation
