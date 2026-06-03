"""Read and write classic Rosetta .params files.

Provides interoperability with Rosetta's params file format for ligand
residue types. Writing produces a standard Rosetta-compatible params file;
reading parses one into a tmol RawResidueType.
"""

import logging
import math
from pathlib import Path

from tmol.database.chemical import (
    Atom,
    ChemicalProperties,
    ChiSamples,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
    Torsion,
    UnresolvedAtom,
)

logger = logging.getLogger(__name__)

_BOND_TOK_TO_TYPE = {
    "1": "SINGLE",
    "2": "DOUBLE",
    "3": "TRIPLE",
    "4": "AROMATIC",
    "SINGLE": "SINGLE",
    "DOUBLE": "DOUBLE",
    "TRIPLE": "TRIPLE",
    "AROMATIC": "AROMATIC",
    "ARO": "AROMATIC",
}


def _chi_number(name: str) -> int:
    """Extract the integer N from a chi torsion name like ``chi3`` (else 0)."""
    if name.startswith("chi") and name[3:].isdigit():
        return int(name[3:])
    return 0


def write_params_file(
    restype: RawResidueType,
    path: str | Path,
    partial_charges: dict[str, float] | None = None,
) -> None:
    """Write a RawResidueType as a Rosetta .params file.

    Args:
        restype: The residue type to write.
        path: Output file path.
        partial_charges: Optional per-atom partial charges. Keys are atom
            names matching restype.atoms[i].name.
    """
    lines: list[str] = []

    lines.append(f"NAME {restype.name}")
    lines.append(f"IO_STRING {restype.name3} Z")
    lines.append("TYPE LIGAND")
    lines.append("AA UNK")

    for atom in restype.atoms:
        charge = 0.0
        if partial_charges and atom.name in partial_charges:
            charge = partial_charges[atom.name]
        lines.append(f"ATOM {atom.name:4s} {atom.atom_type:4s} X {charge:8.4f}")

    for a, b, c, *rest in restype.bonds:
        is_ring = bool(rest[0]) if rest else False
        ring_tok = " RING" if is_ring else ""
        lines.append(f"BOND_TYPE {a:4s} {b:4s} {c:12s}{ring_tok}")

    if restype.default_jump_connection_atom:
        lines.append(f"NBR_ATOM {restype.default_jump_connection_atom}")

    lines.append("NBR_RADIUS 999.0")

    # CHI / PROTON_CHI rotatable-bond DOFs. One CHI line per named torsion;
    # PROTON_CHI lines carry the sample/expansion data for polar-hydrogen chis.
    # Rosetta-only annotations (e.g. a trailing "#biaryl" comment) are NOT
    # emitted — tmol keeps only the semantic content.
    proton_by_chi = {cs.chi_dihedral: cs for cs in restype.chi_samples}
    for tor in sorted(restype.torsions, key=lambda t: _chi_number(t.name)):
        n = _chi_number(tor.name)
        quad = " ".join(f"{(ua.atom or ''):>4s}" for ua in (tor.a, tor.b, tor.c, tor.d))
        lines.append(f"CHI {n:>2d} {quad}")
        cs = proton_by_chi.get(tor.name)
        if cs is not None:
            samples = " ".join(f"{s:g}" for s in cs.samples)
            if cs.expansions:
                extra = f"EXTRA {len(cs.expansions)} " + " ".join(
                    f"{e:g}" for e in cs.expansions
                )
            else:
                extra = "EXTRA 0"
            lines.append(f"PROTON_CHI {n} SAMPLES {len(cs.samples)} {samples} {extra}")

    for ic in restype.icoors:
        phi_deg = math.degrees(ic.phi)
        theta_deg = math.degrees(ic.theta)
        lines.append(
            f"ICOOR_INTERNAL {ic.name:4s} {phi_deg:11.6f} {theta_deg:11.6f} "
            f"{ic.d:11.6f} {ic.parent:4s} {ic.grand_parent:4s} "
            f"{ic.great_grand_parent:4s}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    logger.info("Wrote params file for %s to %s", restype.name, path)


def read_params_file(path: str | Path) -> RawResidueType:
    """Read a Rosetta .params file into a RawResidueType.

    Parses ATOM, BOND, ICOOR_INTERNAL, and NBR_ATOM records. Other
    records are silently ignored.

    Args:
        path: Path to the .params file.

    Returns:
        A RawResidueType populated from the params file.
    """
    name = "UNK"
    name3 = "UNK"
    atoms: list[Atom] = []
    bonds: list[tuple[str, str, str, bool]] = []
    icoors: list[Icoor] = []
    nbr_atom = ""
    chi_torsions: dict[int, Torsion] = {}
    chi_proton: dict[int, ChiSamples] = {}

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "NAME" and len(parts) >= 2:
                name = parts[1]
                name3 = name

            elif parts[0] == "ATOM" and len(parts) >= 4:
                atoms.append(Atom(name=parts[1], atom_type=parts[2]))

            elif parts[0] == "BOND" and len(parts) >= 3:
                bond_type = "SINGLE"
                if len(parts) >= 4:
                    bond_type = _BOND_TOK_TO_TYPE.get(parts[3].upper(), "SINGLE")
                bonds.append((parts[1], parts[2], bond_type, False))
            elif parts[0] == "BOND_TYPE" and len(parts) >= 4:
                order = _BOND_TOK_TO_TYPE.get(parts[3].upper(), "SINGLE")
                ring = len(parts) >= 5 and parts[4].upper() == "RING"
                bonds.append((parts[1], parts[2], order, ring))

            elif parts[0] == "CHI" and len(parts) >= 6:
                # "CHI n a b c d [#biaryl ...]" — trailing comments ignored.
                n = int(parts[1])
                chi_torsions[n] = Torsion(
                    name=f"chi{n}",
                    a=UnresolvedAtom(atom=parts[2]),
                    b=UnresolvedAtom(atom=parts[3]),
                    c=UnresolvedAtom(atom=parts[4]),
                    d=UnresolvedAtom(atom=parts[5]),
                )

            elif parts[0] == "PROTON_CHI" and "SAMPLES" in parts:
                # "PROTON_CHI n SAMPLES k v1..vk [EXTRA m e1..em]"
                n = int(parts[1])
                si = parts.index("SAMPLES")
                k = int(parts[si + 1])
                samples = tuple(float(x) for x in parts[si + 2 : si + 2 + k])
                expansions: tuple[float, ...] = ()
                if "EXTRA" in parts:
                    ei = parts.index("EXTRA")
                    n_extra = int(parts[ei + 1])
                    expansions = tuple(
                        float(x) for x in parts[ei + 2 : ei + 2 + n_extra]
                    )
                chi_proton[n] = ChiSamples(
                    chi_dihedral=f"chi{n}", samples=samples, expansions=expansions
                )

            elif parts[0] == "NBR_ATOM" and len(parts) >= 2:
                nbr_atom = parts[1]

            elif parts[0] == "ICOOR_INTERNAL" and len(parts) >= 8:
                icoors.append(
                    Icoor(
                        name=parts[1],
                        phi=math.radians(float(parts[2])),
                        theta=math.radians(float(parts[3])),
                        d=float(parts[4]),
                        parent=parts[5],
                        grand_parent=parts[6],
                        great_grand_parent=parts[7],
                    )
                )

    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=False,
            polymer_type="NA",
            backbone_type="NA",
            mainchain_atoms=None,
            sidechain_chirality="NA",
            termini_variants=(),
        ),
        chemical_modifications=(),
        connectivity=(),
        protonation=ProtonationProperties(
            protonated_atoms=(),
            protonation_state="neutral",
            pH=7,
        ),
        virtual=(),
    )

    return RawResidueType(
        name=name,
        base_name=name,
        name3=name3,
        io_equiv_class=name3,
        atoms=tuple(atoms),
        atom_aliases=(),
        bonds=tuple(bonds),
        connections=(),
        torsions=tuple(chi_torsions[n] for n in sorted(chi_torsions)),
        icoors=tuple(icoors),
        properties=properties,
        chi_samples=tuple(chi_proton[n] for n in sorted(chi_proton)),
        default_jump_connection_atom=nbr_atom or (atoms[0].name if atoms else ""),
    )
