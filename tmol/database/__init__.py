from __future__ import annotations

import os
import attr
from typing import List, Optional, Mapping

from .chemical import (  # noqa: F401
    AtomType,
    ChemicalDatabase,
    RawResidueType,
    l_base_name,
)
from ._patched_chemdb import PatchedChemicalDatabase  # noqa: F401
from .scoring import ScoringDatabase  # noqa: F401
from .scoring._elec import PartialCharges  # noqa: F401
from .scoring._cartbonded import CartRes  # noqa: F401
from .scoring._mirrored_dunbrack import with_mirrored_libraries


@attr.s(frozen=True)
class ParameterDatabase:
    """Immutable chemical and scoring parameter container used by tmol.

    The process-global accessor ``get_default()`` returns a shared read-only
    instance.  To add ligand or custom residue data, use
    :func:`inject_residue_params` which returns a **new** database.
    """

    __default = None

    @classmethod
    def get_default(cls) -> "ParameterDatabase":
        """Return the process-global cached parameter database (read-only)."""
        if cls.__default is None:
            cls.__default = ParameterDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "default")
            )
        return cls.__default

    scoring: ScoringDatabase = attr.ib()
    chemical: PatchedChemicalDatabase = attr.ib()

    @classmethod
    def from_file(cls, path: str) -> "ParameterDatabase":
        chemdb = ChemicalDatabase.from_file(os.path.join(path, "chemical"))
        patched_chemdb = PatchedChemicalDatabase.from_chem_db(chemdb)
        scoring = ScoringDatabase.from_file(os.path.join(path, "scoring"))

        # a d-amino acid's rotamer statistics are its l form's at negated
        #    torsions; the mirrored libraries are built here rather than stored,
        #    since which residues need them is a property of the chemical database
        d_names = {
            l_base_name(residue): residue.name
            for residue in patched_chemdb.residues
            if residue.name == residue.base_name
            and residue.properties.polymer.sidechain_chirality == "d"
        }
        if d_names:
            scoring = attr.evolve(
                scoring, dun=with_mirrored_libraries(scoring.dun, d_names)
            )
        return cls(scoring=scoring, chemical=patched_chemdb)

    def with_symmetric_gly(self) -> "ParameterDatabase":
        """A copy whose glycine backbone tables are mirror-symmetric.

        Glycine is achiral, but the tables derived from PDB statistics are not,
        so by default a structure and its mirror image score differently. This
        points glycine at the symmetrized tables instead; every other residue is
        untouched, and the chirality of the other 19 is carried by their own
        lookup rows.

        Glycine's bbdep-omega tables become uniformly trans, since an achiral
        residue has no backbone-dependent omega preference to keep.
        """
        rama = self.scoring.rama
        omega = self.scoring.omega_bbdep

        def retarget(rows, mapping):
            return tuple(
                (
                    attr.evolve(row, table_id=mapping[row.table_id])
                    if row.res_middle == "GLY" and row.table_id in mapping
                    else row
                )
                for row in rows
            )

        rama = attr.evolve(
            rama,
            rama_lookup=retarget(
                rama.rama_lookup,
                {"GLY": "GLY_symm", "GLY_prepro": "GLY_prepro_symm"},
            ),
        )
        omega = attr.evolve(
            omega,
            bbdep_omega_lookup=retarget(
                omega.bbdep_omega_lookup,
                {"gly": "gly_symm", "prepro": "prepro_gly_symm"},
            ),
        )
        return attr.evolve(
            self,
            scoring=attr.evolve(
                self.scoring,
                rama=attr.evolve(rama, uniq_id=rama.content_id()),
                omega_bbdep=attr.evolve(omega, uniq_id=omega.content_id()),
            ),
        )

    def create_stable_subset(
        self, desired_names: List[str], desired_variants: List[str]
    ) -> "ParameterDatabase":
        """Create a ParameterDatabase representing a subset of the
        RefinedResidueTypes in this PD's PatchedChemicalDatabase from a list
        of RRT names and patched with the given variants (identified by their
        display names) where the order in which RRTs will appear in the subset
        will be stable over time (as long as this source PCD is only accumulating
        new RRTs over time and not losing the RRTs that it starts with).

        """
        chem_db = self.chemical
        chem_elem_types = chem_db.element_types
        chem_atom_types = chem_db.atom_types

        base_rts = {x.name: x for x in chem_db.residues if x.name == x.base_name}
        for name in desired_names:
            if name not in base_rts:
                message = (
                    "ERROR: could not build the requested PachedChemcialDatabase"
                    + f" subset because '{name}' is not present in the original set"
                )
                raise ValueError(message)
        unpatched_residue_subset = [base_rts[name] for name in desired_names]

        desired_variants = sorted(
            [x for x in chem_db.variants if x.display_name in desired_variants],
            key=lambda x: x.name,
        )

        chemical_db_subset = ChemicalDatabase(
            element_types=chem_elem_types,
            atom_types=chem_atom_types,
            residues=unpatched_residue_subset,
            variants=desired_variants,
        )
        patched_chemical_db_subset = PatchedChemicalDatabase.from_chem_db(
            chemical_db_subset
        )

        return ParameterDatabase(
            scoring=self.scoring, chemical=patched_chemical_db_subset
        )


def inject_residue_params(
    param_db: ParameterDatabase,
    residue_types: list[RawResidueType],
    atom_types: Optional[list[AtomType]] = None,
    partial_charges: Optional[Mapping[str, dict[str, float]]] = None,
    cartbonded_params: Optional[Mapping[str, CartRes]] = None,
    variants: Optional[list] = None,
) -> ParameterDatabase:
    """Return a new ParameterDatabase with additional residue type data.

    This is the primary API for extending a database with ligand or custom
    residue types.  The input ``param_db`` is not modified.

    Args:
        param_db: Base database to extend.
        residue_types: New RawResidueType entries to add.
        atom_types: Optional new AtomType entries (deduplicated by name).
        partial_charges: Per-residue charge dicts ``{res_name: {atom: charge}}``.
        cartbonded_params: Per-residue CartRes ``{res_name: CartRes}``.
        variants: Optional patches the new residues bring with them, applied
            alongside the database's own.

    Returns:
        A new frozen ParameterDatabase with the additional data.
    """
    new_atom_types = param_db.chemical.atom_types
    if atom_types:
        existing_names = {at.name for at in new_atom_types}
        deduped = [at for at in atom_types if at.name not in existing_names]
        if deduped:
            new_atom_types = (*new_atom_types, *deduped)

    # patching runs at db load
    # injected residues get all db variants applied here
    new_patched = param_db.chemical.with_added_residues(
        residue_types, atom_types=new_atom_types, variants=variants
    )

    new_elec = param_db.scoring.elec
    if partial_charges:
        new_entries = tuple(
            PartialCharges(res=res, atom=atom, charge=charge)
            for res, charges in partial_charges.items()
            for atom, charge in charges.items()
        )
        new_elec = attr.evolve(
            new_elec,
            atom_charge_parameters=(*new_elec.atom_charge_parameters, *new_entries),
        )

    new_cart = param_db.scoring.cartbonded
    if cartbonded_params:
        new_res_params = {**new_cart.residue_params, **cartbonded_params}
        new_cart = attr.evolve(new_cart, residue_params=new_res_params)

    new_scoring = attr.evolve(
        param_db.scoring,
        elec=new_elec,
        cartbonded=new_cart,
    )

    return attr.evolve(param_db, scoring=new_scoring, chemical=new_patched)


__all__ = [
    "AtomType",
    "ChemicalDatabase",
    "ParameterDatabase",
    "PatchedChemicalDatabase",
    "RawResidueType",
    "ScoringDatabase",
    "inject_residue_params",
]
