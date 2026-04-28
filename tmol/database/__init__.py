import os
import attr
from typing import List, Optional, Mapping

from .chemical import AtomType, ChemicalDatabase, RawResidueType
from .scoring import ScoringDatabase
from .scoring.elec import ElecDatabase, PartialCharges
from .scoring.cartbonded import CartBondedDatabase, CartRes

from tmol.chemical.patched_chemdb import PatchedChemicalDatabase


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
        return cls(
            scoring=ScoringDatabase.from_file(os.path.join(path, "scoring")),
            chemical=patched_chemdb,
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

    Returns:
        A new frozen ParameterDatabase with the additional data.
    """
    new_chem_residues = (*param_db.chemical.residues, *residue_types)

    new_atom_types = param_db.chemical.atom_types
    if atom_types:
        existing_names = {at.name for at in new_atom_types}
        deduped = [at for at in atom_types if at.name not in existing_names]
        if deduped:
            new_atom_types = (*new_atom_types, *deduped)

    new_patched = attr.evolve(
        param_db.chemical,
        residues=new_chem_residues,
        atom_types=new_atom_types,
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
