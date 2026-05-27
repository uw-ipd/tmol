import os
import attr
from typing import Optional

from .chemical import AtomType, ChemicalDatabase, RawResidueType
from .scoring import ScoringDatabase
from .scoring.elec import PartialCharges
from .scoring.cartbonded import CartRes

# maybe this should live in the database?
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase


@attr.s
class ParameterDatabase:
    """Chemical and scoring parameter container used by tmol.

    This class owns a patched chemical database and the scoring databases
    used to build score functions. The process-global accessor
    `get_current()` returns a shared mutable instance. Use
    `get_fresh_default()` when caller isolation is required.
    """

    __default = None

    @classmethod
    def get_current(cls) -> "ParameterDatabase":
        """Return the process-global cached parameter database."""
        if cls.__default is None:
            cls.__default = ParameterDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "default")
            )
        return cls.__default

    @classmethod
    def get_fresh_default(cls) -> "ParameterDatabase":
        """Load a new default parameter database instance from disk."""
        return ParameterDatabase.from_file(
            os.path.join(os.path.dirname(__file__), "default")
        )

    scoring: ScoringDatabase = attr.ib()
    chemical: PatchedChemicalDatabase = attr.ib()

    def add_residue_scoring_params(
        self,
        res_name: str,
        partial_charges: Optional[dict[str, float]] = None,
        cartbonded_params: Optional[CartRes] = None,
    ) -> None:
        """Add temporary scoring parameters for a residue type.

        Routes to the appropriate sub-databases internally.
        For permanent additions, edit the YAML files in tmol/database/default/scoring/.

        Args:
            res_name: Residue name (e.g. "I4B", "ATP").
            partial_charges: Per-atom partial charges {atom_name: charge}.
            cartbonded_params: CartRes with bond lengths, angles, and impropers.
        """
        if partial_charges is not None:
            new_entries = tuple(
                PartialCharges(res=res_name, atom=atom, charge=charge)
                for atom, charge in partial_charges.items()
            )
            self.scoring.elec.atom_charge_parameters = (
                self.scoring.elec.atom_charge_parameters + new_entries
            )
        if cartbonded_params is not None:
            self.scoring.cartbonded.residue_params[res_name] = cartbonded_params

    def remove_residue_scoring_params(self, res_name: str):
        """Remove all temporary scoring parameters for a residue type.

        Args:
            res_name: Residue name to remove.
        """
        self.scoring.elec.atom_charge_parameters = tuple(
            p for p in self.scoring.elec.atom_charge_parameters if p.res != res_name
        )
        self.scoring.cartbonded.residue_params.pop(res_name, None)

    def add_residue_type(
        self,
        residue_type: RawResidueType,
        new_atom_types: Optional[list[AtomType]] = None,
    ) -> None:
        """Add a residue type to the chemical database.

        Args:
            residue_type: The RawResidueType to add.
            new_atom_types: Optional list of new AtomType entries to register.

        Returns:
            None.
        """
        if new_atom_types:
            self.chemical.atom_types = (*self.chemical.atom_types, *new_atom_types)
        self.chemical.residues = (*self.chemical.residues, residue_type)

    def remove_residue_type(self, res_name: str) -> None:
        """Remove a residue type from the chemical database.

        Args:
            res_name: Residue name to remove.
        """
        self.chemical.residues = [
            r for r in self.chemical.residues if r.name != res_name
        ]

    @classmethod
    def from_file(cls, path: str) -> "ParameterDatabase":
        chemdb = ChemicalDatabase.from_file(os.path.join(path, "chemical"))
        patched_chemdb = PatchedChemicalDatabase.from_chem_db(chemdb)  # apply patches
        return cls(
            scoring=ScoringDatabase.from_file(os.path.join(path, "scoring")),
            chemical=patched_chemdb,
        )

    def create_stable_subset(
        self, desired_names: list[str], desired_variants: list[str]
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

        # TO DO: Decide whether these should be shared between
        # the self PCD and the newly created PCD
        # Should we share _all_ of the residue objects or just
        # the unpatched ones or none?

        # for stability, the order of the RRTs in the newly constructed
        # PCD should be independent of the order that those RRTs
        # appear in this PCD. Therefore, iterate across the desired names
        # and find the corresponding residue type for each

        # for speed, collected the unpatched subset of RRTs and create a named-based
        # lookup
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
