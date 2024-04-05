import os
import attr
from typing import List

from .chemical import ChemicalDatabase
from .scoring import ScoringDatabase

# maybe this should live in the database?
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase


@attr.s
class ParameterDatabase:
    """The parameters describing the available chemical types and the scoring terms"""

    __default = None

    @classmethod
    def get_default(cls) -> "ParameterDatabase":
        """Load and return default parameter database."""
        if cls.__default is None:
            cls.__default = ParameterDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "default")
            )
        return cls.__default

    scoring: ScoringDatabase = attr.ib()
    chemical: PatchedChemicalDatabase = attr.ib()

    @classmethod
    def from_file(cls, path):
        chemdb = ChemicalDatabase.from_file(os.path.join(path, "chemical"))
        patched_chemdb = PatchedChemicalDatabase.from_chem_db(chemdb)  # apply patches
        return cls(
            scoring=ScoringDatabase.from_file(os.path.join(path, "scoring")),
            chemical=patched_chemdb,
        )

    def create_stable_subset(
        self, desired_names: List[str], desired_variants: List[str]
    ):
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
