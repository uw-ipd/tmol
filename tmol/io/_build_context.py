from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol.pose import PackedBlockTypes

if TYPE_CHECKING:
    from tmol.ligand import LigandFragmentDefinition
    from tmol.pack.rotamer import NaChiRotamerSampler
    from tmol.pack.rotamer.dunbrack import DunbrackChiSampler
    from tmol.score import ScoreFunction


@dataclass(frozen=True)
class PoseBuildContext:
    """Immutable, structure-independent construction context.

    Holds only the pieces that depend on the parameter database / ligand set
    (not on any particular input), so it can be built once and reused across
    many inputs that share the same ligand(s).
    """

    canonical_ordering: CanonicalOrdering
    packed_block_types: PackedBlockTypes
    parameter_database: ParameterDatabase
    restype_set: ResidueTypeSet
    # Definitions derived from tmol_fragment_id annotations. These are carried
    # by reusable contexts so each compatible structure can be expanded without
    # repeating ligand preparation.
    fragment_definitions: tuple[LigandFragmentDefinition, ...] = ()
    # SMILES string -> residue type name, for ligands prepared from a sequence.
    ligand_names: dict[str, str] = field(default_factory=dict)

    @cached_property
    def _packing_score_function(self) -> ScoreFunction:
        """Return the score function shared by repeated pose builds."""
        from tmol.score import beta2016_score_function

        return beta2016_score_function(
            self.packed_block_types.device,
            param_db=self.parameter_database,
        )

    @cached_property
    def _opth_score_function(self) -> ScoreFunction:
        """Return beta2016 without terms invariant under OptH sampling."""
        from tmol.score import ScoreType, beta2016_score_function

        score_function = beta2016_score_function(
            self.packed_block_types.device,
            param_db=self.parameter_database,
        )
        # OptH changes proton chis and terminal NHQ groups. These terms depend
        # only on residue identity, disulfide geometry, protein backbone, or
        # nucleic-acid heavy-atom torsions, so they are constant across every
        # OptH candidate and cannot affect the selected assignment.
        for score_type in (
            ScoreType.disulfide,
            ScoreType.omega,
            ScoreType.rama,
            ScoreType.ref,
            ScoreType.na_torsion,
            ScoreType.na_torsion_well,
        ):
            score_function.set_weight(score_type, 0)
        return score_function

    @cached_property
    def _dunbrack_sampler(self) -> DunbrackChiSampler:
        """Return the Dunbrack sampler shared by repeated pose builds."""
        from tmol.pack.rotamer.dunbrack import (
            create_dunbrack_sampler_from_database,
        )

        return create_dunbrack_sampler_from_database(
            self.parameter_database,
            self.packed_block_types.device,
        )

    @cached_property
    def _na_sampler(self) -> NaChiRotamerSampler:
        """Return the nucleic-acid sampler shared by repeated pose builds."""
        from tmol.pack.rotamer import NaChiRotamerSampler

        return NaChiRotamerSampler.from_database(
            self.parameter_database,
            self.packed_block_types.device,
        )
