from dataclasses import dataclass, field
from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol.pose import PackedBlockTypes


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
    fragment_definitions: tuple = ()
    # SMILES string -> residue type name, for ligands prepared from a sequence.
    ligand_names: dict = field(default_factory=dict)
    # (base block-type name, attachment atoms) -> connection-capable clone.
    covalent_variant_names: dict = field(default_factory=dict)
    # (base block-type name, coordination signature) -> metal-capable clone.
    metal_variant_names: dict = field(default_factory=dict)
