"""User-facing sampler bundles for ordinary side-chain packing."""

import attr

from tmol.pack.rotamer._conformer_sampler import ConformerSampler
from tmol.pack.rotamer._conjugated_groups import add_conjugated_group_sampler
from tmol.pack.rotamer._fixed_aa_chi_sampler import FixedAAChiSampler


@attr.s(auto_attribs=True, frozen=True)
class SidechainSampler:
    """Every sampler needed to move an ordinary protein side chain.

    Declaring sampling should not require knowing which sampler covers which
    residue. A rotamer library handles the residues it has entries for and a
    fixed-chi sampler handles the rest; callers who just want "sample side
    chains reasonably" should not have to compose that themselves.

    Examples:
      >>> task.add_conformer_sampler(
      ...     SidechainSampler.from_database(database, pose_stack.device)
      ... )
    """

    samplers: tuple[ConformerSampler, ...]

    @classmethod
    def from_database(cls, database, device) -> "SidechainSampler":
        """Bundle the rotamer-library and fixed-chi samplers for ``database``."""
        from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

        return cls(
            (
                create_dunbrack_sampler_from_database(database, device),
                FixedAAChiSampler(),
            )
        )

    def expand_into_task(self, task) -> None:
        """Register each bundled sampler on ``task``."""
        for sampler in self.samplers:
            task.add_conformer_sampler(sampler)


@attr.s(auto_attribs=True, frozen=True)
class ConjugatedSCSampler:
    """Sample side chains while keeping covalently linked groups together.

    A residue with something bonded to its side chain has to be sampled as one
    unit; moving the two apart would break the bond. This wraps a side-chain
    bundle so that grouping is applied without the caller repeating the
    exclusion bookkeeping.

    Examples:
      >>> task.add_conformer_sampler(
      ...     ConjugatedSCSampler(
      ...         SidechainSampler.from_database(database, pose_stack.device)
      ...     )
      ... )
    """

    sidechain: SidechainSampler

    def expand_into_task(self, task) -> None:
        """Register the side-chain bundle, then group conjugated residues."""
        self.sidechain.expand_into_task(task)
        # Runs after the bundle so the group sampler can find the library.
        add_conjugated_group_sampler(task, task.pose_stack)
