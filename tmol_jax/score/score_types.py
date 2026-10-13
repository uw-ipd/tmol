"""Named energy channels supported by the experimental JAX frontend."""

from enum import IntEnum


class ScoreType(IntEnum):
    """All per-term energy channels that :class:`~tmol_jax.score.ScoreFunction`
    can weight and sum.

    One :class:`EnergyTerm` may provide several score types (for instance, the
    LJLK term provides ``fa_ljatr``, ``fa_ljrep``, and ``fa_lk``). The score
    function owns the weights; the terms own the unweighted evaluations.
    """

    # LJ / LK pair terms (LJLKEnergyTerm)
    fa_ljatr = 0
    fa_ljrep = 1
    fa_lk = 2

    # Electrostatics (ElecEnergyTerm)
    fa_elec = 3

    # Hydrogen bonding (HBondEnergyTerm)
    hbond = 4

    # Cartesian bonded (CartBondedEnergyTerm)
    cart_lengths = 5
    cart_angles = 6
    cart_torsions = 7
    cart_impropers = 8
    cart_hxltorsions = 9

    # Constraints (ConstraintEnergyTerm)
    constraint = 10

    # Disulfide bonds (DisulfideEnergyTerm)
    disulfide = 11

    # Backbone torsions (BackboneTorsionEnergyTerm)
    omega = 12
    rama = 13

    # Dunbrack rotamer library (DunbrackEnergyTerm)
    dunbrack_rot = 14
    dunbrack_rotdev = 15
    dunbrack_semirot = 16

    # Anisotropic lk (LKBallEnergyTerm)
    lk_ball_iso = 17
    lk_ball = 18
    lk_bridge = 19
    lk_bridge_uncpl = 20

    # Per-residue reference energies (RefEnergyTerm)
    ref = 21

    # Generic bonded torsions + impropers (GenBondedEnergyTerm)
    gen_torsions = 22

    n_score_types = 23


__all__ = ["ScoreType"]
