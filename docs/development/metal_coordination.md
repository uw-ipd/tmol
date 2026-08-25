# Programmatic metal coordination

## Scope

TMol recognizes explicit inter-component bonds to single-ion Mg, Ca, and Zn
components by element, not by residue name. The deposited component and atom
names are preserved while the importer generates an ion residue type, up to
eight deterministic connections, and matching connection variants for donor
residues. Coordinating waters are retained; unrelated crystallographic waters
remain filtered.

The implementation intentionally requires an explicit PDB/mmCIF bond table.
It does not infer coordination from distance, so an absent or ambiguous bond
cannot silently alter chemistry. Multi-atom cofactors such as heme and metals
outside Mg/Ca/Zn are not yet supported.

## Rosetta parity

The generated ion parameters match Rosetta `fa_standard`:

| element | atom type | charge | LJ radius | LJ well depth | LK dgfree | LK lambda | LK volume |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mg | `Mg2p` | +2 | 1.185 | 0.015 | -5.0 | 3.5 | 7.0 |
| Ca | `Ca2p` | +2 | 1.367 | 0.120 | 0.0 | 2.0 | 10.7 |
| Zn | `Zn2p` | +2 | 1.090 | 0.250 | -5.0 | 3.5 | 5.4 |

Like Rosetta's `SetupMetalsMover`, import adds chemical connections so
ordinary nonbonded count-pair logic does not treat each donor as a clash. It
also adds deposited-geometry constraints with Rosetta's functional forms:

- virtual-to-donor, metal-to-virtual, and virtual-to-virtual harmonic distance
  constraints with 0.1 Å standard deviation;
- metal-donor-parent harmonic angle constraints with 0.05 radian standard
  deviation wherever the donor has a distinct local parent atom (a
  single-atom deposited water does not); and
- distance and angle strength multipliers, both 1.0 by default.

The constraints are zero at the deposited geometry. As in Rosetta, they only
contribute when the constraint score term has nonzero weight. Set
`ScoreType.constraint` to 1.0 for scoring or Cartesian minimization.

The checked-in regressions use the four-coordinate Zn site from PDB 1CA2 and a
seven-coordinate Ca site from PDB 1CLL. They verify topology, virtual proxy
placement, Rosetta parameter values, zero deposited constraint energy,
positive differentiable energy after a donor perturbation, and finite
beta2016 scoring on CPU and CUDA.

## Deliberate boundary

This layer matches Rosetta's ion nonbonded parameters, explicit connectivity,
and deposited-geometry constraint equations. It does not claim an inferred
metal-binding residue catalog, charge-state mutation for multiply coordinated
donors, or specialized quantum/ligand-field energetics. Those require an
explicit chemistry policy and should not be hidden in generic import.

## Rosetta references

- [SetupMetalsMover documentation](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/SetupMetalsMover)
- [Rosetta metal setup and scoring documentation](https://docs.rosettacommons.org/docs/latest/rosetta_basics/non_protein_residues/Metals)
- [SetupMetalsMover implementation helpers](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/util/metalloproteins_util.cc)
