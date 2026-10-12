# External Support Components

This module provides support components that are *not* runtime
dependencies. This includes, for example, single-pass parameter import,
database generation and analysis components for baseline test generation.

## Database generation tools

Run these modules from a development checkout. Read their module docstrings and
arguments first: they write scientific database assets, and some require
external Rosetta installations or structure inputs. Their absence from runtime
imports is intentional.

| Modules | Maintained assets |
|---|---|
| `chemical._add_d_amino_acids` | Mirrored D-amino-acid residue definitions |
| `chemical._add_metal_ions` | Chemical metal-ion and electrostatic parameter YAML |
| `scoring._add_symmetric_gly_tables` | Mirror-symmetric glycine backbone tables |
| `scoring._add_aib_tables` | AIB backbone tables and lookup rows |
| `scoring._add_equiv_class_lookup_rows` | Backbone-library residue equivalence mappings |
| `scoring._rewrite_dunbrack_binary`, `scoring._rewrite_rama_binary`, `scoring._rewrite_omega_bbdep_binary` | Converted rotamer and backbone parameter libraries |
| `scoring._na_torsion_param_import` | Nucleic-acid torsion parameters from structure inputs |
| `scoring._hbond_param_import` | Hydrogen-bond parameter import helpers |

For example, the AIB generator is invoked as
`python -m tmol.support.scoring._add_aib_tables`. Keep generator code, source
provenance, and generated parameter data together when changing a database.
