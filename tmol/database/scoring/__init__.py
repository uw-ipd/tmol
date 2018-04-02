import os
import yaml
import cerberus
import properties
import pprint

import numpy
import pandas

from tmol.properties.reactive import derived_from

class LJLKDatabase(properties.HasProperties):
    parameter_schema = {
      "global_parameters" : {
        "type" : "dict",
        "schema" : {
          'max_dis':               {'type': 'float', 'required' : True},
          'spline_start':          {'type': 'float', 'required' : True},
          'lj_hbond_OH_donor_dis': {'type': 'float', 'required' : True},
          'lj_hbond_dis':          {'type': 'float', 'required' : True},
          'lj_hbond_hdis':         {'type': 'float', 'required' : True},
          'lj_switch_dis2sigma':   {'type': 'float', 'required' : True},
          'lk_min_dis2sigma':      {'type': 'float', 'required' : True},
        }
      },
      "atom_type_parameters" : {
        "type" : "list",
        "schema" : {
          "type" : "dict",
          "schema" : {
            'name':      {'type': 'string', 'required' : True},
            'elem':      {'type': 'string', 'required' : True},
            'lj_radius': {'type': 'float',  'required' : True},
            'lj_wdepth': {'type': 'float',  'required' : True},
            'lk_dgfree': {'type': 'float',  'required' : True},
            'lk_lambda': {'type': 'float',  'required' : True},
            'lk_volume': {'type': 'float',  'required' : True},
            'is_acceptor':  {'type': 'boolean', 'default' : False},
            'is_donor':     {'type': 'boolean', 'default' : False},
            'is_hydroxyl':  {'type': 'boolean', 'default' : False},
            'is_polarh':    {'type': 'boolean', 'default' : False},
          }
        }
      }
    }

    parameters = properties.Dictionary("ljlk score term parameters")

    atom_type_table : pandas.DataFrame = properties.Instance(
        "Dataframe of per-atom-type parameters, indexed by atom type name.",
        pandas.DataFrame
    )

    @derived_from("atom_type_table",
        properties.Instance("atom type name to atom type index mapping", pandas.Series))
    def name_to_idx(self):
        return pandas.Series(
            data = numpy.arange(len(self.atom_type_table.index)),
            index = self.atom_type_table.index
        )

    @classmethod
    def load(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        validator = cerberus.Validator(cls.parameter_schema)
        parameters = validator.validated(raw)
        if validator.errors:
            raise ValueError(
                f"Error loading database source: {path}:\n" +
                pprint.pformat(validator.errors, indent=2)
            )

        atom_type_table = (
            pandas.DataFrame.from_records(parameters["atom_type_parameters"])
            .set_index("name", verify_integrity=True)
        )

        return cls(
            parameters = parameters,
            atom_type_table = atom_type_table
        )

class ScoringDatabase(properties.HasProperties):

    ljlk : LJLKDatabase = properties.Instance("lj/lk score term parameters", LJLKDatabase)

    @classmethod
    def load(cls, path=os.path.dirname(__file__)):
        ljlk_path = os.path.join(path, "ljlk.yaml")
        ljlk = LJLKDatabase.load(ljlk_path)

        return cls(ljlk = ljlk)
