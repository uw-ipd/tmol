import os
import yaml
import cerberus
import properties
import pprint


class ChemicalDatabase(properties.HasProperties):
    parameter_schema = {
      "atom_types" : {
        "type" : "list",
      },
      "residues" : {
        "type" : "list",
        "schema" : {
          "type" : "dict",
          "schema" : {
            'name' : {'type': 'string', 'required' : True},
            'name3': {'type': 'string' },
            'atoms': {'type': 'list', 'required' : True, "schema" : { "type" : "dict", "schema": {
                'name' : {'type': 'string', 'required' : True},
                'atom_type' : {'type': 'string', 'required' : True},
            }}},
            'bonds': {'type': 'list', 'required' : True, "schema" : {
                "type" : "list", "minlength" : 2, "maxlength" : 2,
            }},
            'lower_connect' : {'type': 'string', 'required' : True},
            'upper_connect' : {'type': 'string', 'required' : True},
            'hbond': {"type" : "dict", "required" : False, "schema": {
              'donors': {'type': 'list', "schema" : { "type" : "dict", "schema": {
                  'd' : {'type': 'string', 'required' : True},
                  'h' : {'type': 'string', 'required' : True},
              }}},
              'acceptors': {"type" : "dict", "schema": {
                  "ring" : {'type': 'list', "schema" : { "type" : "dict", "schema": {
                      'a' : {'type': 'string', 'required' : True},
                      'b' : {'type': 'string', 'required' : True}
                  }}},
                  "sp2" : {'type': 'list', "schema" : { "type" : "dict", "schema": {
                      'a' : {'type': 'string', 'required' : True},
                      'b' : {'type': 'string', 'required' : True},
                      'b0' : {'type': 'string', 'required' : True},
                  }}},
                  "sp3" : {'type': 'list', "schema" : { "type" : "dict", "schema": {
                      'a' : {'type': 'string', 'required' : True},
                      'b' : {'type': 'string', 'required' : True},
                      'b0a' : {'type': 'string', 'required' : True},
                      'b0b' : {'type': 'string', 'required' : True},
                  }}},
              }},
            }},
          }
        }
      }
    }

    parameters = properties.Dictionary("chemical datatypes")

    @classmethod
    def load(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        validator = cerberus.Validator(cls.parameter_schema)
        parameters = validator.validated(raw)
        if validator.errors:
            raise ValueError(
                f"Error loading database source: {path}:\n" +
                pprint.pformat(validator.errors, indent=2)
            )

        return cls(
            parameters = parameters,
        )
