import json
import os

from .chemical import ChemicalDatabase

basedir = os.path.dirname(__file__)

basic = ChemicalDatabase(source = f"{basedir}/basic")
