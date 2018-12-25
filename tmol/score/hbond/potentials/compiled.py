from pathlib import Path
from tmol.utility.cpp_extension import load

_compiled_sources = [str(Path(__file__).parent / s) for s in ("_compiled.cpp",)]
_compiled = load("_compiled", _compiled_sources)

hbond_donor_sp2_score = _compiled.hbond_donor_sp2_score
hbond_donor_sp3_score = _compiled.hbond_donor_sp3_score
