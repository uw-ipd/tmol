import pyrosetta

if pyrosetta.rosetta.basic.was_init_called():
    assert (
        "-beta" in pyrosetta.rosetta.basic.options.initialize().get_argv()
    ), "pyrosetta initialized with non-'-beta' configuration."
else:
    pyrosetta.distributed.maybe_init(options="-beta")
