# MPS port scratch tools

Throwaway harnesses used while porting tmol to Apple Metal. Not part of the
build or the test suite; kept because reproducing them is tedious.

See `MPS_PORT_NOTES.md` at the repo root for the full context.

| file | purpose |
|---|---|
| `metal_probe.mm` | proves runtime MSL compilation + dispatch works without the offline Metal toolchain |
| `msl_probe.mm` | compiles a pre-expanded `.metal` source through `newLibraryWithSource`; prints errors |
| `mslstub/` | empty `metal_stdlib` etc. so `clang -E` can pre-expand tmol headers |
| `phase0*.py` | the profiling scripts behind the numbers in the notes |
| `elec_bench.py` | isolated elec fwd+bwd timing, median of 9 |

Build the probes:

```bash
clang++ -std=c++17 -ObjC++ -fobjc-arc -framework Foundation -framework Metal \
  -o msl_probe msl_probe.mm
```

Compile a tmol header as MSL (runtime compilation cannot resolve filesystem
includes, so pre-expand first):

```bash
clang++ -E -x c++ -std=c++17 -D__METAL_VERSION__=320 \
  -Idev/mps_scratch/mslstub -I. -Itmol/extern probe.metal.in | grep -v '^#' > out.metal
printf '#include <metal_stdlib>\nusing namespace metal;\n' | cat - out.metal > full.metal
./msl_probe full.metal 32
```

**Pass the `32`.** It pins MSL 3.2, which is what torch's shader API asks for.
Without it `msl_probe` uses the newest language version, which infers `thread`
where tmol spells it out, and reports a wall of bogus address-space
mismatches -- for the already-working shaders as much as a new one. A clean
probe at the default version proves nothing.
