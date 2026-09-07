# Figure captions

## Multimodal workload speed

Throughput versus polymer residues per structure for protein, protein–ligand,
and protein–DNA/RNA inputs (columns) and whole-pose scoring with Cartesian
gradients and one-repeat FastRelax (rows). PyRosetta and TMol CPU measurements
use batch size one and one software thread. TMol GPU measurements use batches
1, 10, 100, and 1,000 on one NVIDIA H200. Solid lines show TMol 0.1.55;
dashed lines with open markers show TMol 0.1.46, except protein–DNA/RNA, which
uses the first nucleic-capable historical release, 0.1.47. PyRosetta is the
shared black CPU reference. Score-gradient points are medians of seven
synchronized intervals after five warmups; FastRelax points are reproducibly
seeded complete native workflows. Higher is better. Missing high-size or
high-batch points have explicit failure records in the accompanying tables.
Each panel has independent logarithmic x and y limits because the modalities
and workflows span substantially different residue and throughput ranges.

## Multimodal workload memory

Peak working memory versus polymer residues for the same columns, rows, and
series as the speed figure. CPU values are process peak resident-set size from
a fresh process for each point. GPU values are peak bytes allocated by
PyTorch's CUDA allocator after resetting its peak counter and include pose,
scorer, and workflow setup. These measurements describe the relevant working
set within each device domain; CPU RSS and CUDA allocation are not intended as
byte-for-byte allocator comparisons. Missing points denote recorded loading,
memory, indexing, or timeout failures. Each panel uses independent logarithmic
x and y limits; an unfinished partial render labels panels with no completed
measurements as pending.
