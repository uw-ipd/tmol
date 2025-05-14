# Tmol

`tmol`, short for TensorMol, is a faithful reimplementation of the Rosetta molecular modeling energy function ("beta_nov2016_cart") in PyTorch with custom kernels written in C++ and CUDA. Given the coordinates of one or more proteins, `tmol` can compute both energies and derivatives. `tmol` can also perform gradient-based minimization on those structures. Thus, ML models that produce cartesian coordinates for proteins can include biophysical features in their loss during training or refine their output structures using Rosetta's experimentally validated energy function. You can read the full wiki [here](https://github.com/uw-ipd/tmol/wiki/DevHome).

## Usage

`tmol` can be used as a [standalone](https://github.com/uw-ipd/tmol/wiki/DevHome#Standalone), or as a library for [RosettaFold2](https://github.com/uw-ipd/tmol/wiki/DevHome#RosettaFold2) or [OpenFold](https://github.com/uw-ipd/tmol/wiki/DevHome#OpenFold). 

Each platform has functions for constructing a [PoseStack](https://github.com/uw-ipd/tmol/wiki/PoseStack), performing operations on that PoseStack, and retreiving the structure back from `tmol`.

#### Create a PoseStack from a PDB file
```
import tmol
tmol.pose_stack_from_pdb('1ubq.pdb')
```

#### Create a ScoreFunction and score a PoseStack
```
    sfxn = tmol.beta2016_score_function(pose_stack.device)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

    print(scorer(pose_stack.coords))
```

#### Create a Minimizer and run it on a PoseStack with our ScoreFunction
```
    start_coords = pose_stack.coords.clone()
    pose_stack.coords[:] = start_coords

    cart_sfxn_network = tmol.cart_sfxn_network(sfxn, pose_stack)
    optimizer = tmol.lbfgs_armijo(cart_sfxn_network.parameters())

    cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)


    def closure():
        optimizer.zero_grad()
        E = cart_sfxn_network().sum()
        E.backward()
        return E

    optimizer.step(closure)

    cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)
```

#### Save a PoseStack to a PDB
```
    tmol.write_pose_stack_pdb(pose_stack, 'output.pdb')
```

## Standalone

To setup `tmol`, run:
```
./dev_setup
```

To start using `tmol`, enable the conda environment with:

```
conda activate tmol
```

## RosettaFold2

To use `tmol` within RosettaFold2, first install `tmol` into the RF2 conda environment:

```
    # Activate your RF2 conda environment
    conda install cuda -c nvidia
    cd <your local tmol repository root directory>
    pip install -e .
```
>[!NOTE]
>This has been tested on Ubuntu 20.04 - other platforms should work but are currently untested.

Example usage from within RosettaFold2:

#### Create a PoseStack from RF2 coordinates
```
    seq, xyz, chainlens = rosettafold2_model.infer(sequence)

    pose_stack = tmol.pose_stack_from_rosettafold2(seq[0], xyz[0], chainlens[0])
```

#### Load a PoseStack into RF2 coordinates
```
    xyz = tmol.pose_stack_to_rosettafold2( ... )
```

>[!NOTE]
>Hydrogens and OXT coordinates from the terminal residues in RosettaFold are not preserved across the RF2<->tmol interface.



>[!WARNING]
>You must call `torch.set_grad_enabled(True)` if you wish to use the `tmol` minimizer, as by default RF2 has grad disabled during inference. 


## OpenFold

Full Openfold documentation coming soon.

#### Create a PoseStack from an OpenFold dictionary
```
    output = openfold_model.infer(sequences)
    pose_stack = tmol.pose_stack_from_openfold(output)
```

## Development Workflow

`tmol` uses Test-Driven Development. If you are writing `tmol` code, [you should start by writing test cases for your code](https://github.com/uw-ipd/tmol/wiki/Testing#writing-tests).

### Committing
tmol uses pre-commit hooks to ensure uniform code formatting. These pre-commit hooks run `clang-format` and `black`. If your code needed reformatting, the initial commit will fail, and clang/black will reformat your code. You can see these changes via `git diff`, and you can `git add` the files to accept the new formatting before committing again.

### Submitting changes to master
All changes to master should be performed via pull request flow, with a PR serving as a core point of development, discussion, testing and review. We close pull requests via squash or rebase, so that master contains a tidy, linear project history.

A pull request should land as an "atomic" unit of work, representing a single set of related changes. A larger feature may span multiple pull requests, however each pull request should stand alone. If a request appears to be growing "too large" to review, utilize a stacked pull to partition the work.

### Automated Testing
We maintain an automated test suite executed via buildkite. The test suite must always be passing for master, and is available for any open branch via pull request. By default, the test suit will run on any PR

## Citation
If you use `tmol` in your work, please cite:

Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg, David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio, _tmol: a GPU-accelarated, PyTorch implementation of Rosetta’s relax protocol_, (manuscript in preparation)




