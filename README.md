# neurotopo

Topological Sholl descriptors for neuron morphology. Give it a folder of SWC
reconstructions; it gives you seven descriptor distance matrices and a
combination of them.

```
SWC files -> step functions -> per-descriptor L1 distance matrices
          -> weighted combination -> clusters
```

Method: Khalil R, Kallel S, Farhat A, Dlotko P. Topological Sholl descriptors
for neuronal clustering and classification. *PLoS Comput Biol.*
2022 Jun 22;18(6):e1010229.

## Installation

Requires Python 3.9+.

```bash
pip install neurotopo
```

If it isn't on PyPI yet, install straight from GitHub:

```bash
pip install git+https://github.com/reemkhalilneurolab/neurotopo.git
```

To work on the code itself, clone it and install in editable mode:

```bash
git clone https://github.com/reemkhalilneurolab/neurotopo.git
cd neurotopo
pip install -e ".[test]"
```

## Quickstart

Point it at a folder of `.swc` files:

```
my_neurons/
    control/
        cell01.swc
        cell02.swc
    treated/
        cell03.swc
        cell04.swc
```

```python
import neurotopo as nt

neurons, classes = nt.load_swc_directory("my_neurons")
step_functions = nt.compute_descriptors(neurons)
distances = nt.compute_distances(step_functions)
combined = nt.combine(distances)
clusters = nt.cluster(combined, k=2)
```

or in one call:

```python
result = nt.run("my_neurons", k=2)
result["clusters"]      # cluster assignment per neuron
result["combined"]      # the combined distance matrix
result["descriptors"]   # step functions, one dict per descriptor
```

Each immediate subfolder of the input directory is one class, and its name
becomes the class label — used only for the weighted combination and for
comparing a clustering against known groups, never for computing the
descriptors themselves. A folder with `.swc` files directly inside it, no
subfolders, is read as a single unlabelled class.

## What it computes

Seven descriptors, each a step function on `[0, 1]` against a neuron's radial
distance from the soma, normalised by that neuron's own maximum extent:

| Descriptor | What it measures |
|---|---|
| Tortuosity | path length ÷ Euclidean distance, node to its nearest branch point |
| Branching Pattern | cumulative bifurcation and termination counts by distance |
| Wiring | cumulative path length within each radius |
| Flux | alignment of dendrite segments with the radial direction |
| Leaf | fraction of terminal points reachable from a node |
| Spread | convex-hull volume ÷ path length, by radius |
| Energy | field intensity at the soma, treating each node as a charge |

Because every descriptor has the same form — a step function on `[0, 1]` — one
L1 distance applies to all of them (`neurotopo.calculate_l1_difference`).

## Combining the descriptors

Two ways to turn seven distance matrices into one:

```python
combined = nt.combine(distances)                       # equal weights (default)
combined = nt.combine(distances, weights={"Spread": 2.0, "Leaf": 0.5, ...})

best_weights, score, combined = nt.grid_search(distances, classes)
```

`combine()` rescales each matrix to unit mean before summing, because the
descriptors carry different physical units — Spread is an area, Leaf a
fraction bounded by 1, Branching a count — and an unrescaled sum is dominated
by whichever descriptor has the largest units rather than by what any of them
measure. Pass `rescale=False` to combine raw matrices, but this is rarely what
you want.

`grid_search()` is an exhaustive search over weights on the simplex,
maximising the ratio of between-class to within-class distance (supplementary
section 4.2 of the reference above). It can drive a descriptor's weight to
exactly zero; `combine()` cannot. It costs more: the number of candidate
weight vectors is `C(resolution + k - 1, k - 1)` for `k` descriptors, so with
all seven, `resolution=20` is about 230,000 candidates and `resolution=40` is
about 9.4 million.

`nt.run()` accepts either as `method`:

```python
result = nt.run("my_neurons", method="combine")                    # default
result = nt.run("my_neurons", method="gridsearch", resolution=20)
result["weights"], result["score"]   # only present for method="gridsearch"
```

## Method settings

Settings that change the numbers live in `neurotopo.config` and take effect on
the next call:

```python
import neurotopo as nt

nt.config.REMOVE_TYPES = ["axon"]              # SWC types stripped before any descriptor
nt.config.NORMALIZE_DESCRIPTOR_RADII = True    # normalise radii per neuron
nt.config.DISTANCE_METRIC = "l1"
nt.config.EXCLUDED_NEURONS = ["bad_reconstruction_03"]
```

See the docstring in `neurotopo/config.py` for the full list and what each one
means.

## Plotting and saving

`neurotopo.calculate_spread_descriptor`, `calculate_energy`, and the rest are
plain functions — call them directly if you want a single descriptor for one
neuron. Plotting helpers inside `neurotopo.descriptors` write PNGs to
`neurotopo.config.PLOTS_DIRECTORY` (a `neurotopo_output/` folder in the current
directory by default); nothing in the package writes to disk unless you call
one of those directly, or set `config.PLOT_STEP_FUNCTION = True` before
computing descriptors.

## Development

```bash
pip install -e ".[test]"
pytest
```

The test suite runs the whole pipeline — load, descriptors, distances,
combine, cluster, grid search — on tiny synthetic reconstructions generated on
the fly, so it needs no external data and no network access.

## Citation

If you use this package, please cite:

> Khalil R, Kallel S, Farhat A, Dlotko P. Topological Sholl descriptors for
> neuronal clustering and classification. *PLoS Comput Biol.*
> 2022 Jun 22;18(6):e1010229.

## Licence

MIT. See `LICENSE`.
