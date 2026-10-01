"""
High-level, side-effect-free entry points: SWC files in, distance matrices and
a combined result out.

    import neurotopo as nt

    neurons, classes = nt.load_swc_directory("path/to/swc_root")
    step_functions = nt.compute_descriptors(neurons)
    distances = nt.compute_distances(step_functions)
    combined = nt.combine(distances)
    clusters = nt.cluster(combined, k=2)

or, in one call:

    result = nt.run("path/to/swc_root", k=2)

Nothing here writes to disk. The plotting helpers in neurotopo.descriptors do,
but only when called directly and only into neurotopo.config.PLOTS_DIRECTORY.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from . import config
from .neuron_processor import process_neuron
from . import descriptors as _descriptors
from .distances import compute_pairwise_differences as _compute_pairwise_differences
from .distances import grid_search_optimal_weights as _grid_search_optimal_weights

# Every descriptor is a step function on [0, 1] against normalised radial
# distance from the soma, so a single L1 distance applies to all of them.
DESCRIPTORS = {
    'Tortuosity': _descriptors.calculate_tortuosity,
    'Branching_Pattern': _descriptors.calculate_branching_pattern,
    'Wiring': _descriptors.calculate_wiring_descriptor,
    'Flux': _descriptors.calculate_flux,
    'Leaf': _descriptors.calculate_leaf,
    'Spread': _descriptors.calculate_spread_descriptor,
    'Energy': _descriptors.calculate_energy,
}


def _clean_name(filename: str) -> str:
    stem = re.sub(r'\.swc$', '', filename, flags=re.IGNORECASE)
    stem = stem.replace('.CNG', '')
    return stem.replace(' ', '_').replace('.', '_').replace('-', '_')


def load_swc_directory(path, class_labels: bool = True):
    """
    Read every .swc file under `path`.

    If `path` has subdirectories, each immediate subdirectory is one class,
    its name the class label, and its .swc files are searched for
    recursively. If `path` holds .swc files directly, every neuron gets the
    class label "unlabeled".

    Neurons named in neurotopo.config.EXCLUDED_NEURONS (matched against the
    cleaned filename stem) are skipped.

    Returns
    -------
    neuron_data : dict
        neuron name -> dict, the structure the functions in
        neurotopo.descriptors expect (keys include 'swc_df' and
        'neuron_as_graph').
    classes : pandas.DataFrame
        Indexed by neuron name, with one 'class' column.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f'No such directory: {root}')

    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if subdirs:
        groups = {d.name: sorted(d.rglob('*.swc')) for d in subdirs}
    else:
        groups = {'unlabeled': sorted(root.glob('*.swc'))}

    if sum(len(files) for files in groups.values()) == 0:
        raise FileNotFoundError(f'No .swc files found under {root}')

    neuron_data = {}
    rows = []
    for class_name, files in groups.items():
        for filepath in files:
            name = _clean_name(filepath.name)
            if name in config.EXCLUDED_NEURONS:
                continue

            (graph, bifurcation_ids, termination_ids, marker_ids,
             path_distances, euclidean_distances, swc_df, total_nodes,
             bifurcation_count, termination_count, marker_count,
             primary_branch_count, num_branches, structure_id_counts,
             standardization_status, standardization_description
             ) = process_neuron(str(filepath))

            at_soma_children = swc_df[swc_df['Parent'] == 1]
            bifurcations = swc_df[swc_df['node_type'] == 'bifurcation']
            terminations = swc_df[swc_df['node_type'] == 'termination']
            both = swc_df[swc_df['node_type'].isin(['bifurcation', 'termination'])]

            neuron_data[name] = {
                'class': class_name,
                'swc_df': swc_df,
                'bifurcation_ids': bifurcation_ids,
                'termination_ids': termination_ids,
                'marker_ids': marker_ids,
                'bifurcation_radii': bifurcations['euclidean_distance_to_soma'].tolist(),
                'termination_radii': terminations['euclidean_distance_to_soma'].tolist(),
                'combined_radii': both['euclidean_distance_to_soma'].tolist(),
                'primary_branch_radii': at_soma_children['euclidean_distance_to_soma'].tolist(),
                'bifurcation_radii_normalized': bifurcations['euclidean_distance_to_soma_normalized'].tolist(),
                'termination_radii_normalized': terminations['euclidean_distance_to_soma_normalized'].tolist(),
                'combined_radii_normalized': both['euclidean_distance_to_soma_normalized'].tolist(),
                'primary_branch_radii_normalized': at_soma_children['euclidean_distance_to_soma_normalized'].tolist(),
                'path_distances': path_distances,
                'euclidean_distances': euclidean_distances,
                'neuron_name': name,
                'total_nodes': total_nodes,
                'bifurcation_count': bifurcation_count,
                'termination_count': termination_count,
                'primary_branch_count': primary_branch_count,
                'num_branches': num_branches,
                'structure_id_counts': structure_id_counts,
                'neuron_as_graph': graph,
            }
            rows.append({'neuron_name': name, 'class': class_name})

    classes = pd.DataFrame(rows).set_index('neuron_name')
    return neuron_data, classes


def compute_descriptors(neuron_data: Mapping[str, dict],
                        which: Optional[Iterable[str]] = None) -> Dict[str, dict]:
    """
    Compute step functions for every neuron.

    Returns {descriptor_name: {neuron_name: step_function}}, where a step
    function is a list of (radius, value) tuples with radius in [0, 1].
    `which` restricts to a subset of DESCRIPTORS by name; default is all seven.
    """
    wanted = {name: fn for name, fn in DESCRIPTORS.items()
              if which is None or name in which}
    out = {name: {} for name in wanted}
    for neuron_name, neuron in neuron_data.items():
        for name, fn in wanted.items():
            result = fn(neuron)
            # Spread returns (step_function, volume, path); Energy returns
            # (step_function, vectors). In both cases the first element is
            # the step function.
            out[name][neuron_name] = result[0] if isinstance(result, tuple) else result
    return out


def compute_distances(step_functions: Mapping[str, Mapping[str, list]]) -> Dict[str, pd.DataFrame]:
    """One pairwise L1 distance matrix per descriptor."""
    return _compute_pairwise_differences(dict(step_functions))


def _rescale_to_unit_mean(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.values
    off_diagonal = values[~np.eye(len(values), dtype=bool)]
    mean_distance = off_diagonal.mean()
    return matrix / mean_distance if mean_distance > 0 else matrix


def combine(distance_matrices: Mapping[str, pd.DataFrame],
            weights: Optional[Mapping[str, float]] = None,
            rescale: bool = True) -> pd.DataFrame:
    """
    Combine per-descriptor distance matrices into one.

    The descriptors measure different physical quantities: Spread is an area,
    Leaf a fraction bounded by 1, Branching a count. Their mean pairwise
    distances can differ by orders of magnitude, so an unrescaled sum is
    governed by whichever descriptor carries the largest units rather than by
    what any of them measure. Each matrix is divided by its own mean
    off-diagonal distance before combining, unless `rescale=False`.

    `weights` default to equal. Pass detection rates, or any other per
    descriptor weighting, to weight some descriptors more than others.
    """
    names = list(distance_matrices)
    if weights is None:
        weights = {n: 1.0 for n in names}
    total = None
    for name in names:
        block = distance_matrices[name]
        if rescale:
            block = _rescale_to_unit_mean(block)
        block = block * weights[name]
        total = block if total is None else total + block
    return total


def grid_search(distance_matrices: Mapping[str, pd.DataFrame],
                classes: pd.DataFrame,
                resolution: int = 20):
    """
    Exhaustive search over descriptor weights on the simplex.

    Implements supplementary section 4.2 of the reference method: maximises
    E / sum_k I_k, where I_k is the largest within-class distance in class k
    and E the smallest between-class distance. Unlike `combine`, this can
    drive a descriptor's weight to exactly zero.

    `classes` is the DataFrame returned by `load_swc_directory`: indexed by
    neuron name, with a 'class' column.

    Returns (best_weights, best_score, combined_matrix).
    """
    neuron_class_df = classes.reset_index()
    return _grid_search_optimal_weights(dict(distance_matrices), neuron_class_df,
                                        resolution=resolution)


def cluster(distance_matrix: pd.DataFrame, k: int = 2, method: str = 'ward') -> pd.Series:
    """Flat clusters from a distance matrix, cut at k groups."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    z = linkage(squareform(distance_matrix.values, checks=False), method=method)
    labels = fcluster(z, t=k, criterion='maxclust')
    return pd.Series(labels, index=distance_matrix.index, name='cluster')


def run(swc_directory, k: int = 2, weights: Optional[Mapping[str, float]] = None,
        method: str = 'combine', resolution: int = 20) -> dict:
    """
    Convenience one-call pipeline: an SWC directory in, everything out.

    `method` selects how the seven distance matrices become one:

        'combine'    weighted sum, rescaled to unit mean first (the default).
                     `weights` defaults to equal; pass detection rates or any
                     other per-descriptor weighting to weight some descriptors
                     more than others.
        'gridsearch' exhaustive search over weights on the simplex (see
                     `grid_search`). Slower, and can drop a descriptor
                     entirely, which 'combine' cannot. `resolution` controls
                     its cost; `weights` is ignored for this method.

    Returns a dict with keys 'neurons', 'classes', 'descriptors', 'distances',
    'combined', 'clusters', plus 'weights' and 'score' when method='gridsearch'.
    """
    neuron_data, classes = load_swc_directory(swc_directory)
    step_functions = compute_descriptors(neuron_data)
    distance_matrices = compute_distances(step_functions)

    if method == 'combine':
        combined = combine(distance_matrices, weights=weights)
        extra = {}
    elif method == 'gridsearch':
        best_weights, score, combined = grid_search(distance_matrices, classes,
                                                     resolution=resolution)
        extra = {'weights': best_weights, 'score': score}
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'combine' or 'gridsearch'.")

    clusters = cluster(combined, k=k)
    return {
        'neurons': neuron_data,
        'classes': classes,
        'descriptors': step_functions,
        'distances': distance_matrices,
        'combined': combined,
        'clusters': clusters,
        **extra,
    }
