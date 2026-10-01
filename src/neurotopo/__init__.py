"""
neurotopo: topological Sholl descriptors for neuron morphology.

    import neurotopo as nt

    neurons, classes = nt.load_swc_directory("path/to/swc_root")
    step_functions = nt.compute_descriptors(neurons)
    distances = nt.compute_distances(step_functions)
    combined = nt.combine(distances)
    clusters = nt.cluster(combined, k=2)

or, in one call:

    result = nt.run("path/to/swc_root", k=2)

Method settings (which SWC types to strip, whether to normalise radii, the
distance metric) live in `neurotopo.config` and can be changed before calling
in:

    nt.config.REMOVE_TYPES = ['axon', 'apical']

Method: Khalil R, Kallel S, Farhat A, Dlotko P. Topological Sholl descriptors
for neuronal clustering and classification. PLoS Comput Biol.
2022 Jun 22;18(6):e1010229.
"""

__version__ = "0.1.0"

from . import config
from .neuron_processor import process_neuron, process_swc_directory
from .descriptors import (
    calculate_tortuosity,
    calculate_branching_pattern,
    calculate_wiring_descriptor,
    calculate_flux,
    calculate_leaf,
    calculate_spread_descriptor,
    calculate_energy,
    calculate_polarity,
)
from .distances import (
    compute_pairwise_differences,
    grid_search_optimal_weights,
    combine_matrices_with_weights,
)
from .utils import calculate_l1_difference
from .pipeline import (
    DESCRIPTORS,
    load_swc_directory,
    compute_descriptors,
    compute_distances,
    combine,
    grid_search,
    cluster,
    run,
)

__all__ = [
    "config",
    "process_neuron", "process_swc_directory",
    "calculate_tortuosity", "calculate_branching_pattern",
    "calculate_wiring_descriptor", "calculate_flux", "calculate_leaf",
    "calculate_spread_descriptor", "calculate_energy", "calculate_polarity",
    "compute_pairwise_differences", "grid_search_optimal_weights",
    "combine_matrices_with_weights", "calculate_l1_difference",
    "DESCRIPTORS", "load_swc_directory", "compute_descriptors",
    "compute_distances", "combine", "grid_search", "cluster", "run",
]
