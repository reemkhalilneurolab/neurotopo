"""
Method and output settings for the topological Sholl descriptor package.

Descriptor method: Khalil R, Kallel S, Farhat A, Dlotko P. Topological Sholl
descriptors for neuronal clustering and classification. PLoS Comput Biol.
2022 Jun 22;18(6):e1010229.

This module holds mutable module-level settings, in the same spirit as
matplotlib's rcParams: import it and change an attribute before calling into
the package, and every function that reads that attribute picks up the change.

    import neurotopo as nt
    nt.config.REMOVE_TYPES = ['axon', 'apical']
    nt.config.PLOT_STEP_FUNCTION = True

Nothing here touches the filesystem on import. Output directories are created
lazily, immediately before something is written to them.
"""

import os


# ===========================================================================
# METHOD  --  changes here change the numbers
# ===========================================================================

# SWC component types to strip before any descriptor is computed.
#   0 undefined, 1 soma, 2 axon, 3 basal dendrite, 4 apical dendrite, 5+ custom
# The descriptors characterise the dendritic tree, so the axon is removed by
# default.
REMOVE_TYPES = ['axon']

# Map each neuron's radial coordinate onto [0, 1] by dividing by its own
# maximum distance from the soma, so descriptors compare shape rather than
# size.
NORMALIZE_DESCRIPTOR_RADII = True
NORMALIZE = True

# Distance between two step functions. 'l1' is the L1 distance of equation (4)
# in the reference above.
DISTANCE_METRIC = 'l1'

# Reserved by the combination code; the empty string selects its default.
DISTANCE_MATRICES_COMBINATION_MODE = ''

LINKAGE_METHOD = 'ward'
FEATURE_NORMALIZATION_METHOD = 'min-max'   # 'z-score' or 'none'

# Descriptors to skip when computing all of them at once. 'Polarity' is not a
# step function and is excluded by default; the L1 pipeline does not use it.
EXCLUDED_DESCRIPTORS = ['Polarity']

# Neurons to drop by filename stem, e.g. reconstructions that fail quality
# control. Leave empty to use every file found.
EXCLUDED_NEURONS = []


# ===========================================================================
# OUTPUT
# ===========================================================================

# Where plots and pickles go if you use the plotting/saving helpers without
# specifying a directory explicitly. Relative to the current working
# directory, and created only when something is actually written there.
SAVE_DIRECTORY = os.path.join(os.getcwd(), 'neurotopo_output')
PLOTS_DIRECTORY = os.path.join(SAVE_DIRECTORY, 'plots')
PICKLE_DIRECTORY = os.path.join(SAVE_DIRECTORY, 'pickles')
DESCRIPTORS_DIRECTORY = os.path.join(PLOTS_DIRECTORY, 'Descriptors')

# Directory to read SWC files from, used by process_swc_directory(). Prefer
# neurotopo.load_swc_directory(path) instead, which takes the path as an
# argument and does not need this set.
BASE_DIRECTORY = None

# Per-neuron step function plots. Useful for inspection, but writes one PNG
# per neuron per descriptor and dominates the runtime on a large dataset.
PLOT_STEP_FUNCTION = False
PLOT_NEURON = False
PLOT_NEURON_3D = False
PLOT_POLARITY = False

# SWC type to draw when a neuron is plotted; None draws all retained types.
PLOT_SWC_FILTER_TYPE = 2

# Class colours, cycled if a dataset has more classes than colours.
COLORS_PALETTE = ['#f77189', '#36ada4', '#7BC8F6', '#C79FEF', '#004d40', '#846b54']
# Misspelled alias: some plotting helpers read COLORS_PALLETE. Kept pointing
# at the same list so the two can never disagree.
COLORS_PALLETE = COLORS_PALETTE
