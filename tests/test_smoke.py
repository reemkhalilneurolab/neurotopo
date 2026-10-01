"""
End-to-end smoke test on tiny synthetic reconstructions.

Not a validation of the descriptor mathematics (that lives with the paper
this method comes from); this only confirms that a fresh install runs the
whole pipeline, from SWC files on disk to a clustering, without error.
"""
import numpy as np
import pandas as pd
import pytest

import neurotopo as nt


def _write_swc(path, spread_out):
    """A tiny valid tree: soma, two primary dendrites, one of them bifurcating.

    `spread_out` scales the second primary branch, so two calls with
    different values produce geometrically different neurons.
    """
    scale = 3.0 if spread_out else 1.0
    lines = [
        "1 1 0 0 0 5 -1",
        "2 3 0 0 5 1 1",
        "3 3 0 0 10 1 2",
        f"4 3 {5*scale} {3*scale} 15 1 3",
        f"5 3 {-5*scale} {-3*scale} 15 1 3",
        "6 3 10 0 0 1 1",
        f"7 3 {15*scale} 5 -5 1 6",
    ]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def swc_root(tmp_path):
    group_a = tmp_path / "GroupA"
    group_b = tmp_path / "GroupB"
    group_a.mkdir()
    group_b.mkdir()
    for i in range(3):
        _write_swc(group_a / f"a{i}.swc", spread_out=False)
        _write_swc(group_b / f"b{i}.swc", spread_out=True)
    return tmp_path


def test_load_swc_directory_reads_classes(swc_root):
    neurons, classes = nt.load_swc_directory(swc_root)
    assert len(neurons) == 6
    assert set(classes["class"]) == {"GroupA", "GroupB"}
    for data in neurons.values():
        assert "swc_df" in data and "neuron_as_graph" in data


def test_descriptors_are_step_functions_on_unit_interval(swc_root):
    neurons, _ = nt.load_swc_directory(swc_root)
    step_functions = nt.compute_descriptors(neurons)

    assert set(step_functions) == set(nt.DESCRIPTORS)
    for name, per_neuron in step_functions.items():
        assert set(per_neuron) == set(neurons)
        for fn in per_neuron.values():
            radii = [r for r, _ in fn]
            assert min(radii) >= -1e-9
            assert max(radii) <= 1 + 1e-9


def test_distances_combine_and_cluster(swc_root):
    neurons, classes = nt.load_swc_directory(swc_root)
    step_functions = nt.compute_descriptors(neurons)
    distances = nt.compute_distances(step_functions)

    assert set(distances) == set(nt.DESCRIPTORS)
    for matrix in distances.values():
        assert matrix.shape == (6, 6)
        assert np.allclose(np.diag(matrix.values), 0)

    combined = nt.combine(distances)
    assert combined.shape == (6, 6)

    labels = nt.cluster(combined, k=2)
    assert set(labels.unique()) == {1, 2}
    assert list(labels.index) == list(combined.index)


def test_run_one_call_pipeline(swc_root):
    result = nt.run(swc_root, k=2)
    for key in ("neurons", "classes", "descriptors", "distances", "combined", "clusters"):
        assert key in result
    assert len(result["clusters"]) == 6


def test_run_with_gridsearch_method(swc_root):
    result = nt.run(swc_root, k=2, method="gridsearch", resolution=5)
    assert set(result["weights"]) == set(nt.DESCRIPTORS)
    assert isinstance(result["score"], float)
    assert len(result["clusters"]) == 6

    with pytest.raises(ValueError):
        nt.run(swc_root, method="not-a-real-method")


def test_grid_search_runs(swc_root):
    neurons, classes = nt.load_swc_directory(swc_root)
    distances = nt.compute_distances(nt.compute_descriptors(neurons))
    weights, score, combined = nt.grid_search(distances, classes, resolution=5)
    assert set(weights) == set(nt.DESCRIPTORS)
    assert combined.shape == (6, 6)
