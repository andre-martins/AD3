import pytest
import numpy as np

from ad3 import PFactorGraph
from ad3.extensions import PFactorTree


def test_tree_factor():
    n_nodes = 10
    rng = np.random.RandomState(0)
    g = PFactorGraph()
    arcs = [(h, m) for m in range(1, n_nodes) for h in range(n_nodes)
            if h != m]
    potentials = rng.uniform(0, 1, size=len(arcs))
    arc_vars = [g.create_binary_variable() for _ in arcs]

    for var, potential in zip(arc_vars, potentials):
        var.set_log_potential(potential)

    tree = PFactorTree()
    g.declare_factor(tree, arc_vars)
    tree.initialize(n_nodes, arcs)

    _, posteriors, _, _ = g.solve()
    chosen_arcs = [arc for arc, post in zip(arcs, posteriors)
                   if post > 0.99]

    # check that it's a tree
    selected_nodes = set(a for arc in chosen_arcs for a in arc)
    assert selected_nodes == set(range(n_nodes))

    marked = list(range(n_nodes))
    for h, t in chosen_arcs:
        assert marked[t] != marked[h]
        marked[t] = marked[h]


def test_tree_validate():
    g = PFactorGraph()
    n_nodes = 4
    arcs = [(h, m) for m in range(1, n_nodes) for h in range(n_nodes)
            if h != m]
    arc_vars = [g.create_binary_variable() for _ in arcs]

    tree = PFactorTree()
    g.declare_factor(tree, arc_vars)

    with pytest.raises(TypeError):
        tree.initialize(n_nodes, [-3 for _ in arcs])

    with pytest.raises(TypeError):
        tree.initialize(n_nodes, None)

    with pytest.raises(TypeError):
        tree.initialize(n_nodes, 42)

    with pytest.raises(ValueError):
        tree.initialize(n_nodes, [(100, 100) for _ in arcs])

    with pytest.raises(ValueError):
        tree.initialize(n_nodes, arcs + arcs)

    with pytest.raises(ValueError):
        tree.initialize(n_nodes, arcs[:3])
