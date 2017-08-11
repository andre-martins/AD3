import numpy as np
from ad3 import factor_graph as fg


def test_solve():
    rng = np.random.RandomState(0)
    graph = fg.PFactorGraph()

    a = graph.create_multi_variable(3)
    b = graph.create_multi_variable(3)
    c = graph.create_multi_variable(3)

    a.set_log_potentials(rng.randn(3))
    b.set_log_potentials(rng.randn(3))
    c.set_log_potentials(rng.randn(3))

    graph.create_factor_dense([a, b], rng.randn(3 * 3))
    graph.create_factor_dense([a, c], rng.randn(3 * 3))
    graph.create_factor_dense([b, c], rng.randn(3 * 3))

    val, _, _, status = graph.solve()
    assert status == 'integral'

    val_one_iter, _, _, status_one_iter = graph.solve(max_iter=1)
    assert status_one_iter == 'unsolved'
    assert val_one_iter < val

    val_lowtol, _, _, status_lowtol = graph.solve(tol=0.3)
    assert status_lowtol == 'fractional'
    assert val_lowtol > val

    val_lowtol_bb, _, _, status_lowtol_bb = graph.solve(tol=0.3,
                                                        branch_and_bound=True)
    assert status_lowtol_bb == 'integral'
    assert (val_lowtol_bb - val) ** 2 < 1e-8
