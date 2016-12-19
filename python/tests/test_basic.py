# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_almost_equal, assert_greater

from ad3 import factor_graph as fg


def _solve(graph, eta=0.1, adapt=True, max_iter=100, verbose=0):
    graph.set_verbosity(verbose)
    graph.set_eta_ad3(eta)
    graph.adapt_eta_ad3(adapt)
    graph.set_max_iterations_ad3(max_iter)
    return graph.solve_lp_map_ad3()


def test_instantiate():
    graph = fg.PFactorGraph()
    graph.create_binary_variable()


def test_sequence_dense():

    n_states = 3
    transition = np.eye(n_states).ravel()
    graph = fg.PFactorGraph()

    vars_expected = [0, 1, None, None, 1]
    vars = [graph.create_multi_variable(n_states) for _ in vars_expected]
    factors = [graph.create_factor_dense([prev, curr], transition)
               for prev, curr in zip(vars, vars[1:])]
    for var, ix in zip(vars, vars_expected):
        if ix is not None:
            var.set_log_potential(ix, 1)

    value, marginals, additionals, status = _solve(graph)
    # 3 points for "observed" values, 3 points for consecutive equal vals
    assert_almost_equal(value, 6)

    expected = [0, 1, 1, 1, 1]
    obtained = np.array(marginals).reshape(5, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)


def test_logic_constraints():
    graph = fg.PFactorGraph()
    n_states = 3

    var_a = graph.create_multi_variable(n_states)
    var_b = graph.create_multi_variable(n_states)

    var_a.set_log_potential(0, 1)  # A = 0
    var_b.set_log_potential(1, 10) # B = likely 1, possibly 2
    var_b.set_log_potential(2, 9.5)

    graph.fix_multi_variables_without_factors()

    value_unconstrained, marginals, _, _ = _solve(graph)

    expected = [0, 1]
    obtained = np.array(marginals).reshape(2, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)

    # add logic constraint: (A = 0) -> not (B = 1)
    graph.create_factor_logic('IMPLY', [var_a.get_state(0),
                                        var_b.get_state(1)],
                              [False, True])

    graph.fix_multi_variables_without_factors()  # need to run this again

    value_constrained, marginals, _, _ = _solve(graph)

    expected = [0, 2]
    obtained = np.array(marginals).reshape(2, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)

    # total score must be lower
    assert_greater(value_unconstrained, value_constrained)
