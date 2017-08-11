# Author: Vlad Niculae <vlad@vene.ro>
# License: GNU LGPL v3

import numpy as np
from numpy.testing import assert_array_equal

from ad3 import factor_graph as fg


def test_sequence_dense():

    n_states = 3
    transition = np.eye(n_states).ravel()
    graph = fg.PFactorGraph()

    vars_expected = [0, 1, None, None, 1]
    variables = [graph.create_multi_variable(n_states) for _ in vars_expected]
    for prev, curr in zip(variables, variables[1:]):
        graph.create_factor_dense([prev, curr], transition)
    for var, ix in zip(variables, vars_expected):
        if ix is not None:
            var.set_log_potential(ix, 1)

    value, marginals, additionals, status = graph.solve()
    # 3 points for "observed" values, 3 points for consecutive equal vals
    assert value == 6

    expected = [0, 1, 1, 1, 1]
    obtained = np.array(marginals).reshape(5, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)
