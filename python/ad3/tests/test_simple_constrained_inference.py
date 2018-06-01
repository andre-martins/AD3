# Author: Jean-Luc Meunier, 30 Jan 2017

import numpy as np
import pytest
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)


from ad3.simple_constrained_inference import (
    general_constrained_graph_singletype,
    general_constrained_graph_multitype)

def test_general_constrained_graph_singletype():

    unaries = np.array([[10, 11, 0],
                        [1000, 1100, 1200]], dtype=np.float64)
    edges = np.array([[0, 1]])
    edge_weights = np.array([[[.00, .01, .02],
                              [.10, .11, .12],
                              [0, 0, 0]]
                            ], dtype=np.float64)

    ret = general_constrained_graph_singletype(unaries, edges, edge_weights, None)
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals, np.array([[0, 1, 0],
                                                   [0, 0, 1]]), 5)
    assert_array_almost_equal(edge_marginals.ravel(),
                              np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]), 5)
    assert solver_status == 'integral'


def test_general_constrained_graph_mulitype():

    empty = np.zeros((0, 0))
    l_n_nodes  = [1, 1]
    l_n_states = [2, 3]

    unaries = [np.array([[10, 11]]), np.array([[ 1000, 1100, 1200]])]
    edges = [empty, np.array([[0, 0]]), empty, empty]
    edge_weights = [empty,
                    np.array([[[.00, .01, .02],
                               [.10, .11, .12]]]),
                    empty,
                    empty]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, None)

    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[0, 1]]))
    assert_array_almost_equal(marginals[1], np.array([[0, 0, 1]]))

    assert_array_almost_equal(edge_marginals[0], np.zeros((0, 4)))
    assert_array_almost_equal(edge_marginals[1].reshape(2, 3),
                              np.array([[0, 0, 0],
                                        [0, 0, 1]]), 5)
    assert_array_almost_equal(edge_marginals[2], np.zeros((0, 6)))
    assert_array_almost_equal(edge_marginals[3], np.zeros((0, 9)))

    #a dding some constraints
    c1 = ["OR", [[0], [0]],    # between 0 of typ0 and 0 of typ1
                [[1], [1]],    # between state 1 ans state1
                [False, False]
          ]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, [c1])
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[0, 1]]), 5)
    assert_array_almost_equal(marginals[1], np.array([[0, 0, 1]]), 5)

    c2 = ["OROUT", [[0], [0]],    # between 0 of typ0 and 0 of typ1
                   [[1], [1]],    # between state 1 ans state1
                   [False, False]
          ]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, [c2])
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[1,  0]])   ,5)
    assert_array_almost_equal(marginals[1], np.array([[0, 0, 1]])  ,5)

    c3 = ["OROUT", [[0], [0]],    # between 0 of typ0 and 0 of typ1
                   [[0], [0]],    # between state 0 ans state0
                   [False, False]
          ]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights,
                                              [c2, c3])
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal( marginals[0], np.array([[0, 1]]), 5)
    assert_array_almost_equal( marginals[1], np.array([[0, 1, 0]]), 5)

    c3last = ["OROUT", [[0], []],    # between 0 of typ0 and 0 of typ1
                       [[0], []],    # between state 0 ans state0
                       [False],

                       [1, 0, 0, False] # same with a different way of
                                        # indicating the last variable of the
                                        # logic operator
                                        # type 1, node 0, state 0, not negated
          ]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights,
                                              [c2, c3last])
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[0, 1]]), 5)
    assert_array_almost_equal(marginals[1], np.array([[0, 1, 0]]), 5)


    c41 = ["XOR",  [[0], [0]],    #between 0 of typ0 and 0 of typ1
                   [[0], [0]],
                   [False, False]
          ]
    c42 = ["XOR",  [[0], [0]],    #between 0 of typ0 and 0 of typ1
                   [[1], [1]],
                   [False, False]
          ]
    c43 = ["XOR",  [[0], [0]],    #between 0 of typ0 and 0 of typ1
                   [[2], [2]],
                   [False, False]
          ]
    c44 = ["XOR",  [[0], [0]],    #between 0 of typ0 and 0 of typ1
                   [[1], [2]],
                   [False, False]
          ]

    with pytest.raises(ValueError):
        general_constrained_graph_multitype(unaries, edges, edge_weights,
                                            [c41, c42, c43])

    ret = general_constrained_graph_multitype(unaries, edges, edge_weights,
                                              [c41, c42, c44])
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[0, 1]]), 5)
    assert_array_almost_equal(marginals[1], np.array([[1, 0, 0]]), 5)

    #adding XOR between each state of node 0 and each state of node 1 => NOT SOLVABLE
    lc = [ ("XOR", [[0], [0]], [[i], [j]], [False, False]) for i in range(2) for j in range(3)]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, lc, verbose=1)
    marginals, edge_marginals, value, solver_status = ret
    assert solver_status == "unsolved"

    #forcing one label on node 0
    lc = [("OR", [[0], []], [[0], []], [ False, False ])]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, lc)
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[1, 0]]), 5)
    assert_array_almost_equal(marginals[1], np.array([[0, 0, 1]]), 5)
    lc = [ ("OR", [[0], [ ]], [[0], [ ]], [False, False]),
           ("OR", [[ ], [0]], [[ ], [1]], [False, False])]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, lc)

    lc = [ ("OR", [[0], [ ]], [[0], [ ]], [False, False]),
           ("OR", [[ ], [0]], [[ ], [0]], [False, False])]
    ret = general_constrained_graph_multitype(unaries, edges, edge_weights, lc)
    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[1, 0]]), 5)
    assert_array_almost_equal(marginals[1], np.array([[1, 0, 0]]), 5)

