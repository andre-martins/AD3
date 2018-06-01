# Author: Jean-Luc Meunier, 30 Jan 2017

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)

from ad3 import general_graph

def test_general_graph():
    unaries = np.array([[10, 11, 0 ],
                        [ 1000, 1100, 1200]], dtype=np.float64)
    edges = np.array([[0, 1]])
    edge_weights = np.array([[[.00, .01, .02],
                              [.10, .11, .12],
                              [0, 0, 0]]], dtype=np.float64)
    ret = general_graph(unaries, edges, edge_weights, verbose=1, exact=False)
    marginals, edge_marginals, value, solver_status = ret
    assert (marginals == np.array([[ 0.,  1.,  0.],
                                   [ 0.,  0.,  1.]])).all()
    assert (edge_marginals == np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]])).all()
    assert solver_status == 'integral'

def test_general_graph_multitype():
    empty = np.zeros((0, 0))

    unaries = [np.array([[10, 11]]),
               np.array([[1000, 1100, 1200]])]
    edges = [empty, np.array([[0, 0]]), empty, empty]
    edge_weights = [empty,
                    np.array([[[.00, .01, .02],
                               [.10, .11, .12]]]),
                    empty,
                    empty]

    ret = general_graph(unaries, edges, edge_weights, verbose=1)

    marginals, edge_marginals, value, solver_status = ret
    assert_array_almost_equal(marginals[0], np.array([[0, 1]]))
    assert_array_almost_equal(marginals[1], np.array([[0, 0, 1]]))

    assert_array_almost_equal(edge_marginals[0], np.zeros((0, 4)))
    assert_array_almost_equal(edge_marginals[1].reshape(2, 3),
                              np.array([[0, 0, 0],
                                        [0, 0, 1]]), 5)
    assert_array_almost_equal(edge_marginals[2], np.zeros((0, 6)))
    assert_array_almost_equal(edge_marginals[3], np.zeros((0, 9)))

