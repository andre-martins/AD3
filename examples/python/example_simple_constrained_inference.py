# -*- coding: utf-8 -*-
"""
An example of use of the simple_constrained_inference module.

It also shows that null constraint produces same result as simple example.

JL Meunier
Dec 2017
"""

from __future__ import print_function
import numpy as np
from ad3.simple_constrained_inference import general_constrained_graph
from example_simple_inference import get_grid_label
from example_simple_inference import prepare_single_type
from example_simple_inference import prepare_multi_type

NO_CONSTRAINT = []

print("--- SINGLE TYPE ---")

unaries, edges, edge_weights = prepare_single_type()
marginals, edge_marginals, _, _ = general_constrained_graph(unaries,
                                                            edges,
                                                            edge_weights,
                                                            NO_CONSTRAINT,
                                                            exact=True
                                                            )
labels = get_grid_label(2, 3, marginals)
assert labels.tolist() == [[3, 4, 5],
                           [4, 5, 4]]

# let's say AT_MOST_ONE I for I in 0..6 :-)
constraints = [('ATMOSTONE',            # operator
                [0, 1, 2, 3, 4, 5],     # nodes
                [I, I, I, I, I, I],     # respective state of node
                None) for I in range(6)]
marginals, edge_marginals, _, _ = general_constrained_graph(unaries,
                                                            edges,
                                                            edge_weights,
                                                            constraints,
                                                            exact=True
                                                            )
labels = get_grid_label(2, 3, marginals)
print("--->")
print(labels)

assert labels.tolist() == [[3, 4, 5],
                           [2, 1, 0]]

# ===========================================================================
print("--- MULTI TYPE ---")

# ========   TEST  ========
lUnary, lEdges, lEdgeWeights = prepare_multi_type()
marginals, edge_marginals, _, _ = general_constrained_graph(lUnary,
                                                            lEdges,
                                                            lEdgeWeights,
                                                            NO_CONSTRAINT,
                                                            exact=True
                                                            )
labels_N = get_grid_label(1, 3, marginals[0])
labels_M = get_grid_label(1, 3, marginals[1])
assert labels_N.tolist() == [[3, 2, 3]]
assert labels_M.tolist() == [[2, 1, 2]]


# let's say AT_MOST_ONE I for I in 0..6 for node of type N
# let's say AT_MOST_ONE I for I in 0..3 for node of type M
constraints = [('ATMOSTONE',                # operator
                [[0, 1, 2], []],   # nodes N
                [[I, I, I], []],   # respective state of node
                [None, None]) for I in range(6)]
constraints += [('ATMOSTONE',                # operator
                [[], [0, 1, 2]],   # nodes M
                [[], [I, I, I]],   # respective state of node
                [None, None]) for I in range(3)]

marginals, edge_marginals, _, _ = general_constrained_graph(lUnary,
                                                            lEdges,
                                                            lEdgeWeights,
                                                            constraints,
                                                            exact=True
                                                            )
print("--->")
labels_N = get_grid_label(1, 3, marginals[0])
print(labels_N)
labels_M = get_grid_label(1, 3, marginals[1])
print(labels_M)
# it is better to have N1 labeled 2 to get the +1 bonus from the vertical edge
# it is better to have N2 labeled 3 to get the +1 bonus from the vertical edge
assert labels_N.tolist() == [[3, 2, 1]]
assert labels_M.tolist() == [[2, 1, 0]]

# ========   TEST  ========
unaries_N, unaries_M = lUnary
unaries_N[1, 4] = 2.0  # pushing HARDER for node N 1 being label 4

# inference will prefer gaining 2.0 to losing .2 for gaining 1 (the edge)
marginals, edge_marginals, _, _ = general_constrained_graph(lUnary,
                                                            lEdges,
                                                            lEdgeWeights,
                                                            NO_CONSTRAINT,
                                                            exact=True
                                                            )
labels_N = get_grid_label(1, 3, marginals[0])
labels_M = get_grid_label(1, 3, marginals[1])
assert labels_N.tolist() == [[3, 4, 3]]
assert labels_M.tolist() == [[2, 1, 2]]
